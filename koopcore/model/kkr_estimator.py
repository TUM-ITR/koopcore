import jax
import jax.numpy as jnp
from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from typing import Tuple, Any, Callable, Union
from jaxtyping import Num, Array
from copy import deepcopy

import koopcore
from koopcore.jax.invariant_kernels import make_koopman_kernel
from koopcore.jax.explicit_invariant_kernels import koopman_kernel as koopman_kernel_expl
from koopcore.jax.explicit_invariant_kernels import (
    make_linear_trajectory_kernel as make_linear_trajectory_kernel_expl,
)
from koopcore.auxilliary.data_classes import trajectory


class KoopmanKernelDTRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        kernel_name="square-exponential",
        kernel_params={"scale": 0.01},
        eigenvalues=jnp.array([1.0, 1.0j, -1.0j]),
        regularizer_invariant=1e-8,
        preprocessor=None,
        normalize_eigenfunctions="norm",
        einsum_kwargs={"optimize": True},
        timestep=1.0,
        regularizer_isometric=1e-8,
        device=jax.devices("cpu")[0],
        predictor_timestamps=None,
        inducing_points=None,
        invariant_weights=None,
        isometric_weights=None,
    ):
        self.kernel_name = kernel_name
        self.kernel_params = kernel_params
        self.regularizer_invariant = regularizer_invariant
        self.regularizer_isometric = regularizer_isometric
        self.preprocessor = preprocessor
        self.normalize_eigenfunctions = normalize_eigenfunctions
        self.device = jax.device_get(device)
        self.timestep = timestep
        self.einsum_kwargs = einsum_kwargs
        self.eigenvalues = eigenvalues
        self.predictor_timestamps = predictor_timestamps
        self.inducing_points = inducing_points
        self.invariant_weights = invariant_weights
        self.isometric_weights = isometric_weights

    def fit(self, X: trajectory, y: trajectory):
        f_normalize = self.get_eigenvalue_normalizer()

        if not hasattr(self, "base_kernel"):
            self.base_kernel = koopcore.jax.make_kernel(
                self.kernel_name, **self.kernel_params
            )
        X = deepcopy(X)
        if self.preprocessor:
            self.preprocessor.fit(X.X[:, 0, :])
            X.set_X(
                self.preprocessor.transform(X.X.reshape(-1, X.d)).reshape(X.N, X.H, X.d)
            )
        X_arr = X.X
        X0_arr = X_arr[:, 0, :]
        y_arr = y.X

        N, H, d = (X.N, X.H, X.d)
        timestamps_train = jnp.arange(H)

        if self.inducing_points is None:
            self.inducing_points = X0_arr

        koopman_kernel = make_koopman_kernel(
            self.base_kernel, self.eigenvalues, H, self.einsum_kwargs, f_normalize
        )

        @partial(jax.jit, device=self.device)
        def _fit_jax():
            # fit invariant model
            (
                invariant_weights,
                regression_residuals,
            ) = koopcore.jax.fit_invariant_weights(
                X_arr, y_arr, koopman_kernel["armin"], self.regularizer_invariant
            )

            Phi, _ = koopcore.jax.invariant_predictor(
                X_arr,
                X_arr,
                invariant_weights,
                self.eigenvalues,
                timestamps_train,
                koopman_kernel["coreg"],
            )  # D x N x H x d
            Phi0 = Phi[:, :, 0, :]
            # fit isometric model
            isometric_weights, regression_residuals = koopcore.jax.fit_isometric_weights(
                Phi0,
                X0_arr,
                self.inducing_points,
                self.base_kernel,
                self.regularizer_isometric,
            )

            return isometric_weights, invariant_weights, Phi0

        self.isometric_weights, self.invariant_weights, Phi0 = _fit_jax()
        return self

    def fit_explicit(self, x: trajectory, y: trajectory):
        if not hasattr(self, "base_kernel"):
            self.base_kernel = koopcore.jax.make_kernel(
                self.kernel_name, **self.kernel_params
            )
            
        f_normalize = self.get_eigenvalue_normalizer()
        if self.preprocessor:
            self.preprocessor.fit(x.X[:, 0, :])
            x.set_X(
                self.preprocessor.transform(x.X.reshape(-1, x.d)).reshape(x.N, x.H, x.d)
            )
        if self.inducing_points is None:
            self.inducing_points = x.X[:, 0, :]
        # invariant model
        N, H = (x.N, x.H)
        koopman_kernel = make_koopman_kernel(
            self.base_kernel, self.eigenvalues, H, self.einsum_kwargs, f_normalize
        )
        
        invariant_gramian = jax.jit(partial(koopcore.model.koopman_kernel_expl, self.base_kernel))(x.X, x.X, x.T[0], self.eigenvalues)
        # jitted_gramian = jax.jit(koopman_kernel["coreg"], device=jax.devices()[0])
        # invariant_gramian = jitted_gramian(x.X, x.X)
        L_op =jnp.power(self.eigenvalues[:, None], jnp.arange(H)[None, :])
        lstsq_gramian = jnp.einsum("it, inm, is->tnsm", L_op, invariant_gramian, L_op.conj())
        target = jnp.reshape(y.X, [-1, y.d], order="F")
        invariant_lstsq_res = jnp.linalg.lstsq(
            (
                lstsq_gramian.reshape(H * N, H * N)
                + 
                self.regularizer_invariant * (H * N) * jnp.eye(H * N)
            ),
            target,
            rcond=self.regularizer_invariant * (H * N),
        )
        self.invariant_weights = invariant_lstsq_res[0]
        extract_section = jnp.einsum("DNM, DH->DNHM", invariant_gramian, L_op.conj()) .reshape(self.D, N, H * N)
        Phi0 = jnp.einsum("DNA, Ad -> dND", extract_section, self.invariant_weights)
        # isometric model
        isometric_gramian = self.base_kernel(self.inducing_points, x.X[:, 0, :])
        isometric_lstsq_res = jnp.linalg.lstsq(
            (
                isometric_gramian 
                + 
                self.regularizer_isometric * (N) * jnp.eye(N)
            ),
            Phi0.transpose(1, 0, 2).reshape(N, -1),
            rcond=self.regularizer_isometric * (N)
            )
        self.isometric_weights = isometric_lstsq_res[0]

        return self

    def predict(self, X0: Array, timestamps: Union[Array, int]) -> trajectory:
        if not hasattr(self, "base_kernel"):
            self.base_kernel = koopcore.jax.make_kernel(
                self.kernel_name, **self.kernel_params
            )

        if type(timestamps) == int:
            timestamps = jnp.arange(timestamps + 1)
        if self.preprocessor:
            X0 = self.preprocessor.transform(X0)
        # construct linear predictor for a number of timesteps
        if (
            getattr(self, "predictor", None) is None
            or self.predictor_timestamps is not None
            and timestamps.shape == self.predictor_timestamps.shape
            and not jnp.allclose(timestamps, self.predictor_timestamps)
        ):
            self.predictor = jax.jit(
                partial(
                    koopcore.jax.isometric_predictor,
                    X_base=self.inducing_points,
                    weights=self.isometric_weights,
                    eigvals_dt=self.eigenvalues,
                    kernel=self.base_kernel,
                    einsum_kw=self.einsum_kwargs,
                    timestamps=timestamps,
                ),
                device=self.device,
            )
            self.predictor_timestamps = timestamps
        # evaluate
        return trajectory(self.predictor(X0), timestamps * self.timestep)

    def save(self, path):
        SAVE_ATTRIBUTES = [
            "kernel_name",
            "kernel_params",
            "eigenvalues",
            "regularizer_invariant",
            "preprocessor",
            "normalize_eigenfunctions",
            "einsum_kwargs",
            "timestep",
            "predictor_timestamps",
            "regularizer_isometric",
            "inducing_points",
            "invariant_weights",
            "isometric_weights",
        ]
        import numpy as np

        def jax_to_numpy(a):
            if type(a) == jax.Array:
                return np.array(a)
            else:
                return a

        save_dict = {
            name: jax_to_numpy(getattr(self, name)) for name in SAVE_ATTRIBUTES
        }
        from joblib import dump

        dump(save_dict, path)
        
    def load(self, path):
        import joblib, os
        self.__init__(**joblib.load(path))
        return self
        
    @property
    def D(self):
        return self.eigenvalues.shape[0]
    
    def get_eigenvalue_normalizer(self):
        if self.normalize_eigenfunctions == "norm":
            f_normalize = koopcore.jax.normalize_eigenfunctions_by_norm
        elif self.normalize_eigenfunctions == "step":
            f_normalize = koopcore.jax.normalize_eigenfunctions_by_steps
        elif self.normalize_eigenfunctions == "operator":
            f_normalize = koopcore.jax.normalize_eigenfunctions_by_operator
        else:
            raise NotImplementedError
        return f_normalize