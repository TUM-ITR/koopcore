import jax
import jax.numpy as jnp

from typing import Tuple, Callable
from jaxtyping import Num, Array
from functools import partial


def normalize_eigenfunctions_by_operator(L):
    # only relevant in the multioutput case
    return jnp.einsum("DH, DE, ES->DHES", L, jnp.einsum("DH, ES->DE", L, L.conj())**-1, L.conj())


def normalize_eigenfunctions_by_steps(L): return L / L.shape[1]


def normalize_eigenfunctions_by_norm(
    L): return L / jnp.linalg.norm(L, axis=1)[:, None]


def eigenfunction_rollout(phi0: Num[Array, "d N D"] = None, eigvals_dt=None, timestamps=None, einsum_kw={"optimize": True}):
    M_ef_flow = jnp.power(eigvals_dt[None, :], timestamps[:, None])
    return jnp.einsum("HD, dND->dNHD", M_ef_flow, phi0, **einsum_kw)


def fit_invariant_weights(
        state,
        target,
        koopman_kernel_armin,
        reg=1e-8):
    N, H, d_target = target.shape
    tensor_gramian_armin = koopman_kernel_armin(state, state)
    target = jnp.reshape(target, [-1, d_target], order="F")
    lstsq_res = jnp.linalg.lstsq(
        (
            tensor_gramian_armin.reshape(H * N, H * N)
            + 
            reg * (H * N) * jnp.eye(H * N)
        ),
        target,
        rcond=reg* (H * N),
    )
    weights = lstsq_res[0]
    residual = (tensor_gramian_armin.reshape(
        H * N, H * N) @ weights).real - target
    return weights, jnp.atleast_1d(residual)


def fit_isometric_weights(
    Phi0: Num[Array, "D N0 d"],
    X0: Num[Array, "N0 d"],
    X_new: Num[Array, "N d"],
    new_kernel: callable,
    reg: float = 0.,
) -> Tuple[Num[Array, "N D"], Num[Array, "N0 N"]]:
    _, N0, _ = Phi0.shape
    N, _ = X_new.shape
    Phi0_flat = Phi0.transpose(1, 0, 2).reshape(N0, -1)
    gramian = new_kernel(X0, X_new)
    lstsq_res = jnp.linalg.lstsq(
        gramian 
        + 
        reg * min(N0, N) * jnp.eye(N0, N),
        Phi0_flat
        ,
        rcond=reg * min(N0, N))
    weights = lstsq_res[0]
    residuals = (gramian @ weights) - Phi0_flat

    return weights, residuals


def invariant_predictor_naive(
    X_query: Num[Array, "N d"],
    X_b: Num[Array, "M d"],
    weights=None,
    eigvals_dt=None,
    timestamps=None,
    koopman_kernel_extract=None,
    einsum_kw={"optimize": True}
) -> Tuple[Num[Array, "d N H D"], Num[Array, "d N H"]]:
    MH, d = weights.shape
    D = eigvals_dt.shape[0]
    N = X_query.shape[0]
    phi0 = jnp.einsum(
        "DNi, id->dND",
        koopman_kernel_extract(X_query, X_b).reshape(D, N, MH),
        weights,
        **einsum_kw
    )
    ef_traj = eigenfunction_rollout(
        phi0, eigvals_dt, timestamps)
    obs_traj = jnp.einsum("dNHD->NHd", ef_traj)
    return ef_traj, obs_traj


def invariant_predictor(
    X_query: Num[Array, "N d"] = None,
    X_base: Num[Array, "M d"] = None,
    weights: Num[Array, "M dD"] = None,
    eigvals_dt=None,
    timestamps=None,
    koopman_kernel_coreg=None,
    einsum_kw={"optimize": True}
) -> Tuple[Num[Array, "d N H D"], Num[Array, "d N H"]]:
    MH, d = weights.shape
    D = eigvals_dt.shape[0]
    S = timestamps.shape[0]
    H = S
    M = MH // H
    M_ef_flow = jnp.power(
        eigvals_dt[None, :], timestamps[:, None])
    N = X_query.shape[0]
    ef_traj = jnp.einsum(
        "DNM, SD, SMd, HD->dNHD",
        koopman_kernel_coreg(X_query, X_base).reshape(D, N, M),
        M_ef_flow.conj(),
        weights.reshape(S, M, d),
        M_ef_flow,
        **einsum_kw
    )
    return ef_traj, jnp.einsum("dNHD->NHd", ef_traj)


def isometric_predictor_naive(
    X_query: Num[Array, "N d"],
    X_base: Num[Array, "N d"] = None,
    weights: Num[Array, "N dD"] = None,
    eigvals_dt: Num[Array, "H D"] = None,
    timestamps: Num[Array, "H"] = None,
    kernel=None,
    einsum_kw={"optimize": True}
) -> Tuple[Num[Array, "d N H D"], Num[Array, "d N H D"]]:
    M, Dd = weights.shape
    D = eigvals_dt.shape[0]
    d = Dd // D

    phi0 = jnp.einsum(
        "NM, ME->NE",
        kernel(X_query, X_base),
        weights,
        **einsum_kw
    ).reshape(-1, d, D).transpose(1, 0, 2)
    ef_traj = eigenfunction_rollout(
        phi0, eigvals_dt, timestamps)
    return ef_traj, jnp.einsum("dNHD->NHd", ef_traj)


def isometric_predictor_with_eigenfunctions(
    X_query: Num[Array, "N d"],
    X_base: Num[Array, "N d"] = None,
    weights=None,
    eigvals_dt=None,
    timestamps=None,
    kernel=None,
    einsum_kw={"optimize": True}
) -> Tuple[Num[Array, "d N H D"], Num[Array, "d N H D"]]:
    M, dD = weights.shape
    D = eigvals_dt.shape[0]
    d = dD // D
    M_ef_flow = jnp.power(
        eigvals_dt[None, :], timestamps[:, None])

    N = X_query.shape[0]
    ef_traj = jnp.einsum(
        "NM, MdD, HD->dNHD",
        kernel(X_query, X_base),
        weights.reshape(M, d, D),
        M_ef_flow,
        **einsum_kw
    )
    return ef_traj, jnp.einsum("dNHD->NHd", ef_traj)


def isometric_predictor(
    X_query: Num[Array, "N d"] = None,
    X_base: Num[Array, "N d"] = None,
    weights: Num[Array, "N dD"] = None,
    eigvals_dt: Num[Array, "D"] = None,
    timestamps: Num[Array, "H"] = None,
    kernel=None,
    einsum_kw={"optimize": True}
) -> Tuple[Num[Array, "d N H D"], Num[Array, "d N H D"]]:
    M, dD = weights.shape
    D = eigvals_dt.shape[0]
    d = dD // D
    M_ef_flow = jnp.power(
        eigvals_dt[None, :], timestamps[:, None])

    N = X_query.shape[0]
    obs_traj = jnp.einsum(
        "NM, MdD, HD->NHd",
        kernel(X_query, X_base),
        weights.reshape(M, d, D),
        M_ef_flow,
        **einsum_kw
    )
    return obs_traj
