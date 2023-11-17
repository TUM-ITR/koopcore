from typing import Tuple
import jax
import jax.numpy as jnp


def rk4(f: callable, x0, dt):
    k1 = f(x0)
    k2 = f(x0 + k1 * dt / 2)
    k3 = f(x0 + k2 * dt / 2)
    k4 = f(x0 + k3 * dt)
    return x0 + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


def discretize_RK4(f: callable, dt):
    def disc_f(x):
        return rk4(f, x, dt)
    return disc_f


def get_equilibrium_evals(X, dynsys_ct, k=5):
    dX0dt = dynsys_ct(X)
    v, ii = jax.lax.approx_min_k((dX0dt**2).sum(1), k)
    _j = jax.jacobian(dynsys_ct)
    return jnp.stack([jnp.linalg.eigvals(_j(X[i])) for i in ii])


# ==== 1d ====================================================================

# 1d bernoulli equation for n=3
# equilibria are at x_eq = { 0, sqrt(a/b), -sqrt(a/b) }
# thus a parametrization in terms of the nonzero equilibria is
# (p_a, p_eq) -> p_b = (p_a/ p_eq**2)
# using the dt variant should be beneficial, it is exact

params_ct_cubic_bernoulli_ubox = (4, -4)


def make_1d_ct_cubic_bernoulli(p_a, p_b=-1):

    def _contfcn(x):
        return p_a * x + p_b * x**3
    return jax.vmap(_contfcn, [0], 0)


def make_1d_dt_cubic_bernoulli(dt, p_a, p_b=-1):
    p_mu = jnp.exp(p_a * dt)

    def _fcn(x):
        y = (
            jnp.sign(x)
            *
            jnp.abs(
                (
                    (p_a * p_mu**2)
                    /
                    (-p_b * (p_mu**2 - 1) + p_a / x**2)
                )**0.5
            )
        )
        return y

    return jax.vmap(_fcn, in_axes=[0], out_axes=0, axis_name="N")


# ==== 2d ====================================================================
params_dt_polysys_unitbox = (0.6, 0.9)


def make_2d_dt_polysys(p_a, p_b):
    def _fcn(x):
        y = jnp.empty([2,])
        y = y.at[0].set(p_a * x[0])
        y = y.at[1].set(p_b * x[1] + (p_b - p_a**2) * x[0]**2)
        return y
    return jax.vmap(_fcn, [0], 0)


params_duffing_ct_optimal_construction = (0.5, 4., 0.5)


def make_2d_ct_duffing(p_a, p_b, p_c):
    def _contfcn(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        y0 = x1
        y1 = -p_a * x1 - x0 * (p_b * x0**2 - 1)
        return jnp.array([y0, y1]).T

    return _contfcn

params_2d_cubic_limit_cycle = (-0.1, -2, 2, -0.1)
def make_2d_cubic_limit_cycle(p_a, p_b, p_c, p_d):
    def _contfcn(x):
        y0 = p_a*x[:, 0]**3 + p_b*x[:, 1]**3
        y1 = p_c*x[:, 0]**3 + p_d*x[:, 1]**3
        return jnp.array([y0, y1]).T

    return _contfcn

params_limit_cycle_ct_ubox = (0.5, 0.3)


def make_2d_ct_limit_cycle(p_a, p_b):
    def _contfcn(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        y0 = -p_a * x1 + x0 * (p_b - x0**2 - x1**2)
        y1 = p_a * x0 + x1 * (p_b - x0**2 - x1**2)
        return jnp.array([y0, y1]).T
    return _contfcn


params_van_der_pol_optimal_construction = (2, 2, 5, 0.8)


def make_2d_ct_van_der_pol(p_a, p_mu, p_b, p_c):
    def _contfcn(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        y0 = p_a * x1
        y1 = p_mu * x1 * (1. - p_b * x0**2) - p_c * x0
        return jnp.array([y0, y1]).T
    return _contfcn


def make_van_der_pol_jac(p_a, p_mu, p_b, p_c):
    def jacobian(x):
        x0 = jnp.array(x[:, 0]).reshape(-1, 1, 1)
        x1 = jnp.array(x[:, 1]).reshape(-1, 1, 1)
        y00 = 0 * x0
        y01 = x0*0+p_a
        y10 = p_mu * x1 * p_b * 2 * x0 - p_c
        y11 = p_mu * (1. - p_b * x0**2)
        return jnp.concatenate([jnp.concatenate([y00, y01], axis=1), jnp.concatenate([y10, y11], axis=1)], axis=2)
    return jacobian
# ==== ND ====================================================================


def make_ND_linear_system(dt, A, cont=True):
    if cont:
        Ad = jax.scipy.linalg.expm(A * dt)
        def _contfcn(x): return jax.vmap(jnp.matmul, [None, 0], 0)(A, x)
        def _fcn(x): return jax.vmap(jnp.matmul, [None, 0], 0)(Ad, x)
    else:
        _contfcn = None
        def _fcn(x): return jax.vmap(jnp.matmul, [None, 0], 0)(A, x)

    return _fcn, _contfcn
