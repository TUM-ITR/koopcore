from os import times
import jax.numpy as jnp
import jax
Array = jax.Array

def koopman_kernel(
    base_kernel: callable,
    g_1: Array,
    g_2: Array,
    t_vec: Array,
    l_vec: Array,
) -> Array:
    """Creates a Koopman Invariance Kernel Tensor.

    Args:
        base_kernel (callable): _description_
        g_1 (Array): query trajectories
        g_2 (Array): base trajectories
        t_vec (Array): vector of times where invariance is imposed
        l_vec (Array): vector of eigenvalues to construct the Grammians for

    Returns:
        Array: kernel matrices stored in a (T x D x Nq x Nb)-tensor
    """

    N1, Hq, d = g_1.shape
    N2, Hb, _d = g_2.shape
    H = t_vec.shape[0]
    D = l_vec.shape[0]
    assert d == _d
    assert Hq >= H
    assert Hb >= H

    mu = jnp.exp(l_vec)
    time_exps = t_vec[0] - jnp.reshape(t_vec, (1, H))
    mu_array = mu.reshape(D, 1)
    mu_array = jnp.power(mu_array, time_exps)
    res = jnp.zeros((D, N1, N2))

    for iN in range(N1):
        for jN in range(iN, N2):
            _KXX = base_kernel(g_1[iN][:H], g_2[jN][:H])
            for iD in range(D):  # exploit structure to avoid similarity calculation
                _s = mu_array[iD].repeat(H).reshape(H, H)
                res = res.at[iD, iN, jN].set(
                    jnp.sum(_s * _KXX * _s.T.conj()) / H**2 / D
                )
                res = res.at[iD, jN, iN].set(res[iD, iN, jN])  # symmetry
    return res



def gramian_tensor_to_lstsq_matrix(koopman_gramian: jax.Array,
                                    dtype: jnp.dtype = jnp.float64) -> jax.Array:
    """Turns a Koopman gramian tensor from "koopman_kernel" into a matrix.

    Args:
        koopman_gramian (jax.Array): (T x D x Nq x Nb) gramian tensor
        dtype (jnp.dtype, optional): Datatype for numpy allocations. Defaults to jnp.float64.

    Returns:
        jax.Array: matrix of size (T*Nq x D*Nb)
    """
    T, D, Nq, Nb = koopman_gramian.shape
    koopman_matrix = jnp.transpose(koopman_gramian, (0, 2, 1, 3)
                                   ).reshape(T, Nq, -1).reshape(-1, D * Nb)
    return koopman_matrix


def make_vector_valued_kernel(base_kernel: callable, D, x_base):
    def vector_valued_kernel(xq, xb=x_base, t=None):
        if len(xq.shape) > 2:
            xq = xq[:, 0, :]
        k_xX = base_kernel(xq, xb)
        return jnp.expand_dims(k_xX, 0).repeat(D, 0)
    return vector_valued_kernel


def make_linear_trajectory_kernel(base_kernel_vv: callable, l_vec, t_vec):

    D = l_vec.shape[0]
    H = t_vec.shape[0]
    L_op = l_vec.reshape(1, D) * t_vec.reshape(-1, 1)
    L_op_exp = jnp.exp(L_op)

    def gramian(X_b, t=0.):
        base_gramians = jnp.squeeze(base_kernel_vv(X_b, X_b, t=0.), axis=0)
        return jnp.einsum("ti, inm, is->tnsm", L_op_exp, base_gramians, L_op_exp.conj().T)

    def query(X_q, X_b):
        base_gramians = jnp.squeeze(base_kernel_vv(X_q, X_b, t=0.), axis=0)
        D, Nq, Nb = base_gramians.shape
        return jnp.einsum("inm, is->nsm", base_gramians, L_op_exp.conj().T)

    def extract(X_q, X_b):
        base_gramians = jnp.squeeze(base_kernel_vv(X_q, X_b, t=0.), axis=0)
        return jnp.einsum("inm, is->insm", base_gramians, L_op_exp.conj().T)

    return gramian, query, extract
