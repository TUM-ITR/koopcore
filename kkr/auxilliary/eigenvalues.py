import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
Array = Any


def convert_2d_to_complex(data: Array):
    eigvals = data[:, 0] + data[:, 1] * 1j
    return eigvals


def convert_dt_to_ct(e, dt=1.):
    return np.log(e) / dt


def convert_ct_to_dt(e, dt=1.):
    return np.exp(e * dt)


def construct_multiindices(
    n: int, p_bar: int, verbose: bool = False
):
    from kkr.auxilliary.monomials import mono_between_next_grevlex
    from scipy.special import binom
    """Generates multiindices based on the monomial toolbox by John Burkardt.
    https://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html (09.05.2022)

    Args:
        n (int): immediate state dimension
        p_bar (int): maximum degree
        verbose (bool): print?

    Returns:
        tuple(list[npt.ArrayLike], 'list[str]'): multiindices for each subspace
    """
    if verbose:
        print(
            f"going to construct an {binom(p_bar+n, n)-1} dimensional monomial space\n"
        )
    list_len_multiindices = [int(binom(Ni + n - 1, Ni))
                             for Ni in range(1, p_bar + 1)]
    list_multiindices = [np.zeros([l, n]) for l in list_len_multiindices]
    for i, lli in enumerate(list_len_multiindices):
        p = i + 1
        list_multiindices[i][0][-1] = p
        for j in range(1, lli):
            list_multiindices[i][j] = mono_between_next_grevlex(
                n, p, p, list_multiindices[i][j - 1])
    list_multiindices = [np.flip(mii, axis=0) for mii in list_multiindices]
    if verbose:
        print(
            "dimension of each subspace with equal degree: ",
            list_len_multiindices,
            r"\n",
        )
        print("multiindices: ", np.concatenate(list_multiindices, 0), r"\n")
    return list_multiindices, list_len_multiindices
    # construct_multiindices


def make_lattice(eigvals, order, ct=True):
    eigvals = np.atleast_1d(np.asarray(eigvals))
    n = eigvals.shape[0]
    list_multiindices = np.concatenate(
        construct_multiindices(n, order, verbose=False)[0])
    if ct:
        lattice = np.array([np.sum(eigvals * mi) for mi in list_multiindices])
    else:
        lattice = np.array([np.prod(eigvals ** mi)
                           for mi in list_multiindices])
    return lattice, list_multiindices
