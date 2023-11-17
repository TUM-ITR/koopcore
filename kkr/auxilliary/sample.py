# nopep8
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import kkr
from kkr.auxilliary.data_classes import trajectory

import jax.random as jr
from numpy import savez_compressed
import pandas as pd

from typing import Tuple, List
from jaxtyping import Array, Num
from functools import partial

import matplotlib.pyplot as plt

import os
# nopep8


def get_gamma(steps, x0, dynamics_fcn, dt=0.01, t0=None, dtype=jnp.float32) -> trajectory:
    n, d = x0.shape
    gamma = jnp.empty((steps + 1, n, d), dtype=dtype)
    gamma = gamma.at[0].set(x0)
    # for i in range(1, steps + 1):
    #     gamma = gamma.at[i].set(dynamics_fcn(gamma[i - 1]))

    def _bf(i, g):
        g = g.at[i].set(dynamics_fcn(g[i - 1]))
        return g
    gamma = jax.lax.fori_loop(1, steps + 1, _bf, gamma)
    traj_gamma = trajectory(
        jnp.transpose(gamma, (1, 0, 2)),
        jnp.arange(0, steps + 1, 1) * dt,
        n,
        d,
        steps + 1,
        dt,
        t0=t0)
    return traj_gamma


def get_gamma_ivp(steps, x0, dynamics_fcn_ct, dt=0.01, t0=None, dtype=jnp.float32, dynamics_fcn_ct_time_dependent=False, **odeint_kwargs) -> trajectory:
    n, d = x0.shape
    gamma = jnp.empty((steps + 1, n, d), dtype=dtype)
    T = jnp.arange(0, steps + 1, 1) * dt
    _f = dynamics_fcn_ct if dynamics_fcn_ct_time_dependent else lambda x, t: dynamics_fcn_ct(
        x)

    gamma = odeint(_f, x0, T, **odeint_kwargs)
    traj_gamma = trajectory(
        jnp.transpose(gamma, (1, 0, 2)),
        T,
        n,
        d,
        steps + 1,
        dt,
        t0=t0)
    return traj_gamma


def sample_grid(N1, d, _range: tuple = (-1., 1.),
                dtype=jnp.float32) -> Tuple[Array, int]:
    x1_mesh = jnp.linspace(_range[0], _range[1], N1, dtype=dtype)
    x1_1, x1_2 = jnp.meshgrid(x1_mesh, x1_mesh)
    x1 = jnp.vstack([x1_1.flatten(), x1_2.flatten()]).T
    x = x1
    N1 = x.shape[0]
    return (x, N1)


def sample_circle(N1, r=1., random=False,
                  dtype=jnp.float32, PRNGKey=jr.PRNGKey(0)) -> Tuple[Array, Array, int]:
    if random:
        theta = jr.uniform(PRNGKey, N1) * jnp.pi * 2
    else:
        theta = jnp.linspace(0, 2 * jnp.pi, N1)
    x = r * jnp.vstack([jnp.cos(theta), jnp.sin(theta)]).T
    N1 = x.shape[0]
    return (x, N1)


@partial(jax.jit, static_argnames=["random", "N"])
def sample_disk(N, r=1., sec=2 * jnp.pi, random=False, PRNGKey=jax.random.PRNGKey(0),
                dtype=jnp.float32) -> Tuple[Array, Array, int]:
    if not hasattr(sec, "__iter__"):
        if sec is None:
            sec = (0, jnp.pi)
        else:
            sec = (0, sec)
    ang = sec[1] - sec[0]

    if not hasattr(r, "__iter__"):
        r0 = 0.
        r1 = r
    else:
        r0 = r[0]
        r1 = r[1]

    if random:
        rn = jax.random.uniform(PRNGKey, (2, N))
        theta = rn[0] * ang + sec[0]
        r = rn[1].reshape(N, 1) * (r1 - r0) + r0
        x = jnp.sqrt(r) * jnp.vstack([jnp.cos(theta), jnp.sin(theta)]).T
    else:
        N = int(N**0.5)
        theta = jnp.linspace(0, 1, N) * ang - sec[0]
        r = jnp.linspace(0, r, N).reshape(N, 1, 1, )
        x = jnp.sqrt(
            r) * (jnp.vstack([jnp.cos(theta), jnp.sin(theta)]).T).reshape(1, N, 2)
        x = x.reshape(-1, 2)

    N = x.shape[0]
    return (x, N)


def sample_box(N1, _range=jnp.array([[-1., 1.], [-1., 1.]]), angle=0., PRNGKey=jax.random.PRNGKey(0), dtype=jnp.float32):

    m = _range.sum(-1) / 2
    l = _range[:, 1] - _range[:, 0]
    x = (jax.random.uniform(PRNGKey, shape=(
        N1, _range.shape[0])) - 0.5) * l + m
    rm = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                   [jnp.sin(angle), jnp.cos(angle)]])
    x = (rm @ x.T).T
    N1 = x.shape[0]
    return (x, N1)


def _construct_multiindices(
    n: int, p_bar: int, verbose: bool = False
):
    from kkr.auxilliary.monomials import mono_between_next_grevlex
    from scipy.special import binom
    import numpy as np

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
                n, p, p, list_multiindices[i][j - 1].copy()
            )
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


def make_lattice(eigvals, order):
    n = eigvals.shape[0]
    list_multiindices = jnp.concatenate(
        _construct_multiindices(n, order, verbose=False)[0])
    lattice = jnp.array([jnp.sum(eigvals * mi) for mi in list_multiindices])
    return lattice, list_multiindices
