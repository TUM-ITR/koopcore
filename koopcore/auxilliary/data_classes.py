import jax.numpy as jnp
import jax
import numpy.random as nr
from numpy import savez_compressed
import jax_dataclasses as jdc
from typing import Tuple, List, Dict
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
import pandas as pd
import os
from jaxtyping import Array, Num
from functools import partial
import warnings


class trajectory:
    def __init__(self, X, T=None, N=None, d=None, H=None, dt=1, t0=0):
        if d is None:
            self.d = X.shape[-1] - 1
        else:
            self.d = d

        if T is None:
            if self.d == X.shape[2]:
                self.XT = jnp.concatenate([
                    X,
                    t0 + dt*jnp.arange(0, X.shape[1])
                ])
            else:
                self.XT = X
        else:
            assert T.shape[-1] == X.shape[1]
            if len(T.shape) < 2:
                T = T.reshape(1, -1).repeat(X.shape[0], 0)
            self.XT = jnp.concatenate([X, jnp.expand_dims(T, axis=2)], axis=2)
            self.d = X.shape[2]

    @property
    def X(self):
        return self.XT[:, :, :-1]

    @property
    def T(self):
        return self.XT[:, :, -1]

    @property
    def N(self):
        return self.XT.shape[0]

    @property
    def H(self):
        return self.XT.shape[1]

    @property
    def shape(self):
        return self.XT.shape

    @property
    def dt(self):
        if jnp.allclose(self.T[:, 1:] - self.T[:, :-1], self.T[0, 1] - self.T[0, 0]):
            return self.T[:, 1] - self.T[:, 0]
        return None

    def set_XT(self, X, T):
        self.XT = jnp.concatenate([X, jnp.expand_dims(T, 2)], axis=2)

    def set_X(self, X):
        assert X.shape == self.X.shape
        self.set_XT(X, self.T)

    def set_T(self, T):
        assert T.shape == self.T.shape
        self.set_XT(self.X, T)

    def __str__(self) -> str:
        return f"trajectory: N={self.N}, H={self.H}, d={self.d}, dt={self.dt}, t0={self.t0}"

    def __call__(self, T):  # interpolate the trajectory for X(t)
        T = jnp.asarray(T)
        if T.shape.__len__() == 0:
            T = jnp.atleast_2d(T).repeat(self.N, 0)
        elif T.shape.__len__() == 1:
            T = T.reshape(1, -1).repeat(self.N, 0)
        else:
            pass
        X_interp = jax.vmap(jax.vmap(jnp.interp, [0, 0, 0], 0), [
                            None, None, 2], 2)(T, self.T, self.X)
        return trajectory(X_interp, T, self.N, self.d, T.shape[1], T[0][1] - T[0][0])

    def __getitem__(self, args, /):
        return self.X[args]

    def __neg__(self):
        return trajectory(-self.X, self.T, self.N, self.d, self.H, self.dt)

    def __add__(self, other):
        if isinstance(other, trajectory):
            return trajectory(self.X + other.X, self.T, self.N, self.d, self.H, self.dt)
        if isinstance(other, jnp.ndarray):
            try:
                jnp.broadcast_shapes(self.X.shape, other.shape)
            except:
                raise ValueError(
                    "data to add to trajectory must be broadcastable")
            return trajectory(self.X + other, self.T, self.N, self.d, self.H, self.dt)

        raise ValueError(
            "addition not implemented for types " + str(type(self)) + " and " + str(type(other)) + "")

    def __sub__(self, other):
        return self + (-other)

    @property
    def extended(self):
        return jnp.concatenate([self.X, jnp.expand_dims(self.T, 2)], axis=-1)

    def select_H(self, H_indices):
        if isinstance(H_indices, int):
            H_indices = jnp.arange(H_indices)
        assert max(H_indices) <= self.H
        dt = self.T[0, H_indices][1] - self.T[0, H_indices][0]
        if not jnp.allclose((self.T[:, H_indices][1:] - self.T[:, H_indices][:-1]), dt, rtol=1e-4):
            dt = None
        return trajectory(self.X[:, H_indices, :], self.T[:, H_indices],
                          self.N, self.d, len(H_indices), dt)

    def select_N(self, N_indices):
        if isinstance(N_indices, int):
            N_indices = jnp.arange(N_indices)
        assert max(N_indices) <= self.N

        return trajectory(self.X[N_indices, :, :], self.T[N_indices, ...],
                          len(N_indices), self.d, self.H, self.dt)

    def select_N_jitable(self, N_indices):
        if isinstance(N_indices, int):
            N_indices = jnp.arange(N_indices)
        return trajectory(self.X[N_indices, :, :], self.T[N_indices, ...],
                          len(N_indices), self.d, self.H, self.dt)

    def select_d(self, d_indices):
        assert max(d_indices) <= self.d
        return trajectory(
            jnp.array(self.X[:, :, d_indices]),
            self.T, self.N,
            len(d_indices),
            self.H,
            self.dt)


def save_trajectory_dict(path: str, trajectory_dict: dict[str, trajectory] = {}, **trajectory_kw):
    trajectory_dict.update(trajectory_kw)
    save_dict = {}
    for key, value in zip(trajectory_dict.keys(), trajectory_dict.values()):
        save_dict.update({key + "x": value.X, key + "t": value.T})
    savez_compressed(path, **save_dict)


def load_trajectory_dict(path: str):
    with jnp.load(path) as _d:
        trajectory_dict = {}
        for key in _d.keys():
            if key[-1] == "x":
                trajectory_dict.update(
                    {key[:-1]: trajectory(_d[key], _d[key[:-1] + "t"])})
    return trajectory_dict


def save_trajectory_data_train_test(path, list_data_train, list_data_test):
    data_train_X = jnp.stack([iid.X for iid in list_data_train])
    data_train_T = jnp.stack([iid.T for iid in list_data_train])
    data_test_X = jnp.stack([iid.X for iid in list_data_test])
    data_test_T = jnp.stack([iid.T for iid in list_data_test])
    savez_compressed(path, trainx=data_train_X, traint=data_train_T,
                     testx=data_test_X, testt=data_test_T)


def load_trajectory_data_train_test(path: str) -> Tuple[List[trajectory], List[trajectory]]:
    _l = jnp.load(path)
    tensorR_data_train_X = _l["trainx"]
    tensorR_data_train_T = _l["traint"]
    tensorR_data_test_X = _l["testx"]
    tensorR_data_test_T = _l["testt"]
    list_data_train = []
    list_data_test = []
    for ix, it in zip(tensorR_data_train_X, tensorR_data_train_T):
        list_data_train.append(trajectory(ix, it))
    for ix, it in zip(tensorR_data_test_X, tensorR_data_test_T):
        list_data_test.append(trajectory(ix, it))
    _l.close()
    return list_data_train, list_data_test


if __name__ == "__main__":
    import doctest
    doctest.testmod()
