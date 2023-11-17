import jax
import jax.numpy as jnp
from typing import Any

from jaxtyping import Array


def _normalize_by_h(L, **kwargs):
    return L / L.shape[1]


def make_koopman_kernel(base_kernel, eigenvalues_dt, H, einsum_kwargs={"optimize": True}, normalize=_normalize_by_h):

    D = eigenvalues_dt.shape[0]

    # backwards in time **(-h)
    pullback_mu_DH = normalize(jnp.power(eigenvalues_dt[:, None], - jnp.arange(H)[None, :]))
    # forward in time
    rollout_mu_DH = jnp.power(eigenvalues_dt[:, None], jnp.arange(H)[None, :])

    def first_argument_invariant_f(x_H, y, pullback_mu_H):
        vf = jax.vmap(base_kernel, in_axes=[1, None], out_axes=0)
        return jnp.sum(
            pullback_mu_H[:, None, None]
            *
            vf(x_H, y), axis=0
        )

    def second_argument_invariant_f(x, y_H, pullback_mu_H):
        vf = jax.vmap(base_kernel, in_axes=[None, 1], out_axes=1)
        return jnp.sum(
            pullback_mu_H.conj()[None, None, :]
            *
            vf(x, y_H), axis=1
        )

    def both_argument_invariant_f(x_H, y_H, pullback_mu_H):
        vf = jax.vmap(
            first_argument_invariant_f,
            in_axes=[None, 1, None],
            out_axes=2,
        )
        return jnp.sum(
            pullback_mu_H.conj()[None, None, :]
            *
            vf(x_H, y_H, pullback_mu_H),
            axis=2
        )

    def coreg_f(x_H, y_H):
        vf = jax.vmap(
            both_argument_invariant_f, in_axes=[None, None, 0], out_axes=0
        )
        return (
            vf(x_H, y_H, pullback_mu_DH) / D
        )

    def coreg_first_f(x_H, y):
        vf = jax.vmap(
            first_argument_invariant_f, in_axes=[None, None, 0], out_axes=0
        )
        return (
            vf(x_H, y, pullback_mu_DH)
        )

    def coreg_second_f(x, y_H):
        vf = jax.vmap(
            second_argument_invariant_f, in_axes=[None, None, 0], out_axes=0
        )
        return (
            vf(x, y_H, pullback_mu_DH)
        )

    def armin_f(x_H, y_H):
        return jnp.einsum(
            "DNM, DH, DS->HNSM",
            coreg_f(x_H, y_H),
            rollout_mu_DH,
            rollout_mu_DH.conj(),
            **einsum_kwargs
        )

    def armin_invariant_in_second(x, y_H):
        return jnp.einsum(
            "DNM, DH, DS->HNSM",
            coreg_second_f(x, y_H),
            rollout_mu_DH,
            rollout_mu_DH.conj(),
            **einsum_kwargs
        )

    def extract_r(x_H, y_H):
        return jnp.einsum(
            "DNM, DS->DNSM",
            coreg_f(x_H, y_H),
            rollout_mu_DH.conj(),
            **einsum_kwargs
        )

    def extract_invariant_in_second_r(x, y_H):
        return jnp.einsum(
            "DNM, DS->DNSM",
            coreg_second_f(x, y_H),
            rollout_mu_DH.conj(),
            **einsum_kwargs
        )

    def mercer_first_f(x_H, y):
        return jnp.einsum(
            "DNM, DH->DHNM",
            coreg_first_f(x_H, y),
            rollout_mu_DH,
            **einsum_kwargs
        )

    def mercer_second_f(x, y_H):
        return jnp.einsum(
            "DNM, DS->DNSM",
            coreg_second_f(x, y_H),
            rollout_mu_DH.conj(),
            **einsum_kwargs
        )

    def rollout_second(x, y_H):
        return jnp.einsum(
            "DNM, DH->DHNM",
            coreg_second_f(x, y_H),
            rollout_mu_DH,
            **einsum_kwargs
        )
    return {
        "coreg": coreg_f,
        "coreg_first": coreg_first_f,
        "coreg_second": coreg_second_f,
        "armin": armin_f,
        "armin_invariant_in_second": armin_invariant_in_second,
        "extract": extract_r,
        "extract_inariant_in_second": extract_invariant_in_second_r,
        "mercer_invariant_in_first": mercer_first_f,
        "mercer_invariant_in_second": mercer_second_f,
        "rollout_second": rollout_second
    }
