"""
Optimisation utilities for the VMC tutorials.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


__all__ = [
    "apply_update",
    "sr_update",
    "sr_precondition",
]


def apply_update(params, grad, learning_rate):
    """
    Standard gradient-descent update.

        theta <- theta - eta * grad
    """
    return jax.tree.map(
        lambda p, g: p - learning_rate * g,
        params,
        grad,
    )


def sr_update(
    params,
    grad,
    O,
    learning_rate=0.05,
    diag_shift=1e-2,
):
    """
    Stochastic-reconfiguration update.

    Parameters
    ----------
    params
        Parameter PyTree.

    grad
        Energy gradient PyTree.

    O
        Logarithmic derivatives evaluated on the samples.

        Each leaf must have shape

            (n_samples, *parameter_shape)

    learning_rate
        Optimisation step size.

    diag_shift
        Diagonal-shift regulariser added to the SR matrix.

    Returns
    -------
    Updated parameter PyTree.
    """

    grad_flat, unravel = ravel_pytree(grad)

    O_flat_leaves = []

    for leaf in jax.tree.leaves(O):
        O_flat_leaves.append(
            leaf.reshape(leaf.shape[0], -1)
        )

    O_flat = jnp.concatenate(O_flat_leaves, axis=1)

    O_centered = O_flat - jnp.mean(O_flat, axis=0, keepdims=True)

    S = (
        O_centered.T @ O_centered
    ) / O_centered.shape[0]

    S = S + diag_shift * jnp.eye(S.shape[0])

    direction = jnp.linalg.solve(S, grad_flat)

    delta = unravel(direction)

    return jax.tree.map(
        lambda p, d: p - learning_rate * d,
        params,
        delta,
    )

def sr_precondition(grad, O, diag_shift=1e-2):
    grad_flat, unravel = ravel_pytree(grad)

    O_flat = jnp.concatenate(
        [
            leaf.reshape(leaf.shape[0], -1)
            for leaf in jax.tree.leaves(O)
        ],
        axis=1,
    )

    O_centered = O_flat - jnp.mean(O_flat, axis=0, keepdims=True)

    S = O_centered.T @ O_centered / O_centered.shape[0]
    S = S + diag_shift * jnp.eye(S.shape[0])

    direction_flat = jnp.linalg.solve(S, grad_flat)

    return unravel(direction_flat)
