"""
Hamiltonian utilities for the VMC tutorials.

This module contains small helper functions for the transverse-field Ising
model. The notebook keeps the local-energy estimator visible, while the sparse
Hamiltonian construction used for exact diagonalisation is hidden here.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


__all__ = [
    "tfim_hamiltonian_sparse",
    "exact_ground_state_energy",
]


def int_to_spin_config(x: int, n_sites: int) -> np.ndarray:
    """Convert an integer to a spin configuration with values in {-1, +1}."""
    bits = np.array([(x >> i) & 1 for i in range(n_sites)], dtype=np.int8)
    return 2 * bits - 1


def tfim_hamiltonian_sparse(n_sites: int, g: float) -> sp.csr_matrix:
    """
    Build the sparse Hamiltonian matrix of the 1D transverse-field Ising model.

    The Hamiltonian is

        H = - sum_i sigma^z_i sigma^z_{i+1}
            - g sum_i sigma^x_i

    with periodic boundary conditions.
    """
    dim = 2**n_sites

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for x in range(dim):
        sigma = int_to_spin_config(x, n_sites)

        # Diagonal Ising interaction.
        diagonal = -sum(
            sigma[i] * sigma[(i + 1) % n_sites]
            for i in range(n_sites)
        )

        rows.append(x)
        cols.append(x)
        data.append(float(diagonal))

        # Off-diagonal transverse-field term.
        for i in range(n_sites):
            y = x ^ (1 << i)

            rows.append(x)
            cols.append(y)
            data.append(-float(g))

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(dim, dim),
    )


def exact_ground_state_energy(n_sites: int, g: float) -> float:
    """Compute the exact ground-state energy by sparse diagonalisation."""
    hamiltonian = tfim_hamiltonian_sparse(n_sites, g)

    eigenvalues = spla.eigsh(
        hamiltonian,
        k=1,
        which="SA",
        return_eigenvectors=False,
    )

    return float(eigenvalues[0])
    