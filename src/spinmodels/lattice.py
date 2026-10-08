"""Periodic hypercubic lattices represented as flat neighbor tables.

Sites are labelled ``0 .. N-1`` in C (row-major) order, so a spin array of
shape ``(N, ...)`` can be reshaped to ``lattice.shape + (...)`` for plotting.
Storing the geometry as an integer table rather than relying on ``np.roll``
lets every Numba kernel work unchanged in 1, 2 or 3 (or more) dimensions.
"""

from __future__ import annotations

import numpy as np


class HypercubicLattice:
    """A ``L^dim`` simple hypercubic lattice with periodic boundaries.

    Attributes
    ----------
    neighbors : ndarray of shape (N, 2*dim), int64
        ``neighbors[i, 2*a]`` is the neighbor of site ``i`` one step in the
        ``+a`` direction and ``neighbors[i, 2*a + 1]`` the one in ``-a``.
        The even ("forward") columns enumerate every bond exactly once, which
        is what the energy kernels use.
    """

    def __init__(self, L: int, dim: int):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if L < 3:
            # For L = 2 the +a and -a neighbors coincide, giving doubled bonds.
            raise ValueError(f"L must be >= 3 for a well-defined periodic lattice, got {L}")
        self.L = int(L)
        self.dim = int(dim)
        self.shape = (self.L,) * self.dim
        self.N = self.L**self.dim
        self.coordination = 2 * self.dim
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self) -> np.ndarray:
        idx = np.arange(self.N, dtype=np.int64).reshape(self.shape)
        nbrs = np.empty((self.N, self.coordination), dtype=np.int64)
        for axis in range(self.dim):
            # np.roll(idx, -1)[x] == idx[x + 1]
            nbrs[:, 2 * axis] = np.roll(idx, -1, axis=axis).ravel()
            nbrs[:, 2 * axis + 1] = np.roll(idx, 1, axis=axis).ravel()
        return nbrs

    @property
    def n_bonds(self) -> int:
        return self.N * self.dim

    def reshape(self, site_array: np.ndarray) -> np.ndarray:
        """Reshape a per-site array ``(N, ...)`` to ``(L, ..., L, ...)``."""
        return site_array.reshape(self.shape + site_array.shape[1:])

    def __repr__(self) -> str:
        return f"HypercubicLattice(L={self.L}, dim={self.dim}, N={self.N})"
