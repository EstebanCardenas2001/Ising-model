"""q-state Potts model: ``H = -J sum_<ij> delta(s_i, s_j) - h sum_i delta(s_i, 0)``.

Spins take values ``0 .. q-1``. The order parameter is the vector
``m = (1/N) sum_i (cos(2 pi s_i / q), sin(2 pi s_i / q))``, whose magnitude is
1 in a fully ordered state and ~0 in the disordered phase. For ``q = 2`` this
reduces to the Ising magnetization, and the model maps onto the Ising model
with ``J_Ising = J_Potts / 2``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ._cluster import flatten, union
from .base import LatticeModel


@njit(cache=True)
def _metropolis_kernel(spins, nbrs, q, beta, J, h):
    N, z = nbrs.shape
    accepted = 0
    for _ in range(N):
        i = np.random.randint(N)
        old = spins[i]
        new = np.random.randint(q - 1)  # uniform over the q-1 other states
        if new >= old:
            new += 1
        n_old = 0
        n_new = 0
        for k in range(z):
            sj = spins[nbrs[i, k]]
            if sj == old:
                n_old += 1
            elif sj == new:
                n_new += 1
        dE = -J * (n_new - n_old) - h * (int(new == 0) - int(old == 0))
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i] = new
            accepted += 1
    return accepted / N


@njit(cache=True)
def _wolff_cluster(spins, nbrs, q, p_add, stack):
    N, z = nbrs.shape
    seed = np.random.randint(N)
    s0 = spins[seed]
    s_new = np.random.randint(q - 1)
    if s_new >= s0:
        s_new += 1
    spins[seed] = s_new
    stack[0] = seed
    top = 1
    size = 1
    while top > 0:
        top -= 1
        i = stack[top]
        for k in range(z):
            j = nbrs[i, k]
            if spins[j] == s0 and np.random.random() < p_add:
                spins[j] = s_new
                stack[top] = j
                top += 1
                size += 1
    return size


@njit(cache=True)
def _wolff_kernel(spins, nbrs, q, p_add, stack, n_clusters):
    N = spins.shape[0]
    flipped = 0
    for _ in range(n_clusters):
        flipped += _wolff_cluster(spins, nbrs, q, p_add, stack)
    return flipped / n_clusters / N


@njit(cache=True)
def _sw_kernel(spins, nbrs, q, p_add, parent):
    N, z = nbrs.shape
    for i in range(N):
        parent[i] = i
    for i in range(N):
        for k in range(0, z, 2):
            j = nbrs[i, k]
            if spins[i] == spins[j] and np.random.random() < p_add:
                union(parent, i, j)
    largest = flatten(parent)
    new_state = np.empty(N, dtype=spins.dtype)
    for i in range(N):
        if parent[i] == i:
            new_state[i] = np.random.randint(q)  # uniform over all q states
    for i in range(N):
        spins[i] = new_state[parent[i]]
    return largest


@njit(cache=True)
def _energy_kernel(spins, nbrs, J, h):
    N, z = nbrs.shape
    E = 0.0
    for i in range(N):
        s = spins[i]
        same = 0
        for k in range(0, z, 2):
            if spins[nbrs[i, k]] == s:
                same += 1
        E -= J * same
        if s == 0:
            E -= h
    return E


class PottsModel(LatticeModel):
    """q-state Potts model (``q = 2`` is equivalent to Ising)."""

    name = "Potts"
    n_components = 2

    def __init__(self, L, dim=2, q=3, J=1.0, h=0.0, init="random", seed=None):
        if q < 2:
            raise ValueError(f"q must be >= 2, got {q}")
        self.q = int(q)
        super().__init__(L, dim, J=J, h=h, init=init, seed=seed)
        angles = 2.0 * np.pi * np.arange(self.q) / self.q
        self._state_vectors = np.column_stack([np.cos(angles), np.sin(angles)])

    def _initial_spins(self, init):
        if init == "random":
            return self.rng.integers(0, self.q, size=self.N).astype(np.int32)
        if init == "ordered":
            return np.zeros(self.N, dtype=np.int32)
        raise ValueError(f"Unknown init {init!r}")

    def params(self):
        return {"q": self.q, **super().params()}

    def energy(self):
        return float(_energy_kernel(self.spins, self.neighbors, self.J, self.h))

    def magnetization_vector(self):
        counts = np.bincount(self.spins, minlength=self.q)
        return counts @ self._state_vectors / self.N

    def _metropolis_sweep(self, beta):
        return _metropolis_kernel(self.spins, self.neighbors, self.q, beta, self.J, self.h)

    def _wolff_sweep(self, beta):
        self._check_cluster_valid()
        p_add = 1.0 - np.exp(-beta * self.J)
        return _wolff_kernel(
            self.spins, self.neighbors, self.q, p_add, self._cluster, self.wolff_clusters
        )

    def _swendsen_wang_sweep(self, beta):
        self._check_cluster_valid()
        p_add = 1.0 - np.exp(-beta * self.J)
        return _sw_kernel(self.spins, self.neighbors, self.q, p_add, self._cluster)

    @classmethod
    def critical_temperature(cls, dim, q=3, J=1.0, **_):
        # 2D: exact self-dual point. Continuous for q <= 4, first order for q > 4.
        if dim == 2:
            return J / np.log(1.0 + np.sqrt(q))
        return None
