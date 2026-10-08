"""Ising model: ``H = -J sum_<ij> s_i s_j - h sum_i s_i`` with ``s_i = ±1``."""

from __future__ import annotations

import numpy as np
from numba import njit

from ._cluster import flatten, union
from .base import LatticeModel


@njit(cache=True)
def _metropolis_kernel(spins, nbrs, beta, J, h):
    N, z = nbrs.shape
    accepted = 0
    for _ in range(N):
        i = np.random.randint(N)
        s = spins[i]
        local = 0
        for k in range(z):
            local += spins[nbrs[i, k]]
        dE = 2.0 * s * (J * local + h)
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i] = -s
            accepted += 1
    return accepted / N


@njit(cache=True)
def _wolff_cluster(spins, nbrs, p_add, stack):
    """Grow and flip one Wolff cluster; return its size.

    Spins are flipped as they join the cluster, so "same sign as the seed's
    original value" also means "not yet in the cluster".
    """
    N, z = nbrs.shape
    seed = np.random.randint(N)
    s0 = spins[seed]
    spins[seed] = -s0
    stack[0] = seed
    top = 1
    size = 1
    while top > 0:
        top -= 1
        i = stack[top]
        for k in range(z):
            j = nbrs[i, k]
            if spins[j] == s0 and np.random.random() < p_add:
                spins[j] = -s0
                stack[top] = j
                top += 1
                size += 1
    return size


@njit(cache=True)
def _wolff_kernel(spins, nbrs, p_add, stack, n_clusters):
    N = spins.shape[0]
    flipped = 0
    for _ in range(n_clusters):
        flipped += _wolff_cluster(spins, nbrs, p_add, stack)
    return flipped / n_clusters / N


@njit(cache=True)
def _sw_kernel(spins, nbrs, p_add, parent):
    N, z = nbrs.shape
    for i in range(N):
        parent[i] = i
    for i in range(N):
        for k in range(0, z, 2):
            j = nbrs[i, k]
            if spins[i] == spins[j] and np.random.random() < p_add:
                union(parent, i, j)
    largest = flatten(parent)
    flip = np.empty(N, dtype=np.bool_)
    for i in range(N):
        if parent[i] == i:
            flip[i] = np.random.random() < 0.5
    for i in range(N):
        if flip[parent[i]]:
            spins[i] = -spins[i]
    return largest


@njit(cache=True)
def _energy_kernel(spins, nbrs, J, h):
    N, z = nbrs.shape
    E = 0.0
    for i in range(N):
        s = spins[i]
        bond = 0
        for k in range(0, z, 2):  # forward neighbors: each bond once
            bond += spins[nbrs[i, k]]
        E -= J * s * bond + h * s
    return E


class IsingModel(LatticeModel):
    """Ising model in any dimension (1D, 2D and 3D are the usual cases)."""

    name = "Ising"
    n_components = 1

    def __init__(self, L, dim=2, J=1.0, h=0.0, init="random", seed=None):
        super().__init__(L, dim, J=J, h=h, init=init, seed=seed)

    def _initial_spins(self, init):
        if init == "random":
            return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.N)
        if init == "ordered":
            return np.ones(self.N, dtype=np.int8)
        raise ValueError(f"Unknown init {init!r}")

    def energy(self):
        return float(_energy_kernel(self.spins, self.neighbors, self.J, self.h))

    def magnetization_vector(self):
        return np.array([self.spins.mean(dtype=np.float64)])

    def _metropolis_sweep(self, beta):
        return _metropolis_kernel(self.spins, self.neighbors, beta, self.J, self.h)

    def _wolff_sweep(self, beta):
        self._check_cluster_valid()
        p_add = 1.0 - np.exp(-2.0 * beta * self.J)
        return _wolff_kernel(self.spins, self.neighbors, p_add, self._cluster, self.wolff_clusters)

    def _swendsen_wang_sweep(self, beta):
        self._check_cluster_valid()
        p_add = 1.0 - np.exp(-2.0 * beta * self.J)
        return _sw_kernel(self.spins, self.neighbors, p_add, self._cluster)

    @classmethod
    def critical_temperature(cls, dim, J=1.0, **_):
        # 2D: Onsager (exact). 3D: simple cubic, beta_c = 0.22165463 (MC).
        known = {2: 2.0 / np.log(1.0 + np.sqrt(2.0)), 3: 1.0 / 0.22165463}
        return J * known[dim] if dim in known else None
