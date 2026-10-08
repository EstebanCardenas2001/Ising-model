"""Classical O(n) vector models: XY (n = 2) and Heisenberg (n = 3).

``H = -J sum_<ij> S_i . S_j - h sum_i S_i^x`` with unit vectors ``S_i``.

Both models share the same kernels; spins are stored as an ``(N, n)`` array
of unit vectors (rather than angles), which keeps the code dimension-agnostic
in spin space as well as in real space.

Metropolis proposal: ``S' = normalize(S + delta * g)`` with ``g`` a standard
Gaussian vector. The proposal density depends only on ``S . S'``, so it is
symmetric, and ``delta`` is tuned during equilibration to reach a target
acceptance rate.

Wolff update (Wolff 1989): pick a random unit vector ``r``, and reflect
cluster spins in the hyperplane orthogonal to it, ``S -> S - 2 (S.r) r``.
A bond joins the cluster with probability
``1 - exp(min(0, -2 beta J (S_i.r)(S_j.r)))``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ._cluster import flatten, union
from .base import LatticeModel


@njit(cache=True)
def _metropolis_kernel(spins, nbrs, beta, J, h, delta):
    N, z = nbrs.shape
    n = spins.shape[1]
    new = np.empty(n)
    field = np.empty(n)
    accepted = 0
    for _ in range(N):
        i = np.random.randint(N)
        norm2 = 0.0
        for c in range(n):
            new[c] = spins[i, c] + delta * np.random.normal()
            norm2 += new[c] * new[c]
        inv = 1.0 / np.sqrt(norm2)
        field[:] = 0.0
        for k in range(z):
            j = nbrs[i, k]
            for c in range(n):
                field[c] += spins[j, c]
        dE = 0.0
        for c in range(n):
            new[c] *= inv
            dE -= J * (new[c] - spins[i, c]) * field[c]
        dE -= h * (new[0] - spins[i, 0])
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            for c in range(n):
                spins[i, c] = new[c]
            accepted += 1
    return accepted / N


@njit(cache=True)
def _random_unit_vector(n):
    r = np.empty(n)
    norm2 = 0.0
    for c in range(n):
        r[c] = np.random.normal()
        norm2 += r[c] * r[c]
    return r / np.sqrt(norm2)


@njit(cache=True)
def _wolff_cluster(spins, nbrs, two_beta_J, queue, in_cluster):
    N, z = nbrs.shape
    n = spins.shape[1]
    r = _random_unit_vector(n)

    seed = np.random.randint(N)
    proj = 0.0
    for c in range(n):
        proj += spins[seed, c] * r[c]
    for c in range(n):
        spins[seed, c] -= 2.0 * proj * r[c]
    in_cluster[seed] = True
    queue[0] = seed
    head = 0
    tail = 1

    while head < tail:
        i = queue[head]
        head += 1
        # Spin i is already reflected; its pre-reflection projection is -(S_i . r).
        proj_i = 0.0
        for c in range(n):
            proj_i -= spins[i, c] * r[c]
        for k in range(z):
            j = nbrs[i, k]
            if in_cluster[j]:
                continue
            proj_j = 0.0
            for c in range(n):
                proj_j += spins[j, c] * r[c]
            x = two_beta_J * proj_i * proj_j
            if x > 0.0 and np.random.random() < 1.0 - np.exp(-x):
                for c in range(n):
                    spins[j, c] -= 2.0 * proj_j * r[c]
                in_cluster[j] = True
                queue[tail] = j
                tail += 1

    for m in range(tail):
        in_cluster[queue[m]] = False
    return tail


@njit(cache=True)
def _wolff_kernel(spins, nbrs, two_beta_J, queue, in_cluster, n_clusters):
    N = spins.shape[0]
    flipped = 0
    for _ in range(n_clusters):
        flipped += _wolff_cluster(spins, nbrs, two_beta_J, queue, in_cluster)
    return flipped / n_clusters / N


@njit(cache=True)
def _sw_kernel(spins, nbrs, two_beta_J, parent):
    """Swendsen-Wang with the Wolff embedding: one random hyperplane for the
    whole lattice, every cluster reflected with probability 1/2."""
    N, z = nbrs.shape
    n = spins.shape[1]
    r = _random_unit_vector(n)
    proj = np.zeros(N)
    for i in range(N):
        for c in range(n):
            proj[i] += spins[i, c] * r[c]
    for i in range(N):
        parent[i] = i
    for i in range(N):
        for k in range(0, z, 2):
            j = nbrs[i, k]
            x = two_beta_J * proj[i] * proj[j]
            if x > 0.0 and np.random.random() < 1.0 - np.exp(-x):
                union(parent, i, j)
    largest = flatten(parent)
    flip = np.empty(N, dtype=np.bool_)
    for i in range(N):
        if parent[i] == i:
            flip[i] = np.random.random() < 0.5
    for i in range(N):
        if flip[parent[i]]:
            for c in range(n):
                spins[i, c] -= 2.0 * proj[i] * r[c]
    return largest


@njit(cache=True)
def _energy_kernel(spins, nbrs, J, h):
    N, z = nbrs.shape
    n = spins.shape[1]
    E = 0.0
    for i in range(N):
        for k in range(0, z, 2):
            j = nbrs[i, k]
            dot = 0.0
            for c in range(n):
                dot += spins[i, c] * spins[j, c]
            E -= J * dot
        E -= h * spins[i, 0]
    return E


class VectorModel(LatticeModel):
    """Base class for O(n) models with ``n``-component unit-vector spins."""

    name = "O(n)"
    n_components = 0  # set by subclasses

    #: Bounds and target for the adaptive Metropolis step size.
    target_acceptance = 0.5
    min_step, max_step = 0.05, 10.0

    def __init__(self, L, dim, J=1.0, h=0.0, init="random", seed=None, step_size=1.0):
        super().__init__(L, dim, J=J, h=h, init=init, seed=seed)
        self.step_size = float(step_size)

    def _initial_spins(self, init):
        n = self.n_components
        if init == "random":
            s = self.rng.normal(size=(self.N, n))
            return s / np.linalg.norm(s, axis=1, keepdims=True)
        if init == "ordered":
            s = np.zeros((self.N, n))
            s[:, 0] = 1.0
            return s
        raise ValueError(f"Unknown init {init!r}")

    def energy(self):
        return float(_energy_kernel(self.spins, self.neighbors, self.J, self.h))

    def magnetization_vector(self):
        return self.spins.mean(axis=0)

    def _metropolis_sweep(self, beta):
        return _metropolis_kernel(
            self.spins, self.neighbors, beta, self.J, self.h, self.step_size
        )

    def _wolff_sweep(self, beta):
        self._check_cluster_valid()
        return _wolff_kernel(
            self.spins, self.neighbors, 2.0 * beta * self.J, self._cluster, self._in_cluster,
            self.wolff_clusters,
        )

    def _swendsen_wang_sweep(self, beta):
        self._check_cluster_valid()
        return _sw_kernel(self.spins, self.neighbors, 2.0 * beta * self.J, self._cluster)

    def adapt_step_size(self, acceptance):
        # Multiplicative update; converges geometrically toward the target rate.
        factor = np.exp(acceptance - self.target_acceptance)
        self.step_size = float(np.clip(self.step_size * factor, self.min_step, self.max_step))

    def renormalize(self) -> None:
        """Remove floating-point drift from the unit-length constraint."""
        self.spins /= np.linalg.norm(self.spins, axis=1, keepdims=True)


class XYModel(VectorModel):
    """Classical XY (plane rotator) model, ``n = 2``.

    In 2D there is no long-range order at T > 0 (Mermin–Wagner); instead a
    Berezinskii–Kosterlitz–Thouless transition occurs at ``T_BKT ≈ 0.8929 J``.
    """

    name = "XY"
    n_components = 2

    def __init__(self, L, dim=2, J=1.0, h=0.0, init="random", seed=None, step_size=1.0):
        super().__init__(L, dim, J=J, h=h, init=init, seed=seed, step_size=step_size)

    def angles(self) -> np.ndarray:
        """Spin angles in ``(-pi, pi]``."""
        return np.arctan2(self.spins[:, 1], self.spins[:, 0])

    @classmethod
    def critical_temperature(cls, dim, J=1.0, **_):
        known = {2: 0.8929, 3: 2.20184}  # 2D: T_BKT; 3D: simple cubic
        return J * known[dim] if dim in known else None


class HeisenbergModel(VectorModel):
    """Classical Heisenberg model, ``n = 3``."""

    name = "Heisenberg"
    n_components = 3

    def __init__(self, L, dim=3, J=1.0, h=0.0, init="random", seed=None, step_size=1.0):
        super().__init__(L, dim, J=J, h=h, init=init, seed=seed, step_size=step_size)

    @classmethod
    def critical_temperature(cls, dim, J=1.0, **_):
        return J * 1.44300 if dim == 3 else None  # simple cubic
