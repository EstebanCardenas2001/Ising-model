"""Batched GPU simulation of lattice spin models (CuPy + raw CUDA kernels).

A :class:`GPUModel` holds ``R`` independent replicas of an ``L^dim`` lattice,
each with its own temperature, and updates all of them in a single kernel
launch. This is how the GPU is kept busy: a temperature scan, a
parallel-tempering ensemble or a mosaic of temperatures for a video is just
one batched system.

Algorithms (``model.sweep(algorithm)``):

* ``"local"``           the model's checkerboard single-spin update (two
  half-sweeps): heat-bath for Ising/Potts (checkerboard Metropolis is not
  ergodic for discrete spins, see ``kernels.cu``), Metropolis for O(n).
  Also available under its own name, ``"heatbath"`` or ``"metropolis"``.
* ``"swendsen_wang"``   multi-cluster update: bond activation, parallel
  union-find labelling (atomicMin hooking), and a random update per cluster.
  For O(n) models the Wolff embedding (reflection across a random hyperplane,
  one per replica) is used.

Conventions and Hamiltonians are identical to the CPU models in
:mod:`spinmodels.models`, whose classes are reused for names and known
critical temperatures. Spins are float32 / int8 on the device; energies and
magnetizations are reduced in float64.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import cupy as cp
import numpy as np

from ..models import HeisenbergModel, IsingModel, LatticeModel, PottsModel, XYModel

_THREADS = 256
_MEAS_K = 10  # must match MEAS_K in kernels.cu
_SOURCE = (Path(__file__).with_name("kernels.cu")).read_text()


@lru_cache(maxsize=None)
def _module() -> cp.RawModule:
    names = [
        f"on_{k}<{nc}>"
        for k in ("metropolis", "sw_bonds", "sw_apply", "normalize", "measure")
        for nc in (2, 3)
    ]
    mod = cp.RawModule(code=_SOURCE, options=("-std=c++14", "--use_fast_math"), name_expressions=names)
    return mod


def _kernel(name: str) -> cp.RawKernel:
    return _module().get_function(name)


def _grid(n: int) -> tuple[tuple[int], tuple[int]]:
    return ((int(n) + _THREADS - 1) // _THREADS,), (_THREADS,)


class GPUModel:
    """Base class: ``R`` replicas of one model on the GPU.

    Parameters
    ----------
    L, dim : lattice size (``L`` must be even for the checkerboard) and dimension.
    temperatures : one temperature per replica.
    J, h : coupling and field (as in the CPU models).
    init : "random" or "ordered".
    seed : seeds the Philox stream and host-side randomness.
    """

    cpu_model: ClassVar[type[LatticeModel]]
    n_components: ClassVar[int] = 1

    def __init__(self, L, dim, temperatures, J=1.0, h=0.0, init="random", seed=None):
        if L % 2 or L < 4:
            raise ValueError(f"GPU models need an even L >= 4, got {L}")
        if dim > 4:
            raise ValueError("dim <= 4 supported")
        self.L, self.dim = int(L), int(dim)
        self.N = self.L**self.dim
        self.J, self.h = float(J), float(h)
        self.rng = np.random.default_rng(seed)
        self._key = [np.uint32(k) for k in self.rng.integers(0, 2**32, size=2)]
        self._step = 0
        self.set_temperatures(temperatures)
        self.spins = self._initial_spins(init)
        self._labels = None
        self._bonds = None

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self.cpu_model.name

    @property
    def R(self) -> int:
        return len(self.T)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.L,) * self.dim

    def params(self) -> dict:
        return {"J": self.J, "h": self.h}

    def critical_temperature(self) -> float | None:
        return self.cpu_model.critical_temperature(self.dim, **self.params())

    def set_temperatures(self, temperatures) -> None:
        T = np.atleast_1d(np.asarray(temperatures, dtype=np.float64))
        if np.any(T <= 0):
            raise ValueError("Temperatures must be positive")
        if hasattr(self, "T") and len(T) != len(self.T):
            raise ValueError("Number of replicas is fixed at construction")
        self.T = T
        self.beta = cp.asarray(1.0 / T, dtype=cp.float32)

    def _next(self) -> tuple:
        """Seed key and a fresh launch counter for the RNG."""
        self._step += 1
        return (self._key[0], self._key[1], np.uint64(self._step))

    # ------------------------------------------------------------------ #
    def sweep(self, algorithm: str = "local", n: int = 1) -> None:
        method = getattr(self, f"_{algorithm}", None)
        if method is None:
            raise ValueError(f"Unknown GPU algorithm {algorithm!r}")
        for _ in range(n):
            method()

    def _swendsen_wang(self) -> None:
        if self.h != 0.0 or self.J <= 0.0:
            raise ValueError("Swendsen-Wang requires h = 0 and J > 0")
        total = self.R * self.N
        if self._labels is None:
            self._labels = cp.empty(total, dtype=cp.int32)
            self._bonds = cp.empty(total, dtype=cp.uint8)
        grid = _grid(total)
        self._sw_bonds()
        _kernel("sw_init")(*grid, (self._labels, np.int64(total)))
        _kernel("sw_union")(*grid, (self._bonds, self._labels, np.int32(self.R), np.int32(self.N),
                                    np.int32(self.L), np.int32(self.dim)))
        _kernel("sw_flatten")(*grid, (self._labels, np.int64(total)))
        self._sw_apply()

    def cluster_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Number of clusters and largest-cluster fraction per replica, from
        the most recent Swendsen-Wang labelling."""
        if self._labels is None:
            raise RuntimeError("Run a Swendsen-Wang sweep first")
        labels = self._labels.reshape(self.R, self.N)
        is_root = labels == (cp.arange(self.R * self.N, dtype=cp.int32).reshape(self.R, self.N))
        n_clusters = cp.asnumpy(is_root.sum(axis=1))
        local = labels - (cp.arange(self.R, dtype=cp.int32) * self.N)[:, None]
        largest = np.array([int(cp.bincount(local[r], minlength=1).max()) for r in range(self.R)])
        return n_clusters, largest / self.N

    # ------------------------------------------------------------------ #
    # Model-specific
    # ------------------------------------------------------------------ #
    def _initial_spins(self, init):
        raise NotImplementedError

    def _local(self) -> None:
        raise NotImplementedError

    def _sw_bonds(self) -> None:
        raise NotImplementedError

    def _sw_apply(self) -> None:
        raise NotImplementedError

    def _launch_measure(self, out: cp.ndarray) -> None:
        raise NotImplementedError

    def measure(self) -> np.ndarray:
        """Raw per-replica sums from the fused measurement kernel, shape
        ``(R, 10)`` on the host (see ``kernels.cu`` for the layout)."""
        out = cp.zeros((self.R, _MEAS_K), dtype=cp.float64)
        self._launch_measure(out)
        return cp.asnumpy(out)

    def energies(self) -> np.ndarray:
        """Total energy per replica, shape ``(R,)``."""
        raw = self.measure()
        return -self.J * raw[:, 0] - self.h * raw[:, 1]

    def magnetization_vectors(self) -> np.ndarray:
        """Order-parameter vector per site, shape ``(R, n_components)``."""
        return self.measure()[:, 2 : 2 + self.n_components] / self.N

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Host arrays ``(E/N, |m|)`` for every replica (one kernel launch)."""
        raw = self.measure()
        e = (-self.J * raw[:, 0] - self.h * raw[:, 1]) / self.N
        m = np.linalg.norm(raw[:, 2 : 2 + self.n_components], axis=1) / self.N
        return e, m

    def _measure_grid(self):
        blocks = min((self.N + _THREADS - 1) // _THREADS, 64)
        return (blocks, self.R), (_THREADS,)

    def replica(self, r: int) -> np.ndarray:
        """Host copy of replica ``r``, shaped ``(L,)*dim (+ (n,))``."""
        return cp.asnumpy(self.spins[r]).reshape(self.shape + self.spins.shape[2:])

    def lattice_view(self) -> cp.ndarray:
        """Device spins reshaped to ``(R, L, ..., L[, n])``."""
        return self.spins.reshape((self.R,) + self.shape + self.spins.shape[2:])

    def __repr__(self) -> str:
        return f"GPU{self.name}(L={self.L}, dim={self.dim}, R={self.R}, {self.params()})"


class GPUIsing(GPUModel):
    cpu_model = IsingModel

    def _local(self):
        self._heatbath()

    def _initial_spins(self, init):
        if init == "random":
            s = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.R, self.N))
        elif init == "ordered":
            s = np.ones((self.R, self.N), dtype=np.int8)
        else:
            raise ValueError(init)
        return cp.asarray(s)

    def _heatbath(self):
        grid = _grid(self.R * self.N // 2)
        k = _kernel("ising_heatbath")
        for color in (0, 1):
            k(*grid, (self.spins, np.int32(self.R), np.int32(self.N), np.int32(self.L), np.int32(self.dim),
                      np.int32(color), self.beta, np.float32(self.J), np.float32(self.h), *self._next()))

    def _sw_bonds(self):
        _kernel("ising_sw_bonds")(*_grid(self.R * self.N), (
            self.spins, self._bonds, np.int32(self.R), np.int32(self.N), np.int32(self.L),
            np.int32(self.dim), self.beta, np.float32(self.J), *self._next()))

    def _sw_apply(self):
        total = self.R * self.N
        _kernel("ising_sw_apply")(*_grid(total), (self.spins, self._labels, np.int64(total), *self._next()))

    def _launch_measure(self, out):
        _kernel("ising_measure")(*self._measure_grid(), (
            self.spins, out, np.int32(self.N), np.int32(self.L), np.int32(self.dim)))


class GPUPotts(GPUModel):
    cpu_model = PottsModel
    n_components = 2

    def __init__(self, L, dim, temperatures, q=3, J=1.0, h=0.0, init="random", seed=None):
        if not 2 <= q <= 32:
            raise ValueError("2 <= q <= 32 (MAX_Q in kernels.cu)")
        self.q = int(q)
        super().__init__(L, dim, temperatures, J=J, h=h, init=init, seed=seed)

    def params(self):
        return {"q": self.q, **super().params()}

    def _local(self):
        self._heatbath()

    def _initial_spins(self, init):
        if init == "random":
            s = self.rng.integers(0, self.q, size=(self.R, self.N)).astype(np.int8)
        elif init == "ordered":
            s = np.zeros((self.R, self.N), dtype=np.int8)
        else:
            raise ValueError(init)
        return cp.asarray(s)

    def _heatbath(self):
        grid = _grid(self.R * self.N // 2)
        k = _kernel("potts_heatbath")
        for color in (0, 1):
            k(*grid, (self.spins, np.int32(self.R), np.int32(self.N), np.int32(self.L), np.int32(self.dim),
                      np.int32(color), np.int32(self.q), self.beta, np.float32(self.J), np.float32(self.h),
                      *self._next()))

    def _sw_bonds(self):
        _kernel("potts_sw_bonds")(*_grid(self.R * self.N), (
            self.spins, self._bonds, np.int32(self.R), np.int32(self.N), np.int32(self.L),
            np.int32(self.dim), self.beta, np.float32(self.J), *self._next()))

    def _sw_apply(self):
        total = self.R * self.N
        _kernel("potts_sw_apply")(*_grid(total), (self.spins, self._labels, np.int64(total),
                                                  np.int32(self.q), *self._next()))

    def _launch_measure(self, out):
        _kernel("potts_measure")(*self._measure_grid(), (
            self.spins, out, np.int32(self.N), np.int32(self.L), np.int32(self.dim), np.int32(self.q)))

    def counts(self) -> cp.ndarray:
        """Occupation of each state per replica, shape ``(R, q)``."""
        offsets = (cp.arange(self.R, dtype=cp.int64) * self.q)[:, None]
        flat = (self.spins.astype(cp.int64) + offsets).ravel()
        return cp.bincount(flat, minlength=self.R * self.q).reshape(self.R, self.q)


class GPUVector(GPUModel):
    """O(n) models with unit-vector spins, shape ``(R, N, n)`` float32."""

    target_acceptance = 0.5

    def __init__(self, L, dim, temperatures, J=1.0, h=0.0, init="random", seed=None, step_size=1.0):
        super().__init__(L, dim, temperatures, J=J, h=h, init=init, seed=seed)
        self.step_size = cp.full(self.R, step_size, dtype=cp.float32)
        self._accepted = None
        self._rvec = None
        self._sweeps_since_normalize = 0

    def _initial_spins(self, init):
        n = self.n_components
        if init == "random":
            s = self.rng.normal(size=(self.R, self.N, n))
            s /= np.linalg.norm(s, axis=2, keepdims=True)
        elif init == "ordered":
            s = np.zeros((self.R, self.N, n))
            s[..., 0] = 1.0
        else:
            raise ValueError(init)
        return cp.asarray(s, dtype=cp.float32)

    def _metropolis(self, adapt: bool = False):
        grid = _grid(self.R * self.N // 2)
        k = _kernel(f"on_metropolis<{self.n_components}>")
        if adapt and self._accepted is None:
            self._accepted = cp.empty(self.R * self.N // 2, dtype=cp.uint8)
        rate = 0.0
        for color in (0, 1):
            k(*grid, (self.spins, np.int32(self.R), np.int32(self.N), np.int32(self.L), np.int32(self.dim),
                      np.int32(color), self.beta, self.step_size, np.float32(self.J), np.float32(self.h),
                      *self._next(), self._accepted if adapt else np.uint64(0)))
            if adapt:
                rate = rate + self._accepted.reshape(self.R, -1).mean(axis=1, dtype=cp.float32) / 2
        if adapt:
            factor = cp.exp(rate - self.target_acceptance)
            self.step_size = cp.clip(self.step_size * factor, 0.05, 10.0).astype(cp.float32)
        self._maybe_normalize()

    def _local(self):
        self._metropolis()

    def _metropolis_adapt(self):
        """Metropolis sweep that also tunes each replica's step size
        (use during equilibration only)."""
        self._metropolis(adapt=True)

    def _sw_bonds(self):
        r = self.rng.normal(size=(self.R, self.n_components))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        self._rvec = cp.asarray(r, dtype=cp.float32)
        _kernel(f"on_sw_bonds<{self.n_components}>")(*_grid(self.R * self.N), (
            self.spins, self._rvec, self._bonds, np.int32(self.R), np.int32(self.N), np.int32(self.L),
            np.int32(self.dim), self.beta, np.float32(self.J), *self._next()))

    def _sw_apply(self):
        _kernel(f"on_sw_apply<{self.n_components}>")(*_grid(self.R * self.N), (
            self.spins, self._rvec, self._labels, np.int32(self.R), np.int32(self.N), *self._next()))
        self._maybe_normalize()

    def _maybe_normalize(self, every: int = 50):
        self._sweeps_since_normalize += 1
        if self._sweeps_since_normalize >= every:
            total = self.R * self.N
            _kernel(f"on_normalize<{self.n_components}>")(*_grid(total), (self.spins, np.int64(total)))
            self._sweeps_since_normalize = 0

    def _launch_measure(self, out):
        _kernel(f"on_measure<{self.n_components}>")(*self._measure_grid(), (
            self.spins, out, np.int32(self.N), np.int32(self.L), np.int32(self.dim)))

    @staticmethod
    def stiffness_terms(raw: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """From raw ``measure()`` output of an XY model, per replica and
        averaged over lattice directions: ``A = sum_bonds S_i.S_j`` and
        ``B2 = (sum_bonds (S_i x S_j)_z)^2``. The helicity modulus is
        ``Upsilon = J<A>/N - beta J^2 <B2>/N``."""
        return raw[:, 0] / dim, (raw[:, 5 : 5 + dim] ** 2).mean(axis=1)


class GPUXY(GPUVector):
    cpu_model = XYModel
    n_components = 2


class GPUHeisenberg(GPUVector):
    cpu_model = HeisenbergModel
    n_components = 3


GPU_MODELS = {"ising": GPUIsing, "potts": GPUPotts, "xy": GPUXY, "heisenberg": GPUHeisenberg}
