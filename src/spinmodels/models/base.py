"""Abstract base class shared by all lattice spin models.

Extending the framework
-----------------------
* **New model**: subclass :class:`LatticeModel` and implement
  ``_initial_spins``, ``energy``, ``magnetization_vector`` and
  ``_metropolis_sweep``.
* **New update algorithm**: add a method ``_<name>_sweep(self, beta)`` to a
  model. It is then available as ``model.sweep(T, algorithm="<name>")`` and
  listed by ``model.algorithms()`` — no changes to the simulation driver are
  needed. Wolff cluster updates are implemented this way.

Conventions: ``k_B = 1``, energies in units of ``J``, nearest-neighbor
couplings on a periodic hypercubic lattice, and the Hamiltonian

    H = -J * sum_<ij> f(s_i, s_j) - h * sum_i g(s_i)

with model-specific ``f`` and ``g``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .._rng import seed_numba
from ..lattice import HypercubicLattice


class LatticeModel(ABC):
    """Base class for classical spin models on a periodic hypercubic lattice.

    Parameters
    ----------
    L : int
        Linear system size (``N = L**dim`` sites).
    dim : int
        Spatial dimension.
    J : float
        Nearest-neighbor coupling (``J > 0`` is ferromagnetic).
    h : float
        Uniform external field.
    init : {"random", "ordered"}
        Initial configuration (hot or cold start).
    seed : int or None
        Seed for both NumPy (initial state) and Numba (Monte Carlo moves).
        Numba keeps a single global RNG stream, so models created later
        re-seed it.
    """

    name: ClassVar[str] = "LatticeModel"
    #: Number of components of the order parameter vector.
    n_components: ClassVar[int] = 1

    def __init__(
        self,
        L: int,
        dim: int,
        J: float = 1.0,
        h: float = 0.0,
        init: str = "random",
        seed: int | None = None,
    ):
        self.lattice = HypercubicLattice(L, dim)
        self.J = float(J)
        self.h = float(h)
        self.rng = np.random.default_rng(seed)
        seed_numba(int(self.rng.integers(2**32 - 1)))
        self.spins = self._initial_spins(init)
        # Work buffers shared by cluster algorithms (stack/queue + membership).
        self._cluster = np.empty(self.N, dtype=np.int64)
        self._in_cluster = np.zeros(self.N, dtype=np.bool_)
        #: Wolff clusters per sweep. Must stay fixed while measuring: a
        #: data-dependent rule such as "grow clusters until N spins have
        #: flipped" makes the last cluster before each measurement
        #: size-biased and skews the averages. Tuned by ``adapt``.
        self.wolff_clusters = 1
        self._mean_cluster_fraction: float | None = None

    # ------------------------------------------------------------------ #
    # Geometry shortcuts
    # ------------------------------------------------------------------ #
    @property
    def L(self) -> int:
        return self.lattice.L

    @property
    def dim(self) -> int:
        return self.lattice.dim

    @property
    def N(self) -> int:
        return self.lattice.N

    @property
    def neighbors(self) -> np.ndarray:
        return self.lattice.neighbors

    # ------------------------------------------------------------------ #
    # Model-specific interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initial_spins(self, init: str) -> np.ndarray:
        """Return the initial spin array for ``init`` in {"random", "ordered"}."""

    @abstractmethod
    def energy(self) -> float:
        """Total energy ``H`` of the current configuration."""

    @abstractmethod
    def magnetization_vector(self) -> np.ndarray:
        """Order-parameter vector per site, shape ``(n_components,)``."""

    @abstractmethod
    def _metropolis_sweep(self, beta: float) -> float:
        """Perform ``N`` single-spin Metropolis attempts; return acceptance rate."""

    def params(self) -> dict:
        """Model parameters, used for labelling and saving results."""
        return {"J": self.J, "h": self.h}

    @classmethod
    def critical_temperature(cls, dim: int, **params) -> float | None:
        """Known (exact or best numerical) critical temperature, if any."""
        return None

    # ------------------------------------------------------------------ #
    # Generic functionality
    # ------------------------------------------------------------------ #
    def magnetization(self) -> float:
        """Magnitude of the order parameter per site, ``|m|``."""
        return float(np.linalg.norm(self.magnetization_vector()))

    def observe(self) -> tuple[float, float]:
        """Return ``(E/N, |m|)`` for the current configuration."""
        return self.energy() / self.N, self.magnetization()

    @classmethod
    def algorithms(cls) -> list[str]:
        """Names of the update algorithms this model supports."""
        return sorted(
            attr[1:-len("_sweep")]
            for attr in dir(cls)
            if attr.startswith("_") and not attr.startswith("__") and attr.endswith("_sweep")
        )

    def sweep(self, T: float, algorithm: str = "metropolis") -> float:
        """Advance the Markov chain by one Monte Carlo sweep at temperature ``T``.

        One sweep is ``N`` attempted single-spin updates for Metropolis, or
        ``wolff_clusters`` Wolff clusters (tuned during equilibration to flip
        ~N spins on average), so that both measure time in comparable units.

        Returns an algorithm-specific diagnostic: the acceptance rate for
        Metropolis, the mean cluster size divided by ``N`` for Wolff.
        """
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")
        method = getattr(self, f"_{algorithm}_sweep", None)
        if method is None:
            raise ValueError(
                f"{self.name} does not support algorithm {algorithm!r}; "
                f"available: {self.algorithms()}"
            )
        return method(1.0 / T)

    def adapt(self, algorithm: str, stat: float) -> None:
        """Tune algorithm parameters from the diagnostic returned by ``sweep``.

        Called by the simulation driver during equilibration only: adapting
        while measuring would break detailed balance / bias the averages.
        """
        if algorithm == "metropolis":
            self.adapt_step_size(stat)
        elif algorithm == "wolff":
            # Exponential moving average of <|C|>/N; aim for ~N flips per sweep.
            f = self._mean_cluster_fraction
            f = stat if f is None else 0.9 * f + 0.1 * stat
            self._mean_cluster_fraction = f
            self.wolff_clusters = max(1, int(round(1.0 / f)))

    def adapt_step_size(self, acceptance: float) -> None:
        """Hook for continuous-spin models to tune their Metropolis proposal.

        Discrete models need no tuning.
        """

    def reseed(self, seed: int) -> None:
        """Reset the NumPy generator and Numba's stream (e.g. in a worker process)."""
        self.rng = np.random.default_rng(seed)
        seed_numba(int(self.rng.integers(2**32 - 1)))

    def renormalize(self) -> None:
        """Hook to remove floating-point drift from spin constraints (no-op here)."""

    def _check_cluster_valid(self) -> None:
        """Cluster updates as implemented require a ferromagnet in zero field."""
        if self.h != 0.0:
            raise ValueError("Cluster updates require h = 0 (use Metropolis with a field).")
        if self.J <= 0.0:
            raise ValueError("Cluster updates require a ferromagnetic coupling J > 0.")

    def set_state(self, spins: np.ndarray) -> None:
        """Replace the configuration (shape and dtype are validated)."""
        spins = np.asarray(spins, dtype=self.spins.dtype)
        if spins.shape != self.spins.shape:
            raise ValueError(f"Expected shape {self.spins.shape}, got {spins.shape}")
        self.spins = spins.copy()

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{type(self).__name__}(L={self.L}, dim={self.dim}, {params})"
