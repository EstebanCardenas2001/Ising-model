"""Simulation drivers: single-temperature runs and temperature scans."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .models.base import LatticeModel
from .observables import OBSERVABLES, integrated_autocorr_time, thermodynamics


@dataclass
class TimeSeries:
    """Raw measurements from one Markov chain at fixed temperature."""

    T: float
    energy: np.ndarray  # E / N per measurement
    magnetization: np.ndarray  # |m| per measurement
    sweep_stat: float  # mean acceptance rate (Metropolis) or cluster fraction (Wolff)

    def tau(self) -> tuple[float, float]:
        """Integrated autocorrelation times of ``(energy, magnetization)``,
        in units of measurements."""
        return integrated_autocorr_time(self.energy), integrated_autocorr_time(self.magnetization)


def run(
    model: LatticeModel,
    T: float,
    n_equil: int = 1000,
    n_measure: int = 5000,
    measure_every: int = 1,
    algorithm: str = "metropolis",
    adapt: bool = True,
) -> TimeSeries:
    """Equilibrate ``model`` at temperature ``T`` and record a time series.

    The model is updated in place, so consecutive calls at nearby
    temperatures continue from the previous equilibrium state.
    """
    for _ in range(n_equil):
        stat = model.sweep(T, algorithm)
        if adapt:
            model.adapt(algorithm, stat)

    e = np.empty(n_measure)
    m = np.empty(n_measure)
    stats = 0.0
    for k in range(n_measure):
        for _ in range(measure_every):
            stats += model.sweep(T, algorithm)
        e[k], m[k] = model.observe()

    model.renormalize()
    return TimeSeries(T=T, energy=e, magnetization=m, sweep_stat=stats / (n_measure * measure_every))


@dataclass
class ScanResult:
    """Observables (with errors) as a function of temperature for one system."""

    model: str
    L: int
    dim: int
    algorithm: str
    params: dict
    temperatures: np.ndarray
    mean: dict[str, np.ndarray]
    error: dict[str, np.ndarray]
    tau_energy: np.ndarray
    tau_magnetization: np.ndarray
    sweep_stat: np.ndarray
    series: list[TimeSeries] = field(default_factory=list, repr=False)

    @property
    def N(self) -> int:
        return self.L**self.dim

    @property
    def label(self) -> str:
        extra = f", q={self.params['q']}" if "q" in self.params else ""
        return f"{self.model} {self.dim}D{extra}, L={self.L}"

    def save(self, path: str) -> None:
        """Save to ``.npz`` (raw time series are not stored)."""
        arrays = {"temperatures": self.temperatures}
        for k in OBSERVABLES:
            arrays[f"mean_{k}"] = self.mean[k]
            arrays[f"error_{k}"] = self.error[k]
        np.savez(
            path,
            **arrays,
            tau_energy=self.tau_energy,
            tau_magnetization=self.tau_magnetization,
            sweep_stat=self.sweep_stat,
            meta=np.array(
                [repr({"model": self.model, "L": self.L, "dim": self.dim,
                       "algorithm": self.algorithm, "params": self.params})]
            ),
        )

    @classmethod
    def load(cls, path: str) -> "ScanResult":
        import ast

        d = np.load(path)
        meta = ast.literal_eval(str(d["meta"][0]))
        return cls(
            **meta,
            temperatures=d["temperatures"],
            mean={k: d[f"mean_{k}"] for k in OBSERVABLES},
            error={k: d[f"error_{k}"] for k in OBSERVABLES},
            tau_energy=d["tau_energy"],
            tau_magnetization=d["tau_magnetization"],
            sweep_stat=d["sweep_stat"],
        )


def temperature_scan(
    model: LatticeModel,
    temperatures,
    n_equil: int = 1000,
    n_measure: int = 5000,
    measure_every: int = 1,
    algorithm: str = "metropolis",
    n_blocks: int = 20,
    anneal: bool = True,
    keep_series: bool = False,
    verbose: bool = True,
) -> ScanResult:
    """Measure all observables over a range of temperatures.

    With ``anneal=True`` temperatures are visited from hot to cold, reusing
    the configuration between points (simulated annealing), which shortens
    equilibration in the ordered phase. Results are returned sorted by T.
    """
    temps = np.sort(np.asarray(temperatures, dtype=float))
    order = temps[::-1] if anneal else temps

    results: dict[float, tuple[dict, tuple[float, float], float, TimeSeries]] = {}
    t0 = time.perf_counter()
    for T in order:
        ts = run(model, T, n_equil, n_measure, measure_every, algorithm)
        obs = thermodynamics(ts.energy, ts.magnetization, T, model.N, n_blocks)
        results[T] = (obs, ts.tau(), ts.sweep_stat, ts)
        if verbose:
            print(
                f"[{model.name} L={model.L}] T={T:.4f}  e={obs['energy'][0]:+.5f}  "
                f"|m|={obs['magnetization'][0]:.4f}  U4={obs['binder'][0]:.4f}  "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    return ScanResult(
        model=model.name,
        L=model.L,
        dim=model.dim,
        algorithm=algorithm,
        params=model.params(),
        temperatures=temps,
        mean={k: np.array([results[T][0][k][0] for T in temps]) for k in OBSERVABLES},
        error={k: np.array([results[T][0][k][1] for T in temps]) for k in OBSERVABLES},
        tau_energy=np.array([results[T][1][0] for T in temps]),
        tau_magnetization=np.array([results[T][1][1] for T in temps]),
        sweep_stat=np.array([results[T][2] for T in temps]),
        series=[results[T][3] for T in temps] if keep_series else [],
    )
