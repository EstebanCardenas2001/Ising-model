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
    #: Additional per-temperature arrays (e.g. raw energy samples for
    #: histograms, swap rates). Saved and loaded with the result.
    extras: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def N(self) -> int:
        return self.L**self.dim

    @property
    def label(self) -> str:
        extra = f", q={self.params['q']}" if "q" in self.params else ""
        return f"{self.model} {self.dim}D{extra}, L={self.L}"

    def save(self, path: str) -> None:
        """Save to ``.npz`` (``series`` is not stored; ``extras`` is)."""
        arrays = {"temperatures": self.temperatures}
        for k in self.mean:
            arrays[f"mean_{k}"] = self.mean[k]
            arrays[f"error_{k}"] = self.error[k]
        for k, v in self.extras.items():
            arrays[f"extra_{k}"] = v
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
        names = [k[len("mean_"):] for k in d.files if k.startswith("mean_")]
        return cls(
            **meta,
            temperatures=d["temperatures"],
            mean={k: d[f"mean_{k}"] for k in names},
            error={k: d[f"error_{k}"] for k in names},
            tau_energy=d["tau_energy"],
            tau_magnetization=d["tau_magnetization"],
            sweep_stat=d["sweep_stat"],
            extras={k[len("extra_"):]: d[k] for k in d.files if k.startswith("extra_")},
        )


def _run_independent(args) -> TimeSeries:
    """Worker for parallel scans: an independent chain at one temperature."""
    model, T, seed, kwargs = args
    model.reseed(seed)
    return run(model, T, **kwargs)


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
    n_workers: int = 1,
) -> ScanResult:
    """Measure all observables over a range of temperatures.

    With ``anneal=True`` temperatures are visited from hot to cold, reusing
    the configuration between points (simulated annealing), which shortens
    equilibration in the ordered phase. Results are returned sorted by T.

    With ``n_workers > 1`` every temperature is instead an independent chain
    started from ``model``'s current state, run in a pool of worker
    processes (``anneal`` is ignored; give enough ``n_equil``).
    ``n_workers=-1`` uses all CPU cores.
    """
    temps = np.sort(np.asarray(temperatures, dtype=float))
    order = temps[::-1] if anneal else temps
    t0 = time.perf_counter()

    if n_workers != 1:
        import os
        from concurrent.futures import ProcessPoolExecutor

        workers = os.cpu_count() if n_workers == -1 else n_workers
        seeds = model.rng.integers(0, 2**32 - 1, size=len(temps))
        kwargs = dict(n_equil=n_equil, n_measure=n_measure, measure_every=measure_every, algorithm=algorithm)
        with ProcessPoolExecutor(max_workers=min(workers, len(temps))) as pool:
            series = list(pool.map(_run_independent, [(model, T, int(sd), kwargs) for T, sd in zip(temps, seeds)]))
        order = temps
    else:
        series = []
        for T in order:
            series.append(run(model, T, n_equil, n_measure, measure_every, algorithm))

    results: dict[float, tuple[dict, tuple[float, float], float, TimeSeries]] = {}
    for T, ts in zip(order, series):
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


def parallel_tempering(
    model: LatticeModel,
    temperatures,
    n_equil: int = 1000,
    n_measure: int = 5000,
    algorithm: str = "metropolis",
    n_blocks: int = 20,
    verbose: bool = True,
) -> ScanResult:
    """Replica-exchange Monte Carlo on the CPU.

    One copy of ``model`` per temperature; after every sweep, neighboring
    temperatures attempt to exchange configurations with probability
    ``min(1, exp[(beta_k - beta_{k+1}) (E_k - E_{k+1})])`` (alternating
    even/odd pairs). Exchanges let configurations trapped in metastable
    states at low T escape via high T, which helps at first-order
    transitions and in rugged energy landscapes.

    The swap acceptance rates are returned in ``result.extras``; a ladder
    with rates well above ~0.2 everywhere is well spaced.
    For large batches of replicas see :func:`spinmodels.gpu.scan.gpu_temperature_scan`.
    """
    import copy

    temps = np.sort(np.asarray(temperatures, dtype=float))
    nT = len(temps)
    betas = 1.0 / temps
    replicas = [copy.deepcopy(model) for _ in temps]  # replicas[k] is at temps[k]
    energies = np.array([r.energy() for r in replicas])
    attempts = np.zeros(max(nT - 1, 1))
    accepts = np.zeros(max(nT - 1, 1))
    e = np.empty((n_measure, nT))
    m = np.empty((n_measure, nT))
    t0 = time.perf_counter()

    for step in range(n_equil + n_measure):
        for k, rep in enumerate(replicas):
            stat = rep.sweep(temps[k], algorithm)
            if step < n_equil:
                rep.adapt(algorithm, stat)
            energies[k] = rep.energy()
        for k in range(step % 2, nT - 1, 2):
            delta = (betas[k] - betas[k + 1]) * (energies[k] - energies[k + 1])
            attempts[k] += 1
            if delta >= 0 or model.rng.random() < np.exp(delta):
                accepts[k] += 1
                # Exchange configurations (and tuned step sizes stay with T).
                replicas[k].spins, replicas[k + 1].spins = replicas[k + 1].spins, replicas[k].spins
                energies[k], energies[k + 1] = energies[k + 1], energies[k]
        if step >= n_equil:
            i = step - n_equil
            e[i] = energies / model.N
            m[i] = [r.magnetization() for r in replicas]

    for r in replicas:
        r.renormalize()
    mean = {k: np.empty(nT) for k in OBSERVABLES}
    error = {k: np.empty(nT) for k in OBSERVABLES}
    for j, T in enumerate(temps):
        for k, (v, err) in thermodynamics(e[:, j], m[:, j], T, model.N, n_blocks).items():
            mean[k][j], error[k][j] = v, err
    if verbose:
        print(f"[PT {model.name} L={model.L}] {nT} replicas, {time.perf_counter() - t0:.1f}s, "
              f"swap acceptance min {(accepts / np.maximum(attempts, 1)).min():.2f}", flush=True)
    return ScanResult(
        model=model.name,
        L=model.L,
        dim=model.dim,
        algorithm=f"{algorithm}+PT",
        params=model.params(),
        temperatures=temps,
        mean=mean,
        error=error,
        tau_energy=np.array([integrated_autocorr_time(e[:, j]) for j in range(nT)]),
        tau_magnetization=np.array([integrated_autocorr_time(m[:, j]) for j in range(nT)]),
        sweep_stat=np.full(nT, np.nan),
        extras={"swap_acceptance": accepts / np.maximum(attempts, 1)},
    )
