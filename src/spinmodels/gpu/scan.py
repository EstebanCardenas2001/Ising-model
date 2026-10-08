"""GPU temperature scans: every temperature (times several independent
chains) is a replica of one batched GPU system, optionally coupled by
parallel tempering (replica exchange)."""

from __future__ import annotations

import time

import cupy as cp
import numpy as np

from ..observables import integrated_autocorr_time, jackknife, thermodynamics
from ..simulation import ScanResult
from .engine import GPU_MODELS, GPUModel, GPUVector, GPUXY

#: Default Monte Carlo step: one Swendsen-Wang cluster update followed by one
#: local (heat-bath / Metropolis) sweep. Clusters defeat critical slowing
#: down; the local sweep efficiently relaxes short-wavelength modes.
DEFAULT_STEP = (("swendsen_wang", 1), ("local", 1))


class ParallelTempering:
    """Replica-exchange bookkeeping for ``n_chains`` independent PT chains
    over the same sorted temperature ladder, all inside one :class:`GPUModel`.

    Configurations never move on the device; instead the temperature
    assignment is permuted. ``replica_at[c, k]`` is the replica of chain
    ``c`` currently at temperature ``k``.
    """

    def __init__(self, model: GPUModel, temperatures, n_chains: int = 1):
        self.model = model
        self.temperatures = np.sort(np.asarray(temperatures, dtype=float))
        self.nT = len(self.temperatures)
        if self.nT * n_chains != model.R:
            raise ValueError("model.R must equal n_chains * len(temperatures)")
        self.n_chains = n_chains
        self.replica_at = np.arange(model.R).reshape(n_chains, self.nT)
        self.attempts = np.zeros(self.nT - 1)
        self.accepts = np.zeros(self.nT - 1)
        self._parity = 0
        self._apply()

    def _apply(self) -> None:
        T_of_replica = np.empty(self.model.R)
        T_of_replica[self.replica_at] = self.temperatures[None, :]
        self.model.set_temperatures(T_of_replica)

    def swap(self, energies: np.ndarray, enabled: bool = True) -> None:
        """Attempt exchanges between neighboring temperatures (alternating
        even/odd pairs) in every chain, given each replica's total energy."""
        if not enabled or self.nT < 2:
            return
        beta = 1.0 / self.temperatures
        ks = np.arange(self._parity, self.nT - 1, 2)
        self._parity ^= 1
        if len(ks) == 0:
            return
        a = self.replica_at[:, ks]
        b = self.replica_at[:, ks + 1]
        delta = (beta[ks] - beta[ks + 1]) * (energies[a] - energies[b])
        accept = self.model.rng.random(delta.shape) < np.exp(np.minimum(delta, 0.0))
        self.attempts[ks] += self.n_chains
        self.accepts[ks] += accept.sum(axis=0)
        if not accept.any():
            return
        before = self.replica_at.copy()
        self.replica_at[:, ks] = np.where(accept, b, a)
        self.replica_at[:, ks + 1] = np.where(accept, a, b)
        if isinstance(self.model, GPUVector):
            # Metropolis step sizes are tuned per temperature: the replica now
            # at T_k inherits the step size of the one previously there.
            perm = np.empty(self.model.R, dtype=np.int64)
            perm[self.replica_at.ravel()] = before.ravel()
            self.model.step_size = self.model.step_size[cp.asarray(perm)]
        self._apply()

    @property
    def acceptance(self) -> np.ndarray:
        return self.accepts / np.maximum(self.attempts, 1)


def _mc_step(model: GPUModel, steps, equilibrating: bool) -> None:
    for algorithm, n in steps:
        if equilibrating and algorithm in ("local", "metropolis") and isinstance(model, GPUVector):
            algorithm = "metropolis_adapt"
        model.sweep(algorithm, n)


def gpu_temperature_scan(
    model: str | type[GPUModel],
    L: int,
    dim: int,
    temperatures,
    n_equil: int = 1000,
    n_measure: int = 5000,
    n_chains: int = 1,
    steps=DEFAULT_STEP,
    measure_every: int = 1,
    parallel_tempering: bool = True,
    n_blocks: int = 20,
    keep_samples: int = 4000,
    seed: int | None = None,
    init: str = "random",
    verbose: bool = True,
    **model_kwargs,
) -> ScanResult:
    """Equilibrate and measure a model at many temperatures at once on the GPU.

    Parameters
    ----------
    model : GPU model class or registry name ("ising", "potts", "xy", "heisenberg").
    L, dim : lattice size (even) and dimension.
    temperatures : temperatures to scan.
    n_chains : independent copies of the whole temperature ladder simulated
        in the same batch; their samples are pooled (more statistics at no
        extra wall time when the lattice alone does not fill the GPU).
    steps : sequence of ``(algorithm, n_sweeps)`` making up one MC step.
    parallel_tempering : attempt replica exchanges after every MC step.
    keep_samples : raw ``(E/N, |m|)`` samples per temperature kept in
        ``result.extras`` for histograms (evenly thinned, chains pooled).
    model_kwargs : e.g. ``q=8`` for Potts, ``J``.

    Returns a :class:`~spinmodels.simulation.ScanResult` with the five
    standard observables (plus ``"stiffness"``, the helicity modulus, for XY).
    """
    cls = GPU_MODELS[model] if isinstance(model, str) else model
    temps = np.sort(np.asarray(temperatures, dtype=float))
    nT = len(temps)
    sim = cls(L, dim, np.tile(temps, n_chains), seed=seed, init=init, **model_kwargs)
    pt = ParallelTempering(sim, temps, n_chains)
    is_xy = isinstance(sim, GPUXY)

    t0 = time.perf_counter()
    for _ in range(n_equil):
        _mc_step(sim, steps, equilibrating=True)
        pt.swap(sim.energies(), enabled=parallel_tempering)

    shape = (n_measure, n_chains, nT)
    e = np.empty(shape)
    m = np.empty(shape)
    A = np.empty(shape) if is_xy else None
    B2 = np.empty(shape) if is_xy else None
    for k in range(n_measure):
        for _ in range(measure_every):
            _mc_step(sim, steps, equilibrating=False)
        raw = sim.measure()
        energies = -sim.J * raw[:, 0] - sim.h * raw[:, 1]
        idx = pt.replica_at
        e[k] = energies[idx] / sim.N
        m[k] = np.linalg.norm(raw[idx, 2 : 2 + sim.n_components], axis=-1) / sim.N
        if is_xy:
            a, b2 = GPUVector.stiffness_terms(raw, sim.dim)
            A[k], B2[k] = a[idx], b2[idx]
        pt.swap(energies, enabled=parallel_tempering)
    elapsed = time.perf_counter() - t0

    # Pool chains: concatenate each chain's time series (chain-major), so
    # jackknife blocks stay (mostly) within one chain.
    def pooled(x):
        return x.transpose(1, 0, 2).reshape(n_chains * n_measure, nT)

    e, m = pooled(e), pooled(m)
    names = ["energy", "magnetization", "specific_heat", "susceptibility", "binder"]
    mean = {k: np.empty(nT) for k in names}
    error = {k: np.empty(nT) for k in names}
    blocks = max(n_blocks, n_chains)
    for j, T in enumerate(temps):
        obs = thermodynamics(e[:, j], m[:, j], T, sim.N, blocks)
        for k in names:
            mean[k][j], error[k][j] = obs[k]
    if is_xy:
        A, B2 = pooled(A), pooled(B2)
        mean["stiffness"] = np.empty(nT)
        error["stiffness"] = np.empty(nT)
        J, N = sim.J, sim.N
        for j, T in enumerate(temps):
            est = lambda a, b2, T=T: (J * a.mean() - J**2 / T * b2.mean()) / N  # noqa: E731
            mean["stiffness"][j], error["stiffness"][j] = jackknife(est, A[:, j], B2[:, j], n_blocks=blocks)

    thin = max(1, len(e) // keep_samples)
    extras = {
        "energy_samples": e[::thin].astype(np.float32),
        "magnetization_samples": m[::thin].astype(np.float32),
        "swap_acceptance": pt.acceptance,
    }
    algorithm = "+".join(f"{a}x{n}" if n > 1 else a for a, n in steps)
    if parallel_tempering:
        algorithm += "+PT"
    first_chain = slice(0, n_measure)
    result = ScanResult(
        model=sim.name,
        L=sim.L,
        dim=sim.dim,
        algorithm=algorithm,
        params=sim.params(),
        temperatures=temps,
        mean=mean,
        error=error,
        tau_energy=np.array([integrated_autocorr_time(e[first_chain, j]) for j in range(nT)]),
        tau_magnetization=np.array([integrated_autocorr_time(m[first_chain, j]) for j in range(nT)]),
        sweep_stat=np.full(nT, np.nan),
        extras=extras,
    )
    if verbose:
        sweeps = (n_equil + n_measure * measure_every) * sum(n for _, n in steps)
        rate = sweeps * sim.R * sim.N / elapsed / 1e9
        pt_msg = f", PT acceptance min {pt.acceptance.min():.2f}" if parallel_tempering else ""
        print(f"[GPU {result.label}] {nT} T x {n_chains} chains, {elapsed:.1f}s "
              f"({rate:.2f} G site-sweeps/s{pt_msg})", flush=True)
    return result
