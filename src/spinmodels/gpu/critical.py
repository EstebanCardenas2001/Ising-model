"""GPU data collection for finite-size scaling: many independent chains of
one lattice size at one temperature, simulated as a single batch."""

from __future__ import annotations

import time

import numpy as np

from ..critical import CriticalData
from .engine import GPU_MODELS, GPUModel
from .scan import DEFAULT_STEP, _mc_step


def collect_gpu(
    model: str | type[GPUModel],
    L: int,
    dim: int,
    T0: float,
    n_equil: int = 2000,
    n_measure: int = 20000,
    n_chains: int | None = None,
    steps=DEFAULT_STEP,
    target_sites: int = 2**23,
    seed: int | None = None,
    verbose: bool = True,
    **model_kwargs,
) -> CriticalData:
    """Sample ``n_chains`` independent replicas at temperature ``T0``.

    By default the number of chains fills ``target_sites`` lattice sites
    (between 16 and 512 chains), so small lattices still saturate the GPU
    and every size gets plenty of independent samples. Each chain is one
    jackknife block of the returned :class:`~spinmodels.critical.CriticalData`.
    """
    cls = GPU_MODELS[model] if isinstance(model, str) else model
    N = L**dim
    if n_chains is None:
        n_chains = int(np.clip(target_sites // N, 16, 512))
    sim = cls(L, dim, np.full(n_chains, T0), seed=seed, **model_kwargs)
    t0 = time.perf_counter()
    for _ in range(n_equil):
        _mc_step(sim, steps, equilibrating=True)
    e = np.empty((n_measure, n_chains), dtype=np.float32)
    m = np.empty((n_measure, n_chains), dtype=np.float32)
    for k in range(n_measure):
        _mc_step(sim, steps, equilibrating=False)
        e[k], m[k] = sim.observe()
    elapsed = time.perf_counter() - t0
    if verbose:
        rate = (n_equil + n_measure) * sum(n for _, n in steps) * n_chains * N / elapsed / 1e9
        print(f"[GPU critical {sim.name} {dim}D L={L}] T0={T0:.5f}, {n_chains} chains x {n_measure} samples, "
              f"{elapsed:.1f}s ({rate:.2f} G site-sweeps/s)", flush=True)
    return CriticalData(L, dim, T0, e, m, dict(model=sim.name, steps=str(steps), **sim.params()))
