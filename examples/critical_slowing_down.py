"""Critical slowing down: Metropolis vs Wolff for the 2D Ising model at T_c.

The integrated autocorrelation time of |m| grows as tau ~ L^z, with
z ≈ 2.17 for local Metropolis updates, while Wolff clusters (z ≈ 0.25) keep it
nearly L-independent (time measured in sweeps of ~N spin updates).

    python examples/critical_slowing_down.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spinmodels import IsingModel, run
from spinmodels.plotting import SERIES_COLORS, _STYLE

OUT = "output"
os.makedirs(OUT, exist_ok=True)

Tc = IsingModel.critical_temperature(2)
sizes = np.array([8, 12, 16, 24, 32])

with plt.rc_context(_STYLE):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for algorithm, color in zip(("metropolis", "wolff"), SERIES_COLORS):
        taus = []
        for L in sizes:
            model = IsingModel(L, dim=2, seed=int(L))
            ts = run(model, Tc, n_equil=2000, n_measure=40000, algorithm=algorithm)
            taus.append(ts.tau()[1])
            print(f"{algorithm:10s} L={L:3d}  tau_|m| = {taus[-1]:8.2f} sweeps")
        z, logA = np.polyfit(np.log(sizes), np.log(taus), 1)
        ax.loglog(sizes, taus, "o", color=color, ms=6, label=f"{algorithm}  (z ≈ {z:.2f})")
        ax.loglog(sizes, np.exp(logA) * sizes**z, "-", color=color, lw=1.5)
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}(|m|)$ [sweeps]")
    ax.set_title(r"2D Ising at $T_c$: critical slowing down")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/critical_slowing_down.png", dpi=150)
print(f"Saved {OUT}/critical_slowing_down.png")
