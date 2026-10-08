"""2D Ising model: locate T_c from the Binder-cumulant crossing.

Runs Wolff temperature scans for several system sizes and plots all five
observables. The U4(T) curves for different L cross close to the exact
Onsager temperature T_c = 2 / ln(1 + sqrt 2) ≈ 2.269.

    python examples/ising2d_finite_size.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

from spinmodels import IsingModel, temperature_scan
from spinmodels.plotting import plot_scan

OUT = "output"
os.makedirs(OUT, exist_ok=True)

temperatures = np.linspace(1.8, 2.8, 21)
results = []
for L in (8, 16, 32):
    model = IsingModel(L, dim=2, seed=L)
    results.append(
        temperature_scan(model, temperatures, n_equil=500, n_measure=5000, algorithm="wolff")
    )
    results[-1].save(f"{OUT}/ising2d_L{L}.npz")

Tc = IsingModel.critical_temperature(2)
fig, _ = plot_scan(results, Tc=Tc, title="2D Ising model, Wolff updates (dashed: exact $T_c$)")
fig.savefig(f"{OUT}/ising2d_finite_size.png", dpi=150)
print(f"Saved {OUT}/ising2d_finite_size.png")
