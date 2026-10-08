"""Spin configurations of all four models below, near and above T_c.

    python examples/snapshots.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spinmodels import HeisenbergModel, IsingModel, PottsModel, XYModel
from spinmodels.plotting import plot_configuration

OUT = "output"
os.makedirs(OUT, exist_ok=True)

cases = [
    (lambda: IsingModel(64, dim=2, seed=0), IsingModel.critical_temperature(2)),
    (lambda: PottsModel(64, dim=2, q=3, seed=0), PottsModel.critical_temperature(2, q=3)),
    (lambda: XYModel(32, dim=2, seed=0), XYModel.critical_temperature(2)),
    (lambda: HeisenbergModel(16, dim=3, seed=0), HeisenbergModel.critical_temperature(3)),
]
ratios = (0.6, 1.0, 1.6)

fig, axes = plt.subplots(len(cases), len(ratios), figsize=(4.2 * len(ratios), 4.0 * len(cases)))
for row, (make, Tc) in zip(axes, cases):
    for ax, r in zip(row, ratios):
        model = make()
        T = r * Tc
        for _ in range(300):
            model.sweep(T, "wolff")
        plot_configuration(model, ax=ax)
        ax.set_title(f"{model.name} {model.dim}D, T = {r:g} $T_c$")
fig.tight_layout()
fig.savefig(f"{OUT}/snapshots.png", dpi=120)
print(f"Saved {OUT}/snapshots.png")
