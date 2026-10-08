"""Matplotlib visualisation of scan results and spin configurations."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .models import IsingModel, LatticeModel, PottsModel, VectorModel, XYModel
from .observables import LABELS, OBSERVABLES
from .simulation import ScanResult

#: Categorical series colors, assigned in fixed order (never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d9d6",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2b2a",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "legend.frameon": False,
}


def plot_scan(
    results: ScanResult | Sequence[ScanResult],
    observables: Iterable[str] = OBSERVABLES,
    Tc: float | None = None,
    title: str | None = None,
    ncols: int = 3,
):
    """Plot observables vs temperature, one panel each, one series per result.

    ``Tc`` (if given) is drawn as a dashed reference line on every panel.
    Returns ``(fig, axes)``.
    """
    if isinstance(results, ScanResult):
        results = [results]
    if len(results) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} series per figure; use several figures.")
    observables = list(observables)
    nrows = int(np.ceil(len(observables) / ncols))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
        for ax, obs in zip(axes.flat, observables):
            for res, color in zip(results, SERIES_COLORS):
                ax.errorbar(
                    res.temperatures, res.mean[obs], yerr=res.error[obs],
                    color=color, lw=1.5, marker="o", ms=4, capsize=2, elinewidth=1,
                    label=f"L = {res.L}",
                )
            if Tc is not None:
                ax.axvline(Tc, color="#8a8984", ls="--", lw=1)
            ax.set_xlabel("$T$")
            ax.set_ylabel(LABELS[obs])
        for ax in list(axes.flat)[len(observables):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if len(results) > 1:
            fig.legend(handles, labels, loc="upper right", ncols=len(results))
        fig.suptitle(title or results[0].label.rsplit(",", 1)[0] + f" ({results[0].algorithm})")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes


def _plane(model: LatticeModel, field: np.ndarray) -> np.ndarray:
    """Return a 2D slice of a per-site field (first slice for dim >= 3)."""
    grid = model.lattice.reshape(field)
    if model.dim == 1:
        return grid[np.newaxis, ...]
    while grid.ndim - (field.ndim - 1) > 2:
        grid = grid[0]
    return grid


def plot_configuration(model: LatticeModel, ax=None, arrows: bool | None = None):
    """Draw the current spin configuration (a 2D slice for 3D lattices)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    show_arrows = arrows if arrows is not None else model.L <= 32

    if isinstance(model, IsingModel):
        img = _plane(model, model.spins)
        ax.imshow(img, cmap=ListedColormap(SERIES_COLORS[:2]), vmin=-1, vmax=1,
                  interpolation="nearest")
    elif isinstance(model, PottsModel):
        img = _plane(model, model.spins)
        colors = SERIES_COLORS if model.q <= len(SERIES_COLORS) else plt.cm.tab20.colors
        ax.imshow(img, cmap=ListedColormap(colors[: model.q]), vmin=-0.5,
                  vmax=model.q - 0.5, interpolation="nearest")
    elif isinstance(model, VectorModel):
        if isinstance(model, XYModel):
            img = _plane(model, model.angles())
            im = ax.imshow(img, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
            label = r"$\theta$"
        else:
            # Project onto the magnetization direction: the order is visible
            # whatever direction the system happened to order along.
            m = model.magnetization_vector()
            norm = np.linalg.norm(m)
            axis = m / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            img = _plane(model, model.spins @ axis)
            im = ax.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            label = r"$\mathbf{S}\cdot\hat{\mathbf{m}}$"
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        if show_arrows and model.dim >= 2:
            vec = _plane(model, model.spins)
            ys, xs = np.mgrid[: vec.shape[0], : vec.shape[1]]
            # Column index is x (rightwards), row index is y (downwards in imshow).
            ax.quiver(xs, ys, vec[..., 0], -vec[..., 1], color="#2b2b2a",
                      pivot="middle", scale=1.25, scale_units="xy", width=0.004)
    else:
        raise TypeError(f"No configuration plot for {type(model).__name__}")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{model.name} {model.dim}D, L={model.L}")
    return ax
