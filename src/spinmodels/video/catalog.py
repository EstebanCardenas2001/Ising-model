"""The systems shown in the videos: model, dimension, temperature ranges,
system sizes and the physics captions used on screen."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import MODELS


@dataclass(frozen=True)
class System:
    key: str
    model: str  # registry name: ising / potts / xy / heisenberg
    dim: int
    title: str
    params: dict = field(default_factory=dict)
    #: Absolute temperature range covered by the equilibrium scans.
    T_range: tuple[float, float] = (0.2, 3.0)
    #: Range (in T) swept by this system's own video, hot -> cold.
    video_range: tuple[float, float] | None = None
    #: Sizes used for the finite-size-scaling scans, per preset.
    scan_L: dict = field(default_factory=dict)
    #: Lattice size shown in the videos.
    display_L: int = 480
    #: Concentration of scan temperatures around Tc (see ``temperature_grid``);
    #: first-order transitions need a much finer grid near Tc.
    scan_sharpness: float = 6.0
    #: Panels on the right-hand side of the single-system video.
    panels: tuple[str, ...] = ("magnetization", "susceptibility", "specific_heat", "binder", "energy", "histogram")

    @property
    def Tc(self) -> float | None:
        return MODELS[self.model].critical_temperature(self.dim, **self.params)

    @property
    def short(self) -> str:
        q = f" (q={self.params['q']})" if "q" in self.params else ""
        return f"{self.dim}D {MODELS[self.model].name}{q}"

    def phase(self, T: float) -> tuple[str, str]:
        """(headline, explanation) describing the state at temperature T."""
        return _phase_text(self, T)


def temperature_grid(T_lo: float, T_hi: float, n: int, Tc: float | None, sharpness: float = 6.0) -> np.ndarray:
    """``n`` temperatures in [T_lo, T_hi], denser near ``Tc`` (sinh mapping);
    geometric spacing if there is no transition."""
    if Tc is None or not T_lo < Tc < T_hi:
        return np.geomspace(T_lo, T_hi, n)
    a, b = (np.arcsinh(sharpness * (T - Tc) / Tc) for T in (T_lo, T_hi))
    return Tc * (1 + np.sinh(np.linspace(a, b, n)) / sharpness)


def cooling_path(T_hi: float, T_lo: float, n_frames: int, Tc: float | None,
                 hold: int = 45, sharpness: float = 10.0) -> np.ndarray:
    """Temperature per video frame, hot -> cold, slowing down near ``Tc``,
    with ``hold`` frames held at each end."""
    core = temperature_grid(T_lo, T_hi, n_frames - 2 * hold, Tc, sharpness)[::-1]
    return np.concatenate([np.full(hold, core[0]), core, np.full(hold, core[-1])])


# Scan sizes per preset. "quick" is for checking the pipeline end to end.
_2D = {"quick": [16, 32], "full": [32, 64, 128, 256]}
_3D = {"quick": [8, 12], "full": [8, 16, 24, 32]}

SYSTEMS: dict[str, System] = {
    s.key: s
    for s in [
        System("ising1d", "ising", 1, "1D Ising chain", T_range=(0.3, 6.0), video_range=(4.0, 0.35),
               scan_L={"quick": [64, 256], "full": [64, 256, 1024]}, display_L=960),
        System("ising2d", "ising", 2, "2D Ising model", T_range=(0.5, 6.0), video_range=(3.6, 1.2),
               scan_L=_2D, display_L=960),
        System("ising3d", "ising", 3, "3D Ising model", T_range=(0.5, 6.5), video_range=(6.5, 2.5),
               scan_L=_3D, display_L=240),
        System("potts3_2d", "potts", 2, "2D 3-state Potts model", params={"q": 3}, T_range=(0.2, 3.0),
               video_range=(1.6, 0.5), scan_L=_2D, display_L=960),
        System("potts8_2d", "potts", 2, "2D 8-state Potts model", params={"q": 8}, T_range=(0.2, 3.0),
               video_range=(1.1, 0.4), scan_L={"quick": [16, 32], "full": [16, 32, 64, 128]}, display_L=960,
               scan_sharpness=60.0),
        System("xy2d", "xy", 2, "2D XY model", T_range=(0.2, 3.0), video_range=(1.6, 0.3),
               scan_L=_2D, display_L=480,
               panels=("magnetization", "stiffness", "specific_heat", "binder", "histogram", "zoom")),
        System("xy3d", "xy", 3, "3D XY model", T_range=(0.2, 3.5), video_range=(3.5, 1.0),
               scan_L=_3D, display_L=240),
        System("heis2d", "heisenberg", 2, "2D Heisenberg model", T_range=(0.2, 3.0), video_range=(2.0, 0.25),
               scan_L=_2D, display_L=480),
        System("heis3d", "heisenberg", 3, "3D Heisenberg model", T_range=(0.2, 3.5), video_range=(2.4, 0.6),
               scan_L=_3D, display_L=240),
    ]
}


def _phase_text(s: System, T: float) -> tuple[str, str]:
    if s.key == "ising1d":
        xi = -1.0 / np.log(np.tanh(1.0 / T))
        return ("No phase transition in one dimension",
                f"A domain wall costs only 2J, so entropy always wins at T > 0. "
                f"Correlation length ξ = −1/ln[tanh(J/T)] ≈ {xi:,.1f} sites")
    if s.key == "heis2d":
        return ("No phase transition: Mermin–Wagner theorem",
                "A continuous symmetry cannot break spontaneously in 2D at T > 0: long-wavelength "
                "spin waves destroy order. ξ grows like exp(2πJ/T) but stays finite; a finite lattice "
                "only looks ordered once ξ exceeds L")
    t = T / s.Tc
    if s.key == "xy2d":
        if t > 1.06:
            return ("Disordered phase: free vortices",
                    "Unbound vortices (+) and antivortices (−) scramble the spin angles: correlations decay exponentially")
        if t > 0.97:
            return ("Berezinskii–Kosterlitz–Thouless transition",
                    "Vortex–antivortex pairs unbind; the spin stiffness jumps from 2T/π to zero (topological transition)")
        return ("Quasi-long-range order: bound vortex pairs",
                "Correlations decay as a power law. No true magnetization in an infinite system "
                "(Mermin–Wagner), yet |m| is large on any finite lattice")
    if s.key == "potts8_2d":
        if t > 1.01:
            return ("Disordered phase", "All eight states equally likely: short-range correlations only")
        if t > 0.99:
            return ("First-order transition: phase coexistence",
                    "Ordered and disordered regions coexist; the energy jumps by a latent heat "
                    "(double-peaked energy histogram), no diverging correlation length")
        return ("Ordered phase", "One of the eight states dominates: the S₈ permutation symmetry is broken")
    if t > 1.04:
        return ("Disordered (paramagnetic) phase",
                "Thermal fluctuations win: finite correlation length, no net order")
    if t > 0.97:
        return ("Critical point: fluctuations on all length scales",
                "The correlation length diverges: self-similar (fractal) clusters, susceptibility and "
                "specific heat peak, Binder cumulants of all sizes cross")
    names = {"ising": "up/down (Z₂)", "potts": "Z₃ permutation", "xy": "O(2) rotation", "heisenberg": "O(3) rotation"}
    return ("Ordered phase: spontaneous symmetry breaking",
            f"The {names[s.model]} symmetry is broken: a macroscopic fraction of spins align")
