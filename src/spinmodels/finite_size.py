"""How finite systems approach the thermodynamic limit near a critical point.

On a finite lattice nothing is singular: every "critical" feature is rounded
and shifted. Finite-size scaling predicts how the finite-``L`` values converge:

* **Pseudo-critical temperatures.** The peaks of ``chi``, ``C`` and
  ``d ln<|m|>/d beta`` sit at ``T_L = T_c + a L^(-1/nu)``. Binder-cumulant
  crossings of ``L`` and ``L'`` converge faster, ``~ L^(-1/nu - omega)``.
* **Effective exponents.** The local slope between two sizes,
  ``x_eff = ln(y_2/y_1) / ln(L_2/L_1)``, tends to the true exponent as
  ``x + b L^(-omega)`` (corrections to scaling).
* **Universal amplitudes.** The Binder cumulant at the crossing tends to a
  universal value ``U*`` that characterizes the universality class.

:func:`limit_table` and :func:`plot_limit` summarize one system;
:func:`plot_overview` compares several.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .critical import ExponentResult, binder_crossing

#: Binder cumulant U = 1 - <m^4>/(3<m^2>^2) at criticality (periodic boxes).
#: 2D Ising: Salas & Sokal 2000; 3D: Hasenbusch (2010, 2019, 2020).
BINDER_STAR: dict[str, float] = {
    "ising2d": 0.61069,
    "ising3d": 1 - 1.60357 / 3,
    "xy3d": 1 - 1.24292 / 3,
    "heis3d": 1 - 1.13933 / 3,
}


@dataclass
class PairEstimate:
    L1: int
    L2: int
    value: float
    error: float

    @property
    def L_eff(self) -> float:
        return float(np.sqrt(self.L1 * self.L2))


@dataclass
class LimitAnalysis:
    """Finite-size data of one system, ready for tables and plots."""

    sizes: np.ndarray
    N: np.ndarray
    pseudo_tc: dict[str, list[tuple[float, float]]]  # observable -> [(T_L, err)] per size
    crossings: list[PairEstimate]  # Binder crossing temperatures
    binder_at_crossing: list[PairEstimate]
    effective: dict[str, list[PairEstimate]]  # exponent -> pair slopes
    extrapolated: dict[str, tuple[float, float]]  # exponent -> L -> inf value
    Tc_extrapolated: dict[str, tuple[float, float]]
    nu: float
    omega: float | None
    Tc_ref: float | None
    reference: dict[str, float]
    U_ref: float | None


def _pair_slopes(L, y, err) -> list[PairEstimate]:
    out = []
    for i in range(len(L) - 1):
        r = np.log(L[i + 1] / L[i])
        x = np.log(y[i + 1] / y[i]) / r
        e = np.hypot(err[i] / y[i], err[i + 1] / y[i + 1]) / r
        out.append(PairEstimate(int(L[i]), int(L[i + 1]), float(x), float(e)))
    return out


def _linear_extrapolation(x, y, s) -> tuple[float, float]:
    """Weighted fit y = y0 + c x; returns (y0, error) (y0 = value at x = 0)."""
    x, y, s = (np.asarray(a, dtype=float) for a in (x, y, s))
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    x, y, s = x[ok], y[ok], s[ok]
    if len(x) < 2:
        return float("nan"), float("nan")
    A = np.vstack([np.ones_like(x), x]).T
    W = 1 / s**2
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    coef = cov @ (A.T @ (W * y))
    dof = len(x) - 2
    r = (y - A @ coef) / s
    scale = np.sqrt(max(r @ r / dof, 1.0)) if dof > 0 else 1.0
    return float(coef[0]), float(np.sqrt(cov[0, 0]) * scale)


def analyze_limit(res: ExponentResult, Tc_ref: float | None = None, U_ref: float | None = None) -> LimitAnalysis:
    """Collect pseudo-critical points, Binder crossings and effective exponents."""
    L = res.sizes.astype(float)
    nu = 1 / (res.exponents_corrected or res.exponents)["1/nu"][0]
    omega = res.omega

    pseudo = {k: [(p.T, p.T_err) for p in res.peaks[k]] for k in ("susceptibility", "dlnm", "specific_heat")}
    crossings, binder_x = [], []
    for (r1, r2) in zip(res.reweighters, res.reweighters[1:]):
        Tx, eTx = binder_crossing(r1, r2)
        crossings.append(PairEstimate(r1.data.L, r2.data.L, Tx, eTx))
        if np.isfinite(Tx):
            u, e = r1.at("binder", Tx)
        else:
            u, e = np.nan, np.nan
        binder_x.append(PairEstimate(r1.data.L, r2.data.L, u, e))

    def vals(key):
        return np.array([p.value for p in res.peaks[key]]), np.array([p.value_err for p in res.peaks[key]])

    effective = {
        "1/nu": _pair_slopes(L, *vals("dlnm")),
        "gamma/nu": _pair_slopes(L, *vals("susceptibility")),
        "beta/nu": _pair_slopes(L, np.array([v for v, _ in res.m_at_tc]), np.array([e for _, e in res.m_at_tc])),
    }
    for p in effective["beta/nu"]:
        p.value = -p.value  # |m| ~ L^(-beta/nu)

    w = omega if omega is not None else 1.0
    extrapolated = {}
    for k, pairs in effective.items():
        x = [p.L_eff ** (-w) for p in pairs]
        extrapolated[k] = _linear_extrapolation(x, [p.value for p in pairs], [p.error for p in pairs])

    Tc_ext = {}
    for k, pts in pseudo.items():
        Tc_ext[k] = _linear_extrapolation(L ** (-1 / nu), [t for t, _ in pts], [e for _, e in pts])
    xs = [p.L_eff ** (-1 / nu - w) for p in crossings]
    Tc_ext["binder crossings"] = _linear_extrapolation(xs, [p.value for p in crossings], [p.error for p in crossings])

    return LimitAnalysis(res.sizes, res.sizes**res.dim, pseudo, crossings, binder_x, effective, extrapolated,
                         Tc_ext, nu, omega, Tc_ref, res.reference or {}, U_ref)


def limit_table(a: LimitAnalysis) -> str:
    ref = a.reference
    Tc = a.Tc_ref
    out = ["Approach to the thermodynamic limit", ""]
    head = f"{'L':>5s} {'N = L^d':>10s} {'T_peak(chi)':>13s} {'T_peak(dlnm)':>13s} {'T_peak(C)':>11s}"
    if Tc:
        head += f" {'(T_chi-Tc)/Tc':>14s}"
    out.append(head)
    for i, L in enumerate(a.sizes):
        tchi = a.pseudo_tc["susceptibility"][i][0]
        line = (f"{int(L):5d} {int(a.N[i]):10,d} {tchi:13.5f} {a.pseudo_tc['dlnm'][i][0]:13.5f} "
                f"{a.pseudo_tc['specific_heat'][i][0]:11.5f}")
        if Tc:
            line += f" {(tchi - Tc) / Tc:+14.2e}"
        out.append(line)
    out += ["", f"{'pair':>9s} {'Binder crossing T*':>22s} {'U at crossing':>18s}"]
    for c, u in zip(a.crossings, a.binder_at_crossing):
        out.append(f"{c.L1:>4d}/{c.L2:<4d} {c.value:13.5f} ± {c.error:.5f} {u.value:10.4f} ± {u.error:.4f}")
    out.append("")
    out.append("T_c extrapolated to L -> inf:")
    for k, (t, e) in a.Tc_extrapolated.items():
        dev = f"   ({(t - Tc) / Tc:+.1e} from reference)" if Tc and np.isfinite(t) else ""
        out.append(f"  {k:18s} {t:.5f} ± {e:.5f}{dev}")
    if Tc:
        out.append(f"  {'reference':18s} {Tc:.5f}")
    if a.U_ref:
        out.append(f"Universal Binder cumulant U* = {a.U_ref:.4f} (reference)")
    out += ["", "Effective exponents from neighbouring sizes (local log-log slopes):"]
    refs = {"1/nu": 1 / ref["nu"] if "nu" in ref else None,
            "gamma/nu": ref["gamma"] / ref["nu"] if "nu" in ref else None,
            "beta/nu": ref["beta"] / ref["nu"] if "nu" in ref else None}
    out.append(f"{'pair':>9s} " + "".join(f"{k:>18s}" for k in a.effective))
    for i in range(len(a.sizes) - 1):
        p0 = a.effective["1/nu"][i]
        row = f"{p0.L1:>4d}/{p0.L2:<4d} "
        row += "".join(f"{a.effective[k][i].value:10.4f}±{a.effective[k][i].error:<7.4f}" for k in a.effective)
        out.append(row)
    w = f"L^-{a.omega:g}" if a.omega is not None else "1/L"
    row = f"{'L->inf':>9s} " + "".join(f"{v:10.4f}±{e:<7.4f}" for v, e in a.extrapolated.values())
    out.append(row + f"   (linear in {w})")
    out.append(f"{'reference':>9s} " + "".join(f"{refs[k]:10.4f}{'':8s}" if refs[k] else f"{'':18s}"
                                             for k in a.effective))
    return "\n".join(out)


def plot_limit(a: LimitAnalysis, title: str = ""):
    """Six panels: pseudo-critical temperatures vs L^(-1/nu), their distance
    to T_c (log-log), U at the Binder crossings, and the three effective
    exponents vs L^(-omega) with their extrapolation to L -> inf."""
    import matplotlib.pyplot as plt

    from .plotting import _STYLE

    blue, orange, aqua, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    L = a.sizes.astype(float)
    w = a.omega if a.omega is not None else 1.0
    Tc = a.Tc_ref
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

        ax = axes[0, 0]
        x = L ** (-1 / a.nu)
        xs = np.linspace(0, x.max() * 1.05, 50)
        for key, label, c in (("susceptibility", r"$\chi$ peak", blue), ("dlnm", r"$d\ln|m|/d\beta$ peak", orange),
                              ("specific_heat", "C peak", aqua)):
            T = np.array([t for t, _ in a.pseudo_tc[key]])
            e = np.array([s for _, s in a.pseudo_tc[key]])
            ax.errorbar(x, T, e, fmt="o", color=c, ms=6, capsize=2, label=label)
            t0, _ = a.Tc_extrapolated[key]
            slope = np.polyfit(x, T, 1, w=1 / np.maximum(e, 1e-9))[0]
            ax.plot(xs, t0 + slope * xs, color=c, lw=1, ls="--")
        xc = np.array([p.L_eff ** (-1 / a.nu) for p in a.crossings])
        ax.errorbar(xc, [p.value for p in a.crossings], [p.error for p in a.crossings], fmt="D", color=yellow,
                    ms=6, capsize=2, label="Binder crossings")
        if Tc:
            ax.axhline(Tc, color="#2b2b2a", lw=1.2, label=f"$T_c$ = {Tc:.5f}")
        ax.set_xlim(0, None)
        ax.set_xlabel(r"$L^{-1/\nu}$   ($L\to\infty$ at the left edge)")
        ax.set_ylabel("pseudo-critical temperature $T_c(L)$")
        ax.set_title("Pseudo-critical points converge to $T_c$", loc="left")
        ax.legend(fontsize=9)
        for xi, (t, _) in zip(x, a.pseudo_tc["dlnm"]):
            if xi < 0.25 * x.max():  # skip labels where the largest sizes crowd together
                continue
            ax.annotate(f"L={int(round(xi ** (-a.nu)))}", (xi, t), textcoords="offset points", xytext=(4, 6),
                        fontsize=8, color="#52514e")

        ax = axes[0, 1]
        if Tc:
            for key, label, c in (("susceptibility", r"$\chi$ peak", blue), ("dlnm", r"$d\ln|m|/d\beta$ peak", orange)):
                d = np.abs(np.array([t for t, _ in a.pseudo_tc[key]]) - Tc) / Tc
                ax.loglog(L, d, "o-", color=c, ms=6, lw=1.5, label=label)
            d = np.abs(np.array([p.value for p in a.crossings]) - Tc) / Tc
            ax.loglog([p.L_eff for p in a.crossings], np.maximum(d, 1e-7), "D-", color=yellow, ms=6, lw=1.5,
                      label="Binder crossings")
            ref = np.abs(a.pseudo_tc["susceptibility"][0][0] - Tc) / Tc
            ax.loglog(L, ref * (L / L[0]) ** (-1 / a.nu), color="#8a8984", ls=":", lw=1.5,
                      label=rf"slope $-1/\nu$ = {-1 / a.nu:.2f}")
            ax.set_xlabel("L")
            ax.set_ylabel(r"$|T_c(L) - T_c| / T_c$")
            ax.set_xticks(L, [f"{int(v)}" for v in L])
            ax.minorticks_off()
            ax.legend(fontsize=9)
        ax.set_title("Relative shift of the critical point", loc="left")

        ax = axes[0, 2]
        xb = np.array([p.L_eff ** (-w) for p in a.binder_at_crossing])
        ax.errorbar(xb, [p.value for p in a.binder_at_crossing], [p.error for p in a.binder_at_crossing],
                    fmt="D", color=yellow, ms=7, capsize=3, label="U at crossing of L, L'")
        if a.U_ref:
            ax.axhline(a.U_ref, color="#2b2b2a", lw=1.2, label=f"universal $U^*$ = {a.U_ref:.4f}")
        ax.set_xlim(0, None)
        ax.set_xlabel(rf"$L_{{\rm eff}}^{{-\omega}}$, $\omega$ = {w:g}")
        ax.set_ylabel("$U_4$ at the crossing")
        ax.set_title("Binder cumulant at the crossing → $U^*$", loc="left")
        ax.legend(fontsize=9)

        refs = a.reference
        ref_vals = {"1/nu": 1 / refs["nu"] if "nu" in refs else None,
                    "gamma/nu": refs["gamma"] / refs["nu"] if "nu" in refs else None,
                    "beta/nu": refs["beta"] / refs["nu"] if "nu" in refs else None}
        labels = {"1/nu": r"$1/\nu$", "gamma/nu": r"$\gamma/\nu$", "beta/nu": r"$\beta/\nu$"}
        for ax, (key, c) in zip(axes[1], (("1/nu", orange), ("gamma/nu", blue), ("beta/nu", aqua))):
            pairs = a.effective[key]
            x = np.array([p.L_eff ** (-w) for p in pairs])
            ax.errorbar(x, [p.value for p in pairs], [p.error for p in pairs], fmt="o", color=c, ms=7, capsize=3,
                        label="slope between neighbouring L")
            v, e = a.extrapolated[key]
            xs = np.linspace(0, x.max() * 1.05, 50)
            coef = np.polyfit(x, [p.value for p in pairs], 1, w=1 / np.maximum([p.error for p in pairs], 1e-9))
            ax.plot(xs, np.polyval(coef, xs), color=c, lw=1, ls="--")
            ax.errorbar([0], [v], [e], fmt="*", color=c, ms=14, capsize=3, label=f"$L\\to\\infty$: {v:.4f} ± {e:.4f}")
            if ref_vals[key] is not None:
                ax.axhline(ref_vals[key], color="#2b2b2a", lw=1.2, label=f"reference {ref_vals[key]:.4f}")
            for p, xi in zip(pairs, x):
                ax.annotate(f"{p.L1}/{p.L2}", (xi, p.value), textcoords="offset points", xytext=(6, 6), fontsize=8,
                            color="#52514e")
            ax.set_xlim(-0.02 * x.max(), None)
            ax.set_xlabel(rf"$L_{{\rm eff}}^{{-\omega}}$ with $L_{{\rm eff}} = \sqrt{{L L'}}$, $\omega$ = {w:g}")
            ax.set_title(f"Effective exponent {labels[key]}", loc="left")
            ax.legend(fontsize=9)
        fig.suptitle(title or "Approach to the thermodynamic limit", fontsize=14)
        fig.tight_layout()
    return fig


def plot_overview(analyses: dict[str, tuple[str, LimitAnalysis]]):
    """Compare systems: relative shift of the chi peak, and nu_eff / nu."""
    import matplotlib.pyplot as plt

    from .plotting import _STYLE

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    with plt.rc_context(_STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
        for (key, (label, a)), c in zip(analyses.items(), colors):
            L = a.sizes.astype(float)
            if a.Tc_ref:
                d = np.abs(np.array([t for t, _ in a.pseudo_tc["susceptibility"]]) - a.Tc_ref) / a.Tc_ref
                a1.loglog(L, d, "o-", color=c, ms=6, lw=1.8, label=f"{label} (1/ν = {1 / a.reference['nu']:.3f})")
            pairs = a.effective["1/nu"]
            nu_ref = a.reference.get("nu")
            if nu_ref:
                ratio = np.array([(1 / p.value) / nu_ref for p in pairs])
                err = np.array([p.error / p.value**2 / nu_ref for p in pairs])
                a2.errorbar([p.L_eff for p in pairs], ratio, err, fmt="o-", color=c, ms=6, lw=1.5, capsize=3,
                            label=label)
        a1.set_xlabel("L")
        a1.set_ylabel(r"$|T_\chi(L) - T_c|\,/\,T_c$")
        a1.set_title(r"Shift of the susceptibility peak $\propto L^{-1/\nu}$", loc="left")
        a1.legend(fontsize=9)
        a2.axhline(1, color="#2b2b2a", lw=1.2)
        a2.set_xscale("log")
        a2.set_xlabel(r"$L_{\rm eff} = \sqrt{L L'}$")
        a2.set_ylabel(r"$\nu_{\rm eff} / \nu$")
        a2.set_title(r"Effective $\nu$ from neighbouring sizes, relative to the exact value", loc="left")
        a2.legend(fontsize=9)
        fig.tight_layout()
    return fig
