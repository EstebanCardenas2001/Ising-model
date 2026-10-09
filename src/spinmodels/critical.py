"""Critical exponents from finite-size scaling (FSS).

Workflow
--------
1. For each lattice size ``L``, sample the energy and order parameter at one
   temperature ``T0`` close to ``T_c`` (:func:`collect_cpu`, or
   :func:`spinmodels.gpu.critical.collect_gpu` which runs hundreds of
   independent chains in one batch).
2. :class:`Reweighter` turns those samples into *continuous* functions of
   temperature by single-histogram reweighting (Ferrenberg & Swendsen 1988):
   ``<O>_beta = sum O exp(-(beta-beta0) E) / sum exp(-(beta-beta0) E)``.
   Peaks are located precisely, with blocked-jackknife errors.
3. :func:`analyze` fits the FSS laws at criticality

   ======================================  ==================
   ``max d ln<|m|>/d beta  ~ L^(1/nu)``     correlation length
   ``max chi               ~ L^(gamma/nu)`` susceptibility
   ``<|m|>(T_c)            ~ L^(-beta/nu)`` order parameter
   ``max C                 ~ L^(alpha/nu)`` specific heat
   ``T_peak(L) - T_c       ~ L^(-1/nu)``    shift of pseudo-critical points
   ======================================  ==================

   and derives ``nu, gamma, beta, eta = 2 - gamma/nu`` and, through
   hyperscaling, ``alpha = 2 - d nu`` and ``delta``. The relation
   ``2 beta/nu + gamma/nu = d`` is reported as a consistency check.

Corrections to scaling make small lattices deviate from pure power laws; use
``L_min`` to drop them and watch whether the estimates drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Reference values (k_B = J = 1). Sources: 2D Ising and 2D Potts are exact;
# 3D values from conformal bootstrap / high-precision Monte Carlo
# (Kos et al. 2016; Hasenbusch 2010, 2019; Campostrini et al. 2002, 2006).
KNOWN_EXPONENTS: dict[str, dict[str, float]] = {
    "ising2d": dict(nu=1.0, gamma=1.75, beta=0.125, alpha=0.0, eta=0.25, delta=15.0),
    "ising3d": dict(nu=0.629971, gamma=1.237075, beta=0.326419, alpha=0.110087, eta=0.036298, delta=4.78984),
    "potts3_2d": dict(nu=5 / 6, gamma=13 / 9, beta=1 / 9, alpha=1 / 3, eta=4 / 15, delta=14.0),
    "xy3d": dict(nu=0.6717, gamma=1.3178, beta=0.3486, alpha=-0.0151, eta=0.0381, delta=4.780),
    "heis3d": dict(nu=0.7112, gamma=1.3960, beta=0.3689, alpha=-0.1336, eta=0.0375, delta=4.783),
}

#: Leading correction-to-scaling exponents omega (same sources; 2D Potts: 4/5).
#: 2D Ising has no irrelevant correction below omega = 7/4 (analytic background).
CORRECTION_OMEGA: dict[str, float] = {
    "ising2d": 1.75, "ising3d": 0.832, "potts3_2d": 0.8, "xy3d": 0.789, "heis3d": 0.759,
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class CriticalData:
    """Samples of one lattice size at temperature ``T0``.

    ``e`` and ``m`` have shape ``(n_samples, n_blocks)``: each column is one
    independent chain, or one contiguous block of a single long chain. The
    columns are the jackknife blocks.
    """

    L: int
    dim: int
    T0: float
    e: np.ndarray  # energy per site
    m: np.ndarray  # |order parameter| per site
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.L**self.dim

    @classmethod
    def from_series(cls, L, dim, T0, e, m, n_blocks: int = 32, **meta) -> "CriticalData":
        """Build from one long time series by cutting it into contiguous blocks."""
        e, m = np.asarray(e), np.asarray(m)
        n = len(e) // n_blocks
        return cls(L, dim, T0, e[: n * n_blocks].reshape(n_blocks, n).T,
                   m[: n * n_blocks].reshape(n_blocks, n).T, meta)

    def save(self, path) -> None:
        np.savez(path, L=self.L, dim=self.dim, T0=self.T0, e=self.e, m=self.m, meta=np.array([repr(self.meta)]))

    @classmethod
    def load(cls, path) -> "CriticalData":
        import ast

        d = np.load(path)
        return cls(int(d["L"]), int(d["dim"]), float(d["T0"]), d["e"], d["m"], ast.literal_eval(str(d["meta"][0])))


# --------------------------------------------------------------------------- #
# Reweighting
# --------------------------------------------------------------------------- #
#: Observables available from :meth:`Reweighter.evaluate`.
RW_OBSERVABLES = ("energy", "magnetization", "specific_heat", "susceptibility", "binder",
                  "dlnm", "dlnm2", "dbinder")


class Reweighter:
    """Single-histogram reweighting of one :class:`CriticalData` set.

    All observables are computed from per-block weighted sums, so the
    leave-one-block-out jackknife costs nothing extra: every method returns
    an array whose row 0 is the full estimate and rows ``1..B`` the jackknife
    samples (see :func:`jackknife_error`).
    """

    def __init__(self, data: CriticalData):
        self.data = data
        self.N = data.N
        self.beta0 = 1.0 / data.T0
        E = data.e.astype(np.float64) * data.N
        # Shift energies: variances/covariances are unchanged, cancellation is reduced.
        E = E - E.mean()
        self.B = data.e.shape[1]
        # Reliable reweighting range: |beta - beta0| <~ 2 / sigma_E.
        self.dbeta_max = 2.0 / max(E.std(), 1e-12)
        # The weighted sums dominate the cost; run them on the GPU when available.
        self.xp = _array_module(E.size)
        self.E = self.xp.asarray(E)
        self.m = self.xp.asarray(data.m.astype(np.float64))
        self._grid_cache: dict = {}

    def window(self, width: float = 1.0) -> tuple[float, float]:
        """Temperature range in which reweighting is trustworthy."""
        b_lo = self.beta0 - width * self.dbeta_max
        b_hi = self.beta0 + width * self.dbeta_max
        return 1.0 / b_hi, 1.0 / max(b_lo, 1e-12)

    def _sums(self, T: np.ndarray) -> dict[str, np.ndarray]:
        """Weighted sums, shape (n_T, B) per quantity (summed over samples)."""
        T = np.atleast_1d(T)
        db = 1.0 / T - self.beta0
        E, m = self.E, self.m
        m2, m4 = m * m, m**4
        out = {k: np.empty((len(T), self.B)) for k in ("w", "E", "E2", "m", "m2", "m4", "mE", "m2E", "m4E")}
        xp = self.xp
        to_host = (lambda a: a.get()) if xp is not np else (lambda a: a)
        for i, d in enumerate(db):
            x = -d * E
            w = xp.exp(x - x.max())
            wE, wm = w * E, w * m
            out["w"][i] = to_host(w.sum(0))
            out["E"][i] = to_host(wE.sum(0))
            out["E2"][i] = to_host((wE * E).sum(0))
            out["m"][i] = to_host(wm.sum(0))
            out["m2"][i] = to_host((w * m2).sum(0))
            out["m4"][i] = to_host((w * m4).sum(0))
            out["mE"][i] = to_host((wm * E).sum(0))
            out["m2E"][i] = to_host((w * m2 * E).sum(0))
            out["m4E"][i] = to_host((w * m4 * E).sum(0))
        return out

    def evaluate(self, T, names=RW_OBSERVABLES) -> dict[str, np.ndarray]:
        """Observables at temperatures ``T``: arrays of shape ``(1 + B, n_T)``
        (row 0: all data; rows 1..B: leave-one-block-out)."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        sums = self._sums(T)
        # Totals and leave-one-out sums: (n_T, 1 + B) -> transpose to (1 + B, n_T).
        S = {}
        for k, v in sums.items():
            tot = v.sum(1, keepdims=True)
            S[k] = np.concatenate([tot, tot - v], axis=1).T
        w = S["w"]
        beta = 1.0 / T
        N = self.N
        E1, E2 = S["E"] / w, S["E2"] / w
        m1, m2, m4 = S["m"] / w, S["m2"] / w, S["m4"] / w
        mE, m2E, m4E = S["mE"] / w, S["m2E"] / w, S["m4E"] / w
        res = {}
        # Energy shift is undone for <E>/N only; fluctuations are shift-invariant.
        shift = (self.data.e.astype(np.float64) * N).mean()
        if "energy" in names:
            res["energy"] = (E1 + shift) / N
        if "specific_heat" in names:
            res["specific_heat"] = beta**2 * (E2 - E1**2) / N
        if "magnetization" in names:
            res["magnetization"] = m1
        if "susceptibility" in names:
            res["susceptibility"] = N * beta * (m2 - m1**2)
        if "binder" in names or "dbinder" in names:
            U = 1.0 - m4 / (3.0 * m2**2)
            res["binder"] = U
        if "dlnm" in names:  # d ln<|m|>/d beta = <E> - <|m| E>/<|m|>
            res["dlnm"] = E1 - mE / m1
        if "dlnm2" in names:  # d ln<m^2>/d beta
            res["dlnm2"] = E1 - m2E / m2
        if "dbinder" in names:  # dU/d beta from fluctuations
            dA = m4 * E1 - m4E
            dB = m2 * E1 - m2E
            res["dbinder"] = -(dA / m2**2 - 2.0 * m4 * dB / m2**3) / 3.0
        return res

    def peak(self, name: str, n_grid: int = 161, width: float = 1.0, sign: float = 1.0) -> "Peak":
        """Location and height of the maximum of ``sign * observable`` inside
        the reweighting window, with jackknife errors (parabolic refinement)."""
        key = (n_grid, width)
        if key not in self._grid_cache:  # one pass over the samples serves every observable
            T_grid = np.linspace(*self.window(width), n_grid)
            self._grid_cache[key] = (T_grid, self.evaluate(T_grid))
        T, values = self._grid_cache[key]
        y = sign * values[name]  # (1 + B, n_grid)
        k = np.argmax(y, axis=1)
        interior = (k > 0) & (k < n_grid - 1)
        k = np.clip(k, 1, n_grid - 2)
        rows = np.arange(y.shape[0])
        y0, y1, y2 = y[rows, k - 1], y[rows, k], y[rows, k + 1]
        denom = y0 - 2 * y1 + y2
        shift = np.where(denom != 0, 0.5 * (y0 - y2) / np.where(denom != 0, denom, 1), 0.0)
        dT = T[1] - T[0]
        T_pk = T[k] + shift * dT
        y_pk = y1 - 0.25 * (y0 - y2) * shift
        return Peak(name, *_jk(T_pk), *_jk(sign * y_pk), bool(interior[0]))

    def at(self, name: str, T: float) -> tuple[float, float]:
        """(value, error) of an observable at temperature ``T``."""
        lo, hi = self.window(1.5)
        if not lo <= T <= hi:
            return float("nan"), float("nan")
        return _jk(self.evaluate([T], (name,))[name][:, 0])


def _array_module(n_values: int):
    """CuPy for large sample sets when a CUDA device is usable, else NumPy."""
    if n_values < 200_000:
        return np
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return np


@dataclass
class Peak:
    name: str
    T: float
    T_err: float
    value: float
    value_err: float
    interior: bool


def _jk(rows: np.ndarray) -> tuple[float, float]:
    """(full estimate, jackknife error) from rows [full, leave-one-out...]."""
    full, loo = rows[0], rows[1:]
    B = len(loo)
    err = np.sqrt((B - 1) / B * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(err)


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    exponent: float
    error: float
    amplitude: float
    chi2_dof: float
    L: np.ndarray

    def __str__(self) -> str:
        return f"{self.exponent:.4f} ± {self.error:.4f} (χ²/dof = {self.chi2_dof:.2f}, L = {list(map(int, self.L))})"


def fit_power_law(L, y, err) -> Fit:
    """Weighted least squares fit of ``y = A L^x`` in log-log space.

    The error is inflated by ``sqrt(chi2/dof)`` when the fit is poor, so it
    also reflects systematic deviations (e.g. corrections to scaling).
    """
    L, y, err = (np.asarray(a, dtype=float) for a in (L, y, err))
    ok = np.isfinite(y) & (y > 0) & np.isfinite(err)
    L, y, err = L[ok], y[ok], err[ok]
    if len(L) < 2:
        return Fit(np.nan, np.nan, np.nan, np.nan, L)
    X = np.log(L)
    Y = np.log(y)
    s = np.maximum(err / y, 1e-12)
    W = 1 / s**2
    A = np.vstack([np.ones_like(X), X]).T
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    coef = cov @ (A.T @ (W * Y))
    resid = (Y - A @ coef) / s
    dof = len(L) - 2
    chi2 = float(resid @ resid / dof) if dof > 0 else 0.0
    scale = np.sqrt(max(chi2, 1.0)) if dof > 0 else 1.0
    return Fit(float(coef[1]), float(np.sqrt(cov[1, 1]) * scale), float(np.exp(coef[0])), chi2, L)


def fit_power_law_corrected(L, y, err, omega: float, window: float = 0.5, max_correction: float = 1.0,
                            n_grid: int = 4001) -> Fit:
    """Fit ``y = A L^x (1 + B L^-omega)`` with the correction exponent
    ``omega`` held fixed.

    For each trial ``x`` the model is linear in ``(A, A B)``; the best ``x``
    minimises chi2 and its error is the half-width of ``chi2 <= chi2_min + 1``
    (inflated by ``chi2/dof`` when the fit is poor). Needs >= 4 sizes.

    Two constraints keep the "correction" a correction: ``x`` stays within
    ``window`` of the pure power-law exponent, and the correction is smaller
    than the leading term at the smallest size, ``|B| L_min^-omega <= max_correction``.
    Without them the fit can swap the roles of the two terms.
    """
    L, y, err = (np.asarray(a, dtype=float) for a in (L, y, err))
    ok = np.isfinite(y) & np.isfinite(err) & (err > 0) & (y > 0)
    L, y, s = L[ok], y[ok], err[ok]
    if len(L) < 4:
        return Fit(np.nan, np.nan, np.nan, np.nan, L)
    x0 = fit_power_law(L, y, s).exponent
    xs = np.linspace(x0 - window, x0 + window, n_grid)
    chi2 = np.full(n_grid, np.inf)
    amps = np.full(n_grid, np.nan)
    for i, x in enumerate(xs):
        A = np.vstack([L**x, L ** (x - omega)]).T / s[:, None]
        coef, *_ = np.linalg.lstsq(A, y / s, rcond=None)
        if coef[0] <= 0 or abs(coef[1] / coef[0]) * L.min() ** -omega > max_correction:
            continue
        r = A @ coef - y / s
        chi2[i] = r @ r
        amps[i] = coef[0]
    if not np.isfinite(chi2).any():
        return Fit(np.nan, np.nan, np.nan, np.nan, L)
    k = int(np.argmin(chi2))
    dof = len(L) - 3
    scale = max(chi2[k] / dof, 1.0) if dof > 0 else 1.0
    inside = xs[chi2 <= chi2[k] + scale]
    err_x = 0.5 * (inside.max() - inside.min()) if len(inside) > 1 else np.nan
    return Fit(float(xs[k]), float(err_x), float(amps[k]), float(chi2[k] / dof) if dof > 0 else 0.0, L)


def fit_log(L, y, err) -> Fit:
    """Weighted fit of ``y = a + b ln L`` (a logarithmic divergence, alpha = 0).
    ``exponent`` holds the slope ``b``."""
    L, y, err = (np.asarray(a, dtype=float) for a in (L, y, err))
    ok = np.isfinite(y) & np.isfinite(err)
    L, y, s = L[ok], y[ok], np.maximum(err[ok], 1e-12)
    if len(L) < 2:
        return Fit(np.nan, np.nan, np.nan, np.nan, L)
    A = np.vstack([np.ones_like(L), np.log(L)]).T
    W = 1 / s**2
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    coef = cov @ (A.T @ (W * y))
    resid = (y - A @ coef) / s
    dof = len(L) - 2
    chi2 = float(resid @ resid / dof) if dof > 0 else 0.0
    scale = np.sqrt(max(chi2, 1.0)) if dof > 0 else 1.0
    return Fit(float(coef[1]), float(np.sqrt(cov[1, 1]) * scale), float(coef[0]), chi2, L)


def extrapolate_tc(L, T_peak, T_err, nu: float) -> tuple[float, float]:
    """Fit ``T_peak(L) = T_c + a L^(-1/nu)``; return ``(T_c, error)``."""
    L, T, s = (np.asarray(a, dtype=float) for a in (L, T_peak, T_err))
    ok = np.isfinite(T) & np.isfinite(s)
    L, T, s = L[ok], T[ok], np.maximum(s[ok], 1e-12)
    if len(L) < 2 or not np.isfinite(nu):
        return float("nan"), float("nan")
    x = L ** (-1.0 / nu)
    A = np.vstack([np.ones_like(x), x]).T
    W = 1 / s**2
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    coef = cov @ (A.T @ (W * T))
    dof = len(L) - 2
    resid = (T - A @ coef) / s
    scale = np.sqrt(max(resid @ resid / dof, 1.0)) if dof > 0 else 1.0
    return float(coef[0]), float(np.sqrt(cov[0, 0]) * scale)


def binder_crossing(r1: Reweighter, r2: Reweighter, n_grid: int = 400) -> tuple[float, float]:
    """Temperature where the Binder cumulants of two sizes cross (overlap of
    both reweighting windows), with an error from the jackknife errors of U."""
    lo = max(r1.window(1.5)[0], r2.window(1.5)[0])
    hi = min(r1.window(1.5)[1], r2.window(1.5)[1])
    if lo >= hi:
        return float("nan"), float("nan")
    T = np.linspace(lo, hi, n_grid)
    u1 = r1.evaluate(T, ("binder",))["binder"]
    u2 = r2.evaluate(T, ("binder",))["binder"]
    d = u1[0] - u2[0]
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    i = idx[np.argmin(np.abs(T[idx] - 0.5 * (r1.data.T0 + r2.data.T0)))]
    Tx = T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i])
    slope = (d[i + 1] - d[i]) / (T[i + 1] - T[i])

    def err(u):
        B = u.shape[0] - 1
        return np.sqrt((B - 1) / B * np.sum((u[1:, i] - u[1:, i].mean()) ** 2))

    return float(Tx), float(np.hypot(err(u1), err(u2)) / abs(slope))


# --------------------------------------------------------------------------- #
# Full analysis
# --------------------------------------------------------------------------- #
@dataclass
class ExponentResult:
    dim: int
    sizes: np.ndarray
    Tc: float  # temperature at which <|m|> was evaluated
    Tc_source: str
    fits: dict[str, Fit]
    Tc_extrapolated: dict[str, tuple[float, float]]
    binder_crossings: list[tuple[int, int, float, float]]
    peaks: dict[str, list[Peak]]
    m_at_tc: list[tuple[float, float]]
    reference: dict[str, float] | None = None
    reweighters: list = field(default_factory=list, repr=False)
    corrected_fits: dict[str, Fit] = field(default_factory=dict)
    omega: float | None = None

    # ---- derived exponents (propagated errors) -------------------------
    @property
    def exponents(self) -> dict[str, tuple[float, float]]:
        """Exponents from pure power-law fits."""
        return self._exponents(self.fits)

    @property
    def exponents_corrected(self) -> dict[str, tuple[float, float]]:
        """Exponents from fits with a correction to scaling (if ``omega`` was given)."""
        return self._exponents(self.corrected_fits) if self.corrected_fits else {}

    def _exponents(self, fits: dict[str, Fit]) -> dict[str, tuple[float, float]]:
        d = self.dim
        inv_nu, s_inv = fits["1/nu"].exponent, fits["1/nu"].error
        nu = 1 / inv_nu
        s_nu = s_inv / inv_nu**2
        g, s_g = fits["gamma/nu"].exponent, fits["gamma/nu"].error
        b, s_b = -fits["beta/nu"].exponent, fits["beta/nu"].error
        out = {
            "1/nu": (inv_nu, s_inv),
            "nu": (nu, s_nu),
            "gamma/nu": (g, s_g),
            "beta/nu": (b, s_b),
            "gamma": (g * nu, np.hypot(s_g * nu, g * s_nu)),
            "beta": (b * nu, np.hypot(s_b * nu, b * s_nu)),
            "eta": (2 - g, s_g),
            "alpha (hyperscaling 2-d nu)": (2 - d * nu, d * s_nu),
            "delta": ((d + g) / (d - g), 2 * d * s_g / (d - g) ** 2) if d != g else (np.nan, np.nan),
            "check: 2 beta/nu + gamma/nu (= d)": (2 * b + g, np.hypot(2 * s_b, s_g)),
        }
        if "alpha/nu" in fits:
            a = fits["alpha/nu"]
            out["alpha/nu (from C_max)"] = (a.exponent, a.error)
        return out

    @property
    def specific_heat_form(self) -> str:
        """Which form describes C_max(L) better: a power law or a logarithm."""
        p, g = self.fits.get("alpha/nu"), self.fits.get("C_max = a + b ln L")
        if p is None or g is None or not np.isfinite(p.chi2_dof):
            return "undetermined"
        if g.chi2_dof < p.chi2_dof:
            return (f"C_max = a + b ln L fits better (chi2/dof = {g.chi2_dof:.2f}) than a power law "
                    f"({p.chi2_dof:.2f}): alpha = 0, or a small alpha hidden by the regular background "
                    f"at these sizes; prefer the hyperscaling alpha")
        return (f"power law: alpha/nu = {p.exponent:.3f} ± {p.error:.3f} (chi2/dof = {p.chi2_dof:.2f}; "
                f"logarithm: {g.chi2_dof:.2f})")

    def table(self) -> str:
        ref = self.reference or {}
        lines = [f"Finite-size scaling, d = {self.dim}, L = {list(map(int, self.sizes))}",
                 f"<|m|> evaluated at T = {self.Tc:.5f} ({self.Tc_source})", ""]
        corr = self.exponents_corrected
        head = f"{'quantity':38s} {'power law':>22s}"
        if corr:
            head += f"   {'with L^-' + format(self.omega, '.3g') + ' correction':>24s}"
        lines.append(head + f"   {'reference':>10s}")
        refs = {"nu": ref.get("nu"), "gamma": ref.get("gamma"), "beta": ref.get("beta"),
                "eta": ref.get("eta"), "alpha (hyperscaling 2-d nu)": ref.get("alpha"),
                "delta": ref.get("delta"),
                "1/nu": 1 / ref["nu"] if "nu" in ref else None,
                "gamma/nu": ref["gamma"] / ref["nu"] if "nu" in ref else None,
                "beta/nu": ref["beta"] / ref["nu"] if "nu" in ref else None,
                "check: 2 beta/nu + gamma/nu (= d)": float(self.dim),
                "alpha/nu (from C_max)": ref["alpha"] / ref["nu"] if "nu" in ref else None}
        for k, (v, e) in self.exponents.items():
            r = refs.get(k)
            line = f"{k:38s} {v:>12.4f} ± {e:<8.4f}"
            if corr:
                cv, ce = corr.get(k, (np.nan, np.nan))
                line += f"   {cv:>14.4f} ± {ce:<7.4f}" if np.isfinite(cv) else f"   {'':>24s}"
            lines.append(line + f"   {'' if r is None else f'{r:10.4f}'}")
        lines.append(f"specific heat: {self.specific_heat_form}")
        lines.append("")
        for k, (t, e) in self.Tc_extrapolated.items():
            lines.append(f"T_c from {k} peaks (L -> inf): {t:.5f} ± {e:.5f}")
        for L1, L2, t, e in self.binder_crossings:
            lines.append(f"Binder crossing L = {L1}/{L2}: T = {t:.5f} ± {e:.5f}")
        lines.append("")
        for k, f in self.fits.items():
            lines.append(f"fit {k:10s}: {f}")
        for k, f in self.corrected_fits.items():
            lines.append(f"fit {k:10s} (omega = {self.omega}): {f}")
        return "\n".join(lines)


def analyze(datasets: list[CriticalData], Tc: float | None = None, L_min: int = 0,
            reference: dict | None = None, omega: float | None = None,
            verbose: bool = True) -> ExponentResult:
    """Extract critical exponents from samples at several lattice sizes.

    Parameters
    ----------
    datasets : one :class:`CriticalData` per size (same model, dimension).
    Tc : temperature at which ``<|m|> ~ L^(-beta/nu)`` is evaluated. If None,
        the extrapolated ``T_c`` from the ``d ln<|m|>/d beta`` peaks is used
        (it must lie inside every size's reweighting window).
    L_min : ignore sizes below this in all fits.
    omega : if given, also fit ``1/nu``, ``gamma/nu`` and ``beta/nu`` with a
        leading correction to scaling ``(1 + B L^-omega)`` (see
        :func:`fit_power_law_corrected`; ``CORRECTION_OMEGA`` lists values).
    """
    t0 = time.perf_counter()
    datasets = sorted([d for d in datasets if d.L >= L_min], key=lambda d: d.L)
    dim = datasets[0].dim
    rws = [Reweighter(d) for d in datasets]
    L = np.array([d.L for d in datasets])

    peaks = {name: [rw.peak(name) for rw in rws] for name in ("dlnm", "dlnm2", "dbinder", "susceptibility",
                                                              "specific_heat")}

    def vals(name):
        return np.array([p.value for p in peaks[name]]), np.array([p.value_err for p in peaks[name]])

    fits = {"1/nu": fit_power_law(L, *vals("dlnm")),
            "1/nu [d ln m^2]": fit_power_law(L, *vals("dlnm2")),
            "1/nu [dU/dbeta]": fit_power_law(L, *np.abs(vals("dbinder"))),
            "gamma/nu": fit_power_law(L, *vals("susceptibility")),
            "alpha/nu": fit_power_law(L, *vals("specific_heat")),
            "C_max = a + b ln L": fit_log(L, *vals("specific_heat"))}
    nu = 1 / fits["1/nu"].exponent
    Tc_ext = {name: extrapolate_tc(L, [p.T for p in peaks[name]], [p.T_err for p in peaks[name]], nu)
              for name in ("dlnm", "susceptibility", "specific_heat")}

    if Tc is None:
        Tc_eval, source = Tc_ext["dlnm"][0], "extrapolated from d ln<|m|>/d beta peaks"
    else:
        Tc_eval, source = float(Tc), "given"
    m_tc = [rw.at("magnetization", Tc_eval) for rw in rws]
    fits["beta/nu"] = fit_power_law(L, [v for v, _ in m_tc], [e for _, e in m_tc])
    corrected = {}
    if omega is not None:
        corrected["1/nu"] = fit_power_law_corrected(L, *vals("dlnm"), omega)
        corrected["gamma/nu"] = fit_power_law_corrected(L, *vals("susceptibility"), omega)
        corrected["beta/nu"] = fit_power_law_corrected(L, [v for v, _ in m_tc], [e for _, e in m_tc], omega)

    crossings = []
    for (r1, d1), (r2, d2) in zip(zip(rws, datasets), zip(rws[1:], datasets[1:])):
        crossings.append((d1.L, d2.L, *binder_crossing(r1, r2)))

    res = ExponentResult(dim, L, Tc_eval, source, fits, Tc_ext, crossings, peaks, m_tc, reference, rws,
                         corrected, omega)
    if verbose:
        print(f"[analysis {time.perf_counter() - t0:.1f}s]")
    return res


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_fss(res: ExponentResult, title: str = ""):
    """Four panels: power-law fits, Binder crossings and two data collapses
    (``|m| L^(beta/nu)`` and ``chi L^(-gamma/nu)`` vs ``(T - T_c) L^(1/nu)``)."""
    import matplotlib.pyplot as plt

    from .plotting import _STYLE, SERIES_COLORS

    ex = res.exponents_corrected or res.exponents
    inv_nu, g_nu, b_nu = ex["1/nu"][0], ex["gamma/nu"][0], ex["beta/nu"][0]
    L = res.sizes.astype(float)
    colors = (SERIES_COLORS * 2)[: len(L)]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        ax = axes[0, 0]
        series = [
            ("susceptibility", r"$\chi_{\max}$", "gamma/nu", +1),
            ("dlnm", r"$\max\, d\ln\langle|m|\rangle/d\beta$", "1/nu", +1),
        ]
        for (key, label, fit_key, _), c in zip(series, ("#2a78d6", "#eb6834")):
            y = np.array([p.value for p in res.peaks[key]])
            e = np.array([p.value_err for p in res.peaks[key]])
            f = res.fits[fit_key]
            ax.errorbar(L, y, e, fmt="o", color=c, ms=7, capsize=3, label=f"{label}: slope {f.exponent:.3f}")
            ax.plot(L, f.amplitude * L**f.exponent, color=c, lw=1.5)
        y = np.array([v for v, _ in res.m_at_tc])
        e = np.array([v for _, v in res.m_at_tc])
        f = res.fits["beta/nu"]
        ax.errorbar(L, y, e, fmt="s", color="#1baf7a", ms=7, capsize=3,
                    label=rf"$\langle|m|\rangle(T_c)$: slope {f.exponent:.3f}")
        ax.plot(L, f.amplitude * L**f.exponent, color="#1baf7a", lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(L, [f"{int(x)}" for x in L])
        ax.minorticks_off()
        ax.set_xlabel("L")
        ax.set_title("Finite-size scaling at criticality (log-log)", loc="left")
        ax.legend(fontsize=9)

        ax = axes[0, 1]
        for rw, c, Lv in zip(res.reweighters, colors, L):
            T = np.linspace(*rw.window(1.2), 200)
            U = rw.evaluate(T, ("binder",))["binder"][0]
            ax.plot(T, U, color=c, lw=2, label=f"L = {int(Lv)}")
        for _, _, t, _ in res.binder_crossings:
            if np.isfinite(t):
                ax.axvline(t, color="#8a8984", lw=0.8, ls=":")
        ax.axvline(res.Tc, color="#2b2b2a", lw=1.2, ls="--", label=f"T = {res.Tc:.4f}")
        ax.set_xlabel("T")
        ax.set_ylabel("$U_4$")
        ax.set_title("Binder cumulant (reweighted): curves cross at $T_c$", loc="left")
        ax.legend(fontsize=9)
        lo = max(rw.window(1.2)[0] for rw in res.reweighters[-2:])
        hi = min(rw.window(1.2)[1] for rw in res.reweighters[-2:])
        span = max(hi - lo, 1e-3)
        ax.set_xlim(lo - 2 * span, hi + 2 * span)

        for ax, key, power, ylabel in (
            (axes[1, 0], "magnetization", b_nu, r"$\langle|m|\rangle\, L^{\beta/\nu}$"),
            (axes[1, 1], "susceptibility", -g_nu, r"$\chi\, L^{-\gamma/\nu}$"),
        ):
            for rw, c, Lv in zip(res.reweighters, colors, L):
                T = np.linspace(*rw.window(1.2), 200)
                y = rw.evaluate(T, (key,))[key][0]
                ax.plot((T - res.Tc) * Lv**inv_nu, y * Lv**power, color=c, lw=2, label=f"L = {int(Lv)}")
            ax.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
            ax.set_ylabel(ylabel)
            ax.set_title("Data collapse with the measured exponents", loc="left")
            ax.legend(fontsize=9)
        fig.suptitle(title or f"Critical exponents, d = {res.dim}", fontsize=14)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Data collection (CPU)
# --------------------------------------------------------------------------- #
def collect_cpu(model, T0: float, n_equil: int = 2000, n_measure: int = 20000,
                algorithm: str = "swendsen_wang", n_chains: int = 1, n_workers: int = 1) -> CriticalData:
    """Sample ``model`` at ``T0`` on the CPU. With ``n_chains > 1`` the
    chains are independent copies (optionally in parallel processes) and
    serve as jackknife blocks; a single chain is cut into 32 blocks."""
    from .simulation import _run_independent, run

    kwargs = dict(n_equil=n_equil, n_measure=n_measure, algorithm=algorithm)
    if n_chains == 1:
        ts = run(model, T0, **kwargs)
        return CriticalData.from_series(model.L, model.dim, T0, ts.energy, ts.magnetization,
                                        model=model.name, **model.params())
    seeds = model.rng.integers(0, 2**32 - 1, size=n_chains)
    tasks = [(model, T0, int(s), kwargs) for s in seeds]
    if n_workers != 1:
        import os
        from concurrent.futures import ProcessPoolExecutor

        workers = os.cpu_count() if n_workers == -1 else n_workers
        with ProcessPoolExecutor(max_workers=min(workers, n_chains)) as pool:
            series = list(pool.map(_run_independent, tasks))
    else:
        series = [_run_independent(t) for t in tasks]
    e = np.stack([s.energy for s in series], axis=1)
    m = np.stack([s.magnetization for s in series], axis=1)
    return CriticalData(model.L, model.dim, T0, e, m, dict(model=model.name, **model.params()))


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
#: Default sizes per system (keys as in ``spinmodels.video.catalog.SYSTEMS``).
DEFAULT_SIZES = {
    "ising2d": [16, 32, 64, 128, 256],
    "ising3d": [8, 12, 16, 24, 32, 48],
    "potts3_2d": [16, 32, 64, 128, 256],
    "xy3d": [8, 12, 16, 24, 32, 48],
    "heis3d": [8, 12, 16, 24, 32, 48],
}


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from .video.catalog import SYSTEMS

    ap = argparse.ArgumentParser(prog="python -m spinmodels.critical",
                                 description="Critical exponents by finite-size scaling.")
    ap.add_argument("system", choices=sorted(DEFAULT_SIZES))
    ap.add_argument("-L", type=int, nargs="+", help="lattice sizes (even for the GPU)")
    ap.add_argument("--T0", type=float, help="simulation temperature (default: known T_c)")
    ap.add_argument("--equil", type=int, default=2000)
    ap.add_argument("--measure", type=int, default=10000, help="samples per chain")
    ap.add_argument("--chains", type=int, default=None, help="chains per size (default: fill the GPU)")
    ap.add_argument("--L-min", type=int, default=0, help="drop smaller sizes from the fits")
    ap.add_argument("--omega", default="auto",
                    help="correction-to-scaling exponent for the corrected fits: a number, "
                         "'auto' (literature value) or 'none'")
    ap.add_argument("--cpu", action="store_true", help="use the CPU (Numba) engine")
    ap.add_argument("--workers", type=int, default=-1, help="CPU processes (with --cpu)")
    ap.add_argument("--out", type=Path, default=Path("output/critical"))
    ap.add_argument("--reuse", action="store_true", help="reuse saved samples when present")
    args = ap.parse_args(argv)

    s = SYSTEMS[args.system]
    sizes = args.L or DEFAULT_SIZES[args.system]
    T0 = args.T0 or s.Tc
    args.out.mkdir(parents=True, exist_ok=True)
    data = []
    for L in sizes:
        path = args.out / f"{args.system}_L{L}.npz"
        if args.reuse and path.exists():
            data.append(CriticalData.load(path))
            continue
        if args.cpu:
            from .models import MODELS

            model = MODELS[s.model](L, dim=s.dim, seed=L, **s.params)
            d = collect_cpu(model, T0, args.equil, args.measure, n_chains=args.chains or 16, n_workers=args.workers)
        else:
            from .gpu.critical import collect_gpu

            d = collect_gpu(s.model, L, s.dim, T0, args.equil, args.measure, n_chains=args.chains,
                            seed=L, **s.params)
        d.save(path)
        data.append(d)

    omega = (CORRECTION_OMEGA.get(args.system) if args.omega == "auto"
             else None if args.omega == "none" else float(args.omega))
    res = analyze(data, Tc=T0, L_min=args.L_min, reference=KNOWN_EXPONENTS.get(args.system), omega=omega)
    text = res.table()
    print(text)
    (args.out / f"{args.system}_exponents.txt").write_text(text + "\n")
    import matplotlib

    matplotlib.use("Agg")
    fig = plot_fss(res, title=f"{s.title}: critical exponents from finite-size scaling")
    fig.savefig(args.out / f"{args.system}_fss.png", dpi=130)
    print(f"Saved {args.out / (args.system + '_fss.png')}")

    from .finite_size import BINDER_STAR, analyze_limit, limit_table, plot_limit

    lim = analyze_limit(res, Tc_ref=s.Tc, U_ref=BINDER_STAR.get(args.system))
    text = limit_table(lim)
    print("\n" + text)
    (args.out / f"{args.system}_limit.txt").write_text(text + "\n")
    plot_limit(lim, f"{s.title}: approach to the thermodynamic limit").savefig(
        args.out / f"{args.system}_limit.png", dpi=110)


if __name__ == "__main__":
    main()
