"""Thermodynamic observables and their statistical errors.

Given Monte Carlo time series of the energy per site ``e`` and the order
parameter magnitude per site ``m = |M|/N`` (``k_B = 1``):

* energy                  ``<e>``
* magnetization           ``<m>``
* specific heat           ``C   = N beta^2 (<e^2> - <e>^2)``
* susceptibility          ``chi = N beta (<m^2> - <m>^2)``
* Binder cumulant         ``U4  = 1 - <m^4> / (3 <m^2>^2)``

Using ``|M|`` in ``chi`` gives the finite-size susceptibility that peaks near
``T_c`` (on a finite lattice ``<M> = 0`` exactly by symmetry). ``U4`` tends to
2/3 in the ordered phase and to ``1 - (n+2)/(3n)`` in the disordered phase for
an ``n``-component order parameter (0 for Ising); curves for different ``L``
cross near ``T_c``.

Errors on these nonlinear estimators come from a blocked jackknife, which
also accounts for autocorrelations as long as the block length is well above
the integrated autocorrelation time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

OBSERVABLES = ("energy", "magnetization", "specific_heat", "susceptibility", "binder")

LABELS = {
    "energy": r"$\langle E\rangle / N$",
    "magnetization": r"$\langle |m| \rangle$",
    "specific_heat": r"$C_v / N$",
    "susceptibility": r"$\chi / N$",
    "binder": r"$U_4$",
    "stiffness": r"$\Upsilon$",
}


def binder_cumulant(m: np.ndarray) -> float:
    m2 = np.mean(m**2)
    if m2 == 0.0:
        return 0.0
    return 1.0 - np.mean(m**4) / (3.0 * m2**2)


def jackknife(
    estimator: Callable[..., float], *series: np.ndarray, n_blocks: int = 20
) -> tuple[float, float]:
    """Blocked jackknife estimate ``(value, error)`` of ``estimator(*series)``."""
    n = len(series[0])
    n_blocks = min(n_blocks, n)
    if n_blocks < 2:
        return float(estimator(*series)), float("nan")
    block = n // n_blocks
    used = block * n_blocks
    data = [np.asarray(x[:used]) for x in series]
    full = estimator(*data)
    keep = np.ones(used, dtype=bool)
    leave_one_out = np.empty(n_blocks)
    for b in range(n_blocks):
        keep[b * block : (b + 1) * block] = False
        leave_one_out[b] = estimator(*(x[keep] for x in data))
        keep[b * block : (b + 1) * block] = True
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((leave_one_out - leave_one_out.mean()) ** 2))
    return float(full), float(err)


def thermodynamics(
    e: np.ndarray, m: np.ndarray, T: float, N: int, n_blocks: int = 20
) -> dict[str, tuple[float, float]]:
    """Compute all observables with jackknife errors.

    Parameters
    ----------
    e, m : time series of energy per site and ``|m|`` per site.
    T : temperature.  N : number of sites.
    """
    beta = 1.0 / T
    return {
        "energy": jackknife(np.mean, e, n_blocks=n_blocks),
        "magnetization": jackknife(np.mean, m, n_blocks=n_blocks),
        "specific_heat": jackknife(lambda x: N * beta**2 * np.var(x), e, n_blocks=n_blocks),
        "susceptibility": jackknife(lambda x: N * beta * np.var(x), m, n_blocks=n_blocks),
        "binder": jackknife(binder_cumulant, m, n_blocks=n_blocks),
    }


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time ``tau_int = 1/2 + sum_t rho(t)``.

    Uses an FFT autocorrelation and Sokal's automatic window: the sum is cut
    at the smallest ``M`` with ``M >= c * tau_int(M)``. The effective number
    of independent samples is ``n / (2 tau_int)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if n < 2 or var == 0.0:
        return 0.5
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
    tau = np.cumsum(acf) - 0.5
    window = np.arange(n) >= c * tau
    m = int(np.argmax(window)) if window.any() else n - 1
    return float(tau[m])
