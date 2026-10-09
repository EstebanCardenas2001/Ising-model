import numpy as np
import pytest

from spinmodels.finite_size import _linear_extrapolation, _pair_slopes


def test_pair_slopes_of_a_pure_power_law_are_exact():
    L = np.array([8.0, 16.0, 32.0])
    pairs = _pair_slopes(L, 2.0 * L**1.75, np.full(3, 1e-6))
    assert [p.value for p in pairs] == pytest.approx([1.75, 1.75])
    assert pairs[0].L_eff == pytest.approx(np.sqrt(8 * 16))


def test_effective_exponents_extrapolate_through_corrections():
    """x_eff(L) = x + b L^-omega is linear in L^-omega: the intercept is x."""
    L = np.array([8.0, 12.0, 16.0, 24.0, 32.0, 48.0])
    omega = 0.83
    y = L**1.6 * np.exp(0.5 * L**-omega)  # local slope = 1.6 - 0.5 omega L^-omega exactly
    pairs = _pair_slopes(L, y, 1e-6 * y)
    x = [p.L_eff ** -omega for p in pairs]
    x_inf, _ = _linear_extrapolation(x, [p.value for p in pairs], [p.error for p in pairs])
    assert abs(pairs[0].value - 1.6) > 0.05
    assert x_inf == pytest.approx(1.6, abs=5e-3)


def test_pseudo_critical_extrapolation():
    L = np.array([16.0, 32.0, 64.0, 128.0])
    T = 2.2692 + 0.9 * L**-1.0
    t0, _ = _linear_extrapolation(L**-1.0, T, np.full(4, 1e-5))
    assert t0 == pytest.approx(2.2692, abs=1e-9)
