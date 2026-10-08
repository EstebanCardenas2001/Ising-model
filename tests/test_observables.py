import numpy as np
import pytest

from spinmodels import integrated_autocorr_time, jackknife


def test_jackknife_of_mean_matches_standard_error():
    x = np.random.default_rng(0).normal(size=100000)
    value, err = jackknife(np.mean, x, n_blocks=100)
    assert value == pytest.approx(x.mean())
    assert err == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.2)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_autocorrelation_time_of_ar1(rho):
    rng = np.random.default_rng(1)
    n = 200000
    noise = rng.normal(size=n)
    x = np.empty(n)
    x[0] = noise[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    exact = 0.5 * (1 + rho) / (1 - rho)
    assert integrated_autocorr_time(x) == pytest.approx(exact, rel=0.1)
