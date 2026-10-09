import numpy as np
import pytest

from spinmodels import IsingModel, run, thermodynamics
from spinmodels.critical import (
    KNOWN_EXPONENTS,
    CriticalData,
    Reweighter,
    analyze,
    collect_cpu,
    fit_log,
    fit_power_law,
)

TC_2D = 2 / np.log(1 + np.sqrt(2))


@pytest.fixture(scope="module")
def ising_data():
    model = IsingModel(16, dim=2, seed=1)
    ts = run(model, TC_2D, n_equil=500, n_measure=20000, algorithm="swendsen_wang")
    return ts, CriticalData.from_series(16, 2, TC_2D, ts.energy, ts.magnetization, n_blocks=40)


def test_reweighting_at_t0_reproduces_plain_averages(ising_data):
    ts, data = ising_data
    rw = Reweighter(data)
    direct = thermodynamics(ts.energy[: data.e.size], ts.magnetization[: data.e.size], TC_2D, data.N)
    values = rw.evaluate([TC_2D])
    for key in ("energy", "magnetization", "specific_heat", "susceptibility", "binder"):
        assert values[key][0, 0] == pytest.approx(direct[key][0], rel=1e-9)


@pytest.mark.parametrize("deriv, base, transform", [
    ("dbinder", "binder", lambda x: x),
    ("dlnm", "magnetization", np.log),
])
def test_fluctuation_derivatives_match_finite_differences(ising_data, deriv, base, transform):
    """d<O>/d beta = <O><E> - <OE> must agree with differentiating the reweighted curve."""
    rw = Reweighter(ising_data[1])
    T = TC_2D * 1.01
    h = 1e-4
    up, dn = (rw.evaluate([1 / (1 / T + s)], (base,))[base][0, 0] for s in (h, -h))
    numeric = (transform(up) - transform(dn)) / (2 * h)
    assert rw.evaluate([T], (deriv,))[deriv][0, 0] == pytest.approx(numeric, rel=1e-4)


def test_power_law_and_log_fits_recover_parameters():
    L = np.array([8, 16, 32, 64, 128])
    rng = np.random.default_rng(0)
    y = 2.5 * L**1.75 * (1 + 0.002 * rng.normal(size=L.size))
    f = fit_power_law(L, y, 0.002 * y)
    assert f.exponent == pytest.approx(1.75, abs=0.01)
    g = fit_log(L, 0.3 + 0.5 * np.log(L), np.full(L.size, 1e-3))
    assert g.exponent == pytest.approx(0.5, abs=1e-6)


def test_cpu_exponents_2d_ising():
    """End to end on small lattices: exponents within ~10% of the exact values."""
    data = []
    for L in (8, 12, 16, 24):
        model = IsingModel(L, dim=2, seed=L)
        data.append(collect_cpu(model, TC_2D, n_equil=300, n_measure=3000, n_chains=8))
    res = analyze(data, Tc=TC_2D, reference=KNOWN_EXPONENTS["ising2d"], verbose=False)
    ex = res.exponents
    assert ex["nu"][0] == pytest.approx(1.0, abs=0.1)
    assert ex["gamma/nu"][0] == pytest.approx(1.75, abs=0.1)
    assert ex["beta/nu"][0] == pytest.approx(0.125, abs=0.03)
    assert ex["check: 2 beta/nu + gamma/nu (= d)"][0] == pytest.approx(2.0, abs=0.1)
    assert "ln L" in res.specific_heat_form or "power law" in res.specific_heat_form


def test_gpu_exponents_2d_ising():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from spinmodels.gpu.critical import collect_gpu

    data = [collect_gpu("ising", L, 2, TC_2D, n_equil=200, n_measure=1000, n_chains=128, seed=L, verbose=False)
            for L in (16, 32, 64)]
    res = analyze(data, Tc=TC_2D, verbose=False)
    assert res.exponents["nu"][0] == pytest.approx(1.0, abs=0.06)
    assert res.exponents["gamma/nu"][0] == pytest.approx(1.75, abs=0.05)
    for _, _, t, _ in res.binder_crossings:
        assert t == pytest.approx(TC_2D, abs=0.01)


def test_corrected_fit_recovers_exponent_with_scaling_corrections():
    L = np.array([8, 12, 16, 24, 32, 48, 64], dtype=float)
    y = 3.0 * L**1.6 * (1 - 0.8 * L**-0.83)
    from spinmodels.critical import fit_power_law_corrected

    pure = fit_power_law(L, y, 1e-4 * y)
    fixed = fit_power_law_corrected(L, y, 1e-4 * y, omega=0.83)
    assert abs(pure.exponent - 1.6) > 0.05  # the pure power law is biased...
    assert fixed.exponent == pytest.approx(1.6, abs=2e-3)  # ...the corrected fit is not
