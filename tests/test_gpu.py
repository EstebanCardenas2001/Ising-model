"""GPU engine checks against exact results (skipped without a CUDA device)."""

import itertools

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

from spinmodels.gpu import GPUHeisenberg, GPUIsing, GPUPotts, GPUXY  # noqa: E402
from spinmodels.gpu.scan import gpu_temperature_scan  # noqa: E402


def replica_average(model, algorithm, n_equil=300, n_measure=400):
    """Mean E/N over replicas and time; error from the spread of replica means."""
    model.sweep(algorithm, n_equil)
    es = []
    for _ in range(n_measure):
        model.sweep(algorithm)
        es.append(model.observe()[0])
    per_replica = np.mean(es, axis=0)
    return per_replica.mean(), per_replica.std() / np.sqrt(len(per_replica))


def check(value, err, exact):
    assert abs(value - exact) < 5 * err + 2e-4, f"{value:.5f} ± {err:.5f} vs exact {exact:.5f}"


def _bessel_i(n, x):
    t = np.linspace(0.0, np.pi, 20001)
    return np.trapezoid(np.exp(x * np.cos(t)) * np.cos(n * t), t) / np.pi


@pytest.mark.parametrize("algorithm", ["local", "swendsen_wang"])
@pytest.mark.parametrize("T", [1.0, 2.0])
def test_ising_chain(T, algorithm):
    L = 64
    t = np.tanh(1 / T)
    exact = -(t + t ** (L - 1)) / (1 + t**L)
    check(*replica_average(GPUIsing(L, 1, np.full(1024, T), seed=1), algorithm), exact)


@pytest.mark.parametrize("algorithm", ["local", "swendsen_wang"])
@pytest.mark.parametrize("q", [3, 5])
def test_potts_chain(q, algorithm):
    # Z = l1^L + (q-1) l2^L with l1 = e^K + q - 1, l2 = e^K - 1 (K = J/T).
    L, T = 64, 0.8
    K = 1 / T
    l1, l2 = np.exp(K) + q - 1, np.exp(K) - 1
    lnZ = lambda K: np.log((np.exp(K) + q - 1) ** L + (q - 1) * (np.exp(K) - 1) ** L)  # noqa: E731
    exact = -(lnZ(K + 1e-6) - lnZ(K - 1e-6)) / 2e-6 / L
    check(*replica_average(GPUPotts(L, 1, np.full(1024, T), q=q, seed=2), algorithm), exact)


@pytest.mark.parametrize("algorithm", ["local", "swendsen_wang"])
@pytest.mark.parametrize(
    "cls, exact",
    [(GPUXY, -_bessel_i(1, 1.0) / _bessel_i(0, 1.0)), (GPUHeisenberg, -(1 / np.tanh(1.0) - 1.0))],
    ids=["xy", "heisenberg"],
)
def test_vector_chains(cls, exact, algorithm):
    model = cls(128, 1, np.full(512, 1.0), seed=3)
    if algorithm == "local":
        model.sweep("metropolis_adapt", 200)
    check(*replica_average(model, algorithm), exact)


@pytest.mark.parametrize("algorithm", ["local", "swendsen_wang"])
def test_ising_4x4_enumeration(algorithm):
    T = 2.4
    states = np.array(list(itertools.product([-1, 1], repeat=16))).reshape(-1, 4, 4)
    E = -(states * np.roll(states, -1, 1)).sum((1, 2)) - (states * np.roll(states, -1, 2)).sum((1, 2))
    w = np.exp(-(E - E.min()) / T)
    exact = (w * E).sum() / w.sum() / 16
    check(*replica_average(GPUIsing(4, 2, np.full(4096, T), seed=4), algorithm), exact)


def test_scan_with_parallel_tempering_2d_ising():
    """Binder cumulant at the exact T_c is close to the universal U* ≈ 0.611."""
    Tc = 2 / np.log(1 + np.sqrt(2))
    res = gpu_temperature_scan("ising", 32, 2, [0.95 * Tc, Tc, 1.05 * Tc], n_equil=300,
                               n_measure=2000, n_chains=8, seed=5, verbose=False)
    assert np.all(res.extras["swap_acceptance"] > 0.1)
    assert res.mean["binder"][1] == pytest.approx(0.611, abs=0.02)
    assert res.mean["magnetization"][0] > res.mean["magnetization"][2]


def test_xy_stiffness_low_temperature():
    """Helicity modulus: spin-wave result Upsilon ≈ J - T/4 at low T, and
    ~0 well above T_BKT."""
    res = gpu_temperature_scan("xy", 16, 2, [0.2, 1.5], n_equil=300, n_measure=1000,
                               n_chains=8, seed=6, verbose=False)
    assert res.mean["stiffness"][0] == pytest.approx(1 - 0.2 / 4, abs=0.03)
    assert abs(res.mean["stiffness"][1]) < 0.1
