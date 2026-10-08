"""Statistical tests of the Monte Carlo against exact results.

Each MC estimate must agree with the exact value within 5 jackknife standard
errors (plus a tiny absolute tolerance for quantities with near-zero error).
"""

import itertools

import numpy as np
import pytest

from spinmodels import (
    HeisenbergModel,
    IsingModel,
    PottsModel,
    XYModel,
    binder_cumulant,
    run,
    thermodynamics,
)


def assert_agrees(mc, exact, nsigma=5.0, atol=2e-3):
    value, err = mc
    assert abs(value - exact) < nsigma * err + atol, f"MC {value:.5f} ± {err:.5f} vs exact {exact:.5f}"


def exact_enumeration(model, T):
    """Exact observables by summing over all q^N states of a tiny lattice."""
    q = 2 if isinstance(model, IsingModel) else model.q
    values = np.array([-1, 1]) if q == 2 and isinstance(model, IsingModel) else np.arange(q)
    states = np.array(list(itertools.product(values, repeat=model.N)))
    nb = model.neighbors[:, ::2]
    if isinstance(model, IsingModel):
        E = -model.J * np.einsum("si,sik->s", states, states[:, nb])
        m = np.abs(states.mean(axis=1))
    else:
        E = -model.J * np.sum(states[:, :, None] == states[:, nb], axis=(1, 2))
        z = np.exp(2j * np.pi * states / q).mean(axis=1)
        m = np.abs(z)
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    e = E / model.N
    avg = lambda x: np.sum(w * x)  # noqa: E731
    N, beta = model.N, 1.0 / T
    m2, m4 = avg(m**2), avg(m**4)
    return {
        "energy": avg(e),
        "magnetization": avg(m),
        "specific_heat": N * beta**2 * (avg(e**2) - avg(e) ** 2),
        "susceptibility": N * beta * (m2 - avg(m) ** 2),
        "binder": 1 - m4 / (3 * m2**2),
    }


@pytest.mark.parametrize("algorithm", ["metropolis", "wolff"])
@pytest.mark.parametrize(
    "factory, T",
    [
        (lambda: IsingModel(3, dim=2, seed=11), 2.5),
        (lambda: IsingModel(3, dim=2, seed=12), 1.5),
        (lambda: PottsModel(3, dim=2, q=3, seed=13), 1.0),
    ],
    ids=["ising-T2.5", "ising-T1.5", "potts3-T1.0"],
)
def test_small_lattice_vs_enumeration(factory, T, algorithm):
    model = factory()
    exact = exact_enumeration(model, T)
    ts = run(model, T, n_equil=500, n_measure=60000, algorithm=algorithm)
    mc = thermodynamics(ts.energy, ts.magnetization, T, model.N, n_blocks=50)
    for key, value in exact.items():
        assert_agrees(mc[key], value)


def _bessel_i(n, x):
    t = np.linspace(0.0, np.pi, 20001)
    return np.trapezoid(np.exp(x * np.cos(t)) * np.cos(n * t), t) / np.pi


@pytest.mark.parametrize("algorithm", ["metropolis", "wolff"])
@pytest.mark.parametrize("T", [0.7, 2.0])
def test_1d_ising_chain(T, algorithm):
    L = 64
    model = IsingModel(L, dim=1, seed=5)
    t = np.tanh(1.0 / T)
    exact = -(t + t ** (L - 1)) / (1 + t**L)
    ts = run(model, T, n_equil=500, n_measure=20000, algorithm=algorithm)
    mc = thermodynamics(ts.energy, ts.magnetization, T, model.N, n_blocks=40)
    assert_agrees(mc["energy"], exact)


@pytest.mark.parametrize("algorithm", ["metropolis", "wolff"])
@pytest.mark.parametrize(
    "cls, exact_e",
    [
        # Thermodynamic-limit energies of the classical chains (K = J/T = 1);
        # periodic-boundary corrections are O(u^L) and negligible for L = 128.
        (XYModel, lambda K: -_bessel_i(1, K) / _bessel_i(0, K)),
        (HeisenbergModel, lambda K: -(1.0 / np.tanh(K) - 1.0 / K)),
    ],
    ids=["xy", "heisenberg"],
)
def test_1d_vector_chains(cls, exact_e, algorithm):
    T = 1.0
    model = cls(128, dim=1, seed=7)
    ts = run(model, T, n_equil=1000, n_measure=20000, algorithm=algorithm)
    mc = thermodynamics(ts.energy, ts.magnetization, T, model.N, n_blocks=40)
    assert_agrees(mc["energy"], exact_e(1.0 / T))


@pytest.mark.parametrize(
    "factory, T",
    [(lambda s: XYModel(8, dim=2, seed=s), 0.9),
     (lambda s: HeisenbergModel(4, dim=3, seed=s), 1.4)],
    ids=["xy2d", "heisenberg3d"],
)
def test_metropolis_and_wolff_agree(factory, T):
    """Two independent algorithms must sample the same distribution."""
    obs = {}
    for alg, seed in [("metropolis", 1), ("wolff", 2)]:
        model = factory(seed)
        ts = run(model, T, n_equil=1000, n_measure=20000, algorithm=alg)
        obs[alg] = thermodynamics(ts.energy, ts.magnetization, T, model.N, n_blocks=40)
    for key in ("energy", "magnetization", "binder"):
        (a, ea), (b, eb) = obs["metropolis"][key], obs["wolff"][key]
        assert abs(a - b) < 5 * np.hypot(ea, eb) + 2e-3, f"{key}: {a} ± {ea} vs {b} ± {eb}"


def test_binder_limits():
    assert binder_cumulant(np.ones(100)) == pytest.approx(2 / 3)
    rng = np.random.default_rng(0)
    # Disordered Ising-like: m ~ |Gaussian| gives U4 = 0.
    assert binder_cumulant(np.abs(rng.normal(size=400000))) == pytest.approx(0.0, abs=0.01)
