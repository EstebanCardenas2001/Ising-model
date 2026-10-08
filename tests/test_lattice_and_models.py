import numpy as np
import pytest

from spinmodels import (
    HeisenbergModel,
    HypercubicLattice,
    IsingModel,
    PottsModel,
    XYModel,
)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_neighbor_table_is_symmetric(dim):
    lat = HypercubicLattice(4, dim)
    nbrs = lat.neighbors
    assert nbrs.shape == (4**dim, 2 * dim)
    for i in range(lat.N):
        for a in range(dim):
            # +a neighbor's -a neighbor is the site itself.
            assert nbrs[nbrs[i, 2 * a], 2 * a + 1] == i
        assert len(set(nbrs[i])) == 2 * dim  # all distinct for L >= 3


def test_neighbor_geometry_2d():
    lat = HypercubicLattice(5, 2)
    grid = np.arange(25).reshape(5, 5)
    i = grid[2, 4]
    assert lat.neighbors[i, 0] == grid[3, 4]  # +axis0
    assert lat.neighbors[i, 2] == grid[2, 0]  # +axis1 wraps around


def test_small_lattice_rejected():
    with pytest.raises(ValueError):
        HypercubicLattice(2, 2)


@pytest.mark.parametrize(
    "model",
    [IsingModel(6, dim=2, init="ordered"), PottsModel(6, q=4, init="ordered"),
     XYModel(6, init="ordered"), HeisenbergModel(4, init="ordered")],
    ids=lambda m: m.name,
)
def test_ordered_ground_state(model):
    e, m = model.observe()
    assert e == pytest.approx(-model.dim * model.J)
    assert m == pytest.approx(1.0)


def test_algorithm_registry_and_errors():
    m = IsingModel(4, seed=0)
    assert m.algorithms() == ["metropolis", "swendsen_wang", "wolff"]
    with pytest.raises(ValueError, match="does not support"):
        m.sweep(1.0, "heat_bath")
    with pytest.raises(ValueError):
        m.sweep(0.0)
    with pytest.raises(ValueError, match="h = 0"):
        IsingModel(4, h=0.1).sweep(1.0, "wolff")


def test_potts_q2_maps_to_ising():
    """H_Potts(J) = -J * sum (1 + s_i s_j)/2 = H_Ising(J/2) - J * N_bonds / 2."""
    ising = IsingModel(5, dim=2, J=0.5, seed=3)
    potts = PottsModel(5, dim=2, q=2, J=1.0)
    potts.set_state((ising.spins < 0).astype(np.int32))
    assert potts.energy() == pytest.approx(ising.energy() - ising.lattice.n_bonds / 2)
    assert potts.magnetization() == pytest.approx(ising.magnetization())


@pytest.mark.parametrize("cls", [XYModel, HeisenbergModel])
@pytest.mark.parametrize("algorithm", ["metropolis", "wolff"])
def test_vector_spins_stay_normalized(cls, algorithm):
    m = cls(6, seed=1)
    for _ in range(20):
        m.sweep(0.8, algorithm)
    np.testing.assert_allclose(np.linalg.norm(m.spins, axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("cls", [IsingModel, PottsModel, XYModel, HeisenbergModel])
def test_seed_reproducibility(cls):
    runs = []
    for _ in range(2):
        m = cls(6, seed=42)
        for _ in range(10):
            m.sweep(1.0)
        runs.append(m.spins.copy())
    np.testing.assert_array_equal(*runs)


def test_energy_kernel_matches_numpy():
    """Compiled energy (including the field term) vs a vectorised reference."""
    for model in [IsingModel(5, dim=3, h=0.3, seed=0), PottsModel(6, q=3, h=0.2, seed=0),
                  XYModel(6, h=0.4, seed=0), HeisenbergModel(4, h=0.4, seed=0)]:
        for _ in range(5):
            model.sweep(1.0)
        nb = model.neighbors[:, ::2]
        s = model.spins
        if isinstance(model, PottsModel):
            ref = -model.J * np.sum(s[:, None] == s[nb]) - model.h * np.sum(s == 0)
        elif s.ndim == 1:
            ref = -model.J * np.sum(s[:, None] * s[nb]) - model.h * s.sum()
        else:
            ref = -model.J * np.einsum("ic,ikc->", s, s[nb]) - model.h * s[:, 0].sum()
        assert model.energy() == pytest.approx(ref)
