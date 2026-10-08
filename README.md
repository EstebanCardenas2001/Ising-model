# spinmodels

Numba-accelerated Monte Carlo simulations of classical lattice spin models:

| Model | Spins | Default lattice | Known transition |
|---|---|---|---|
| `IsingModel` | $s_i = \pm 1$ | 2D (any `dim`) | 2D: $T_c = 2/\ln(1+\sqrt2) \approx 2.269$ (Onsager); 3D: $T_c \approx 4.5115$ |
| `PottsModel` | $s_i \in \{0,\dots,q-1\}$ | 2D | 2D: $T_c = 1/\ln(1+\sqrt q)$ (continuous for $q\le4$, first order for $q>4$) |
| `XYModel` | unit vectors in $\mathbb R^2$ | 2D | BKT transition, $T_{BKT} \approx 0.893$ |
| `HeisenbergModel` | unit vectors in $\mathbb R^3$ | 3D | $T_c \approx 1.443$ |

All models live on periodic hypercubic lattices of any dimension, with coupling $J$ and field $h$
($k_B = 1$):

$$
H_\text{Ising} = -J\sum_{\langle ij\rangle} s_i s_j - h\sum_i s_i,\qquad
H_\text{Potts} = -J\sum_{\langle ij\rangle} \delta_{s_i s_j} - h\sum_i \delta_{s_i,0},\qquad
H_{O(n)} = -J\sum_{\langle ij\rangle} \mathbf S_i\cdot\mathbf S_j - h\sum_i S_i^x .
$$

## Install

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"
pytest                                     # ~15 s, validates against exact results
```

## Quick start

```python
import numpy as np
from spinmodels import IsingModel, temperature_scan
from spinmodels.plotting import plot_scan

results = [
    temperature_scan(IsingModel(L, dim=2, seed=L), np.linspace(1.8, 2.8, 21),
                     n_equil=500, n_measure=5000, algorithm="wolff")
    for L in (8, 16, 32)
]
fig, _ = plot_scan(results, Tc=IsingModel.critical_temperature(2))
```

Or from the command line:

```bash
python -m spinmodels ising --dim 3 -L 6 8 12 --tmin 4.2 --tmax 4.8 --nt 13 --algorithm wolff --out output/ising3d
python -m spinmodels potts -q 3 -L 16 32 --tmin 0.9 --tmax 1.1 --algorithm wolff --out output/potts3
python -m spinmodels heisenberg -L 8 12 --tmin 1.3 --tmax 1.6 --algorithm wolff --out output/heis
```

Low-level use (one Markov chain):

```python
from spinmodels import XYModel, run, thermodynamics

model = XYModel(32, seed=0)
ts = run(model, T=0.9, n_equil=1000, n_measure=10000, algorithm="metropolis")
obs = thermodynamics(ts.energy, ts.magnetization, T=0.9, N=model.N)  # {name: (value, error)}
tau_e, tau_m = ts.tau()                                               # autocorrelation times
```

## Examples

| Script | Output |
|---|---|
| `examples/ising2d_finite_size.py` | All observables for L = 8, 16, 32. The Binder cumulants cross at the exact $T_c$. |
| `examples/snapshots.py` | Configurations of all four models at 0.6, 1.0, 1.6 $T_c$. |
| `examples/critical_slowing_down.py` | $\tau_\text{int}(L)$ at $T_c$: Metropolis grows with $z\approx2$, while Wolff stays nearly flat. |

Figures go to `output/` (git-ignored).

## Observables

From time series of $e = E/N$ and $m = |\mathbf M|/N$:

| Observable | Estimator |
|---|---|
| Energy | $\langle e\rangle$ |
| Magnetization | $\langle m\rangle$ |
| Specific heat | $C_v/N = N\beta^2(\langle e^2\rangle - \langle e\rangle^2)$ |
| Susceptibility | $\chi/N = N\beta(\langle m^2\rangle - \langle m\rangle^2)$ |
| Binder cumulant | $U_4 = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2$ |

Errors come from a blocked jackknife. Integrated autocorrelation times use Sokal's
automatic windowing. Use `measure_every` or more blocks if $\tau_\text{int}$ is not much smaller
than the block length. For an $n$-component order parameter, $U_4 \to 2/3$ when ordered and
$\to 1-(n+2)/3n$ when disordered (0 for Ising, 1/3 for XY and the Potts vector order parameter,
4/9 for Heisenberg).

## Architecture

```
src/spinmodels/
├── lattice.py        HypercubicLattice: flat (N, 2·dim) neighbor table → kernels are dimension-agnostic
├── models/
│   ├── base.py       LatticeModel ABC: state, sweep() dispatch, adaptation hooks
│   ├── ising.py      Numba kernels + IsingModel
│   ├── potts.py      Numba kernels + PottsModel
│   └── vector.py     shared O(n) kernels; VectorModel → XYModel (n=2), HeisenbergModel (n=3)
├── observables.py    estimators, jackknife, autocorrelation time
├── simulation.py     run(), temperature_scan(), ScanResult (save/load .npz)
├── plotting.py       plot_scan(), plot_configuration()
└── __main__.py       CLI
```

**Update algorithms are plug-ins.** `model.sweep(T, algorithm="name")` dispatches to the
method `_name_sweep(beta)`, and `model.algorithms()` discovers these methods automatically. A new
algorithm (Swendsen–Wang, heat bath, over-relaxation, …) is therefore one new method on a model.
The driver code doesn't change. Currently implemented:

- **`metropolis`**: $N$ random single-spin attempts per sweep. Vector models propose
  $\mathbf S' = \mathrm{normalize}(\mathbf S + \delta\,\mathbf g)$, and the step $\delta$ is tuned
  toward 50% acceptance during equilibration.
- **`wolff`**: single-cluster updates for all four models (spin flip for Ising, relabelling
  $s_0\to s'$ for Potts, reflection across a random hyperplane for O(n)). Requires $J>0$, $h=0$.
  Each sweep grows a *fixed* number of clusters, tuned during equilibration to flip $\approx N$
  spins. Measuring after "as many clusters as needed to flip N spins" would size-bias the last
  cluster and skew the averages; the exact-enumeration tests catch this.

**Adding a model:** subclass `LatticeModel` and implement `_initial_spins`, `energy`,
`magnetization_vector` and `_metropolis_sweep`. Everything else (driver, observables,
error analysis, CLI via the `MODELS` registry) works unchanged.

## Validation (`tests/`)

- 3×3 Ising (two temperatures) and q = 3 Potts: all five observables vs **exact enumeration**
  over every state ($2^9$ and $3^9$), for both algorithms.
- 1D chains vs **closed-form solutions**: Ising $e = -(t + t^{L-1})/(1+t^L)$, XY
  $e = -I_1(K)/I_0(K)$, Heisenberg $e = -(\coth K - 1/K)$, for both algorithms.
- Metropolis vs Wolff agreement for 2D XY and 3D Heisenberg, plus the Potts(q=2) ↔ Ising mapping,
  the estimators on synthetic AR(1) data, lattice geometry, spin normalization and seeding.

## Notes

- Numba draws from one global RNG stream. Every model re-seeds it on construction, so
  `seed=` makes a single run reproducible. For parallel runs, use separate processes.
- `temperature_scan(anneal=True)` (the default) visits temperatures from hot to cold and reuses
  the configuration between points.
- Kernels are compiled with `cache=True`, so only the very first run pays the compilation time.
