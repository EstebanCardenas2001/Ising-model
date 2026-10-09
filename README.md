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
uv venv && uv pip install -e ".[dev]"        # CPU only
uv pip install -e ".[dev,gpu]"               # + CUDA engine and videos (CuPy, NVIDIA GPU)
pytest                                        # ~30 s, validates against exact results (GPU tests skip without CUDA)
```

Videos additionally need `ffmpeg` (NVENC hardware encoding is used when available).

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

Parallel CPU scan (independent chains, one process per temperature) and CPU parallel tempering:

```python
from spinmodels import IsingModel, PottsModel, parallel_tempering, temperature_scan

res = temperature_scan(IsingModel(32, seed=0), np.linspace(2.0, 2.6, 16),
                       algorithm="swendsen_wang", n_workers=-1)      # all cores
pt = parallel_tempering(PottsModel(24, q=8, seed=0), np.linspace(0.70, 0.80, 12))
pt.extras["swap_acceptance"]
```

GPU scan: every temperature × several independent chains are replicas of one batched CUDA simulation,
updated by Swendsen–Wang + local sweeps and coupled by parallel tempering:

```python
from spinmodels.gpu.scan import gpu_temperature_scan

res = gpu_temperature_scan("ising", L=256, dim=2, temperatures=np.linspace(2.1, 2.45, 64),
                           n_equil=1000, n_measure=10000, n_chains=2)
res.mean["binder"], res.error["binder"], res.extras["swap_acceptance"]
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
├── models/           CPU models (Numba): base.py (ABC, sweep dispatch), ising.py, potts.py,
│                     vector.py (XY, Heisenberg), _cluster.py (union-find for Swendsen–Wang)
├── observables.py    estimators, jackknife, autocorrelation time
├── simulation.py     run(), temperature_scan() (serial / multi-process), parallel_tempering(), ScanResult
├── plotting.py       plot_scan(), plot_configuration()
├── gpu/              CUDA engine: kernels.cu, engine.py (GPUIsing/GPUPotts/GPUXY/GPUHeisenberg),
│                     scan.py (batched scans + parallel tempering)
├── video/            catalog.py (systems + physics captions), produce.py (GPU frames),
│                     render.py (Matplotlib + ffmpeg/NVENC), pipeline.py (orchestration)
└── __main__.py       CLI
```

**Update algorithms are plug-ins.** `model.sweep(T, algorithm="name")` dispatches to the
method `_name_sweep(beta)`, and `model.algorithms()` discovers these methods automatically. A new
algorithm is therefore one new method on a model; the driver code doesn't change.

| Algorithm | CPU (Numba) | GPU (CUDA) | Notes |
|---|---|---|---|
| Metropolis | `metropolis`: random-site | `metropolis`: checkerboard, O(n) only | vector proposal $\mathrm{normalize}(\mathbf S+\delta\mathbf g)$, $\delta$ tuned to 50% acceptance during equilibration |
| Heat bath | | `heatbath` (Ising, Potts) | exact conditional sampling; `"local"` picks heat bath or Metropolis per model |
| Wolff | `wolff` | | single cluster; spin flip / Potts relabel / O(n) reflection |
| Swendsen–Wang | `swendsen_wang` | `swendsen_wang` | all clusters at once; GPU labelling by parallel union-find (atomicMin hooking) |
| Parallel tempering | `parallel_tempering()` | `gpu_temperature_scan()` | replica exchange between neighbouring temperatures |

Cluster updates require $J>0$ and $h=0$. Two subtleties the exact tests caught:

- **Wolff "sweeps"** grow a *fixed* number of clusters, tuned during equilibration to flip
  $\approx N$ spins. Measuring after "as many clusters as needed to flip N spins" size-biases
  the last cluster and skews the averages.
- **Checkerboard Metropolis is not ergodic for discrete spins.** Moves with $\Delta E=0$ are
  accepted deterministically, so in 1D every domain wall moves ballistically and the difference
  between left- and right-movers is conserved (a 27σ energy bias at T = 1). The GPU therefore uses
  heat-bath updates for Ising and Potts. The CPU picks sites at random, which is ergodic.

**GPU engine** (`spinmodels.gpu`): `R` replicas of an `L^dim` lattice (one temperature each) live in one
device array and are updated in a single kernel launch. Neighbours are computed arithmetically, so
the same kernels serve 1D–4D. Random numbers come from a counter-based Philox4x32-10 generator
(no stored RNG state; reproducible), and energies, magnetizations and the XY helicity modulus come
from one fused reduction kernel. On a Tesla T4: 6–14 G site-updates/s for local updates and
1–2.4 G/s for Swendsen–Wang.

**Adding a model:** subclass `LatticeModel` and implement `_initial_spins`, `energy`,
`magnetization_vector` and `_metropolis_sweep`. Everything else (driver, observables,
error analysis, CLI via the `MODELS` registry) works unchanged.

## Videos

```bash
python -m spinmodels.video --preset full          # GPU scans (~1 h on a T4) + all 19 videos
python -m spinmodels.video --preset quick         # small/fast pipeline check
python -m spinmodels.video --preset full --skip-scans --only ising2d mosaic_xy2d
python -m spinmodels.video --list
```

Output goes to `output/<preset>/`: `videos/*.mp4` (1920×1080, 30 fps, H.264),
`figures/scan_*.png` (static finite-size-scaling plots) and `data/scan_*.npz` (cached scans).

| Video | What it shows |
|---|---|
| `ising1d`, `ising2d`, `ising3d`, `potts3_2d`, `potts8_2d`, `xy2d`, `xy3d`, `heis2d`, `heis3d` | One system cooled slowly through its transition (large lattice; 3D shown as a slice; 1D as a space-time diagram). Beside it are equilibrium curves for several L (order parameter, χ, C, U₄, energy, or the helicity modulus for XY) with a moving temperature cursor, the live system's value, and the energy histogram at the current T (single-histogram Ferrenberg–Swendsen reweighting from the nearest scan temperature, so it changes smoothly; at the first-order 8-state Potts transition it shows the two coexisting peaks exchanging weight). A caption names the phase. The XY video zooms in on spins and vortices. |
| `compare_ising_dimensions` | 1D / 2D / 3D Ising at the same T: T_c grows with dimension; 1D never orders |
| `compare_2d_symmetries` | Ising, 3-Potts, 8-Potts (first order), XY (BKT), Heisenberg (Mermin–Wagner) in 2D |
| `compare_continuous_symmetry` | XY and Heisenberg in 2D vs 3D |
| `mosaic_*` | 16 fixed temperatures around T_c under local dynamics: domain coarsening, critical slowing down, coexistence at the first-order transition |

How it works: the GPU runs each scene's simulations (Swendsen–Wang + local updates, so every frame is
an equilibrium sample; mosaics use local dynamics only) and streams frame chunks to RAM disk. A pool
of CPU processes renders them with Matplotlib, and segments are encoded with NVENC and concatenated.
Images are gauge-fixed (shown relative to the current order-parameter direction), so global flips
and rotations from cluster updates do not recolour the picture.

## Validation (`tests/`)

- 3×3 Ising (two temperatures) and q = 3 Potts: all five observables vs **exact enumeration**
  over every state ($2^9$ and $3^9$), for Metropolis, Wolff, Swendsen–Wang and parallel tempering.
- 1D chains vs **closed-form solutions**: Ising $e = -(t + t^{L-1})/(1+t^L)$, XY
  $e = -I_1(K)/I_0(K)$, Heisenberg $e = -(\coth K - 1/K)$, for every CPU algorithm. Multi-process
  scans are checked the same way.
- **GPU**: Ising, q-state Potts ($Z=\lambda_1^L+(q-1)\lambda_2^L$), XY and Heisenberg chains plus
  4×4 Ising enumeration, for local and Swendsen–Wang updates. The parallel-tempering scan reproduces
  the universal Binder cumulant $U^*\approx0.611$ at $T_c$, and the XY helicity modulus has the
  spin-wave limit $\Upsilon\approx J-T/4$.
- Metropolis vs Wolff agreement for 2D XY and 3D Heisenberg, plus the Potts(q=2) ↔ Ising mapping,
  the estimators on synthetic AR(1) data, lattice geometry, spin normalization and seeding.

## Notes

- Numba draws from one global RNG stream. Every model re-seeds it on construction, so
  `seed=` makes a single run reproducible. For parallel runs, use separate processes.
- `temperature_scan(anneal=True)` (the default) visits temperatures from hot to cold and reuses
  the configuration between points.
- Kernels are compiled with `cache=True`, so only the very first run pays the compilation time.
