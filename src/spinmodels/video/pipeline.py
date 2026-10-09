"""End-to-end production of the phase-transition videos.

1. **Equilibrium scans (GPU).** For every system and size: all temperatures
   (x several independent chains) in one batched GPU simulation with
   Swendsen-Wang + local updates and parallel tempering. Cached in
   ``<out>/data``.
2. **Scenes (GPU -> CPU).** The GPU runs each scene's display simulations
   and streams frame chunks to RAM disk; a pool of CPU workers renders
   them into video segments (encoded with NVENC), which are finally
   concatenated into ``<out>/videos/<scene>.mp4``.

Run ``python -m spinmodels.video --help``.
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .catalog import SYSTEMS, temperature_grid
from .produce import Scene, write_chunks

# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
PRESETS = {
    # Pipeline check: small scans, short videos.
    "quick": dict(nT=24, n_equil=200, n_measure=1000, frames=240, mosaic_frames=150, target_sites=2**21),
    # Production.
    "full": dict(nT=64, n_equil=1500, n_measure=12000, frames=1500, mosaic_frames=900, target_sites=2**23),
}

MOSAIC_T = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.05, 1.08, 1.12, 1.20)
MOSAIC_T_FIRST_ORDER = (0.90, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1.00, 1.005, 1.01, 1.015, 1.02, 1.03, 1.05, 1.10)


def scenes(preset: str) -> list[Scene]:
    p = PRESETS[preset]
    F, FM = p["frames"], p["mosaic_frames"]
    out = []
    for key, s in SYSTEMS.items():
        algo = "heat-bath dynamics" if s.dim == 1 else "Swendsen–Wang + local updates on the GPU"
        geom = "space-time diagram, one row per sweep" if s.dim == 1 else (
            "slice through the 3D lattice" if s.dim == 3 else f"{s.display_L}×{s.display_L} lattice")
        out.append(Scene(f"{key}", "single", [key], n_frames=F, title=s.title,
                         subtitle=f"Slow cooling · {geom} · L = {s.display_L} · {algo}"))
    out += [
        Scene("compare_ising_dimensions", "compare", ["ising1d", "ising2d", "ising3d"], n_frames=F,
              title="Ising model: the role of dimension",
              subtitle="Same Hamiltonian, same temperature. 1D never orders; T$_c$ grows with the number of neighbours",
              T_range=(6.0, 0.4), L={"ising1d": 384, "ising2d": 384, "ising3d": 128}),
        Scene("compare_2d_symmetries", "compare", ["ising2d", "potts3_2d", "potts8_2d", "xy2d", "heis2d"],
              n_frames=F, title="Two dimensions: the role of symmetry",
              subtitle="Discrete symmetries order (continuous or first order); XY has a topological BKT "
                       "transition; Heisenberg never orders (Mermin–Wagner)",
              T_range=(3.0, 0.25), L={k: 256 for k in ("ising2d", "potts3_2d", "potts8_2d", "xy2d", "heis2d")}),
        Scene("compare_continuous_symmetry", "compare", ["xy2d", "xy3d", "heis2d", "heis3d"], n_frames=F,
              title="Continuous spins in 2D vs 3D",
              subtitle="In 3D both XY and Heisenberg order; in 2D spin waves forbid true long-range order",
              T_range=(3.5, 0.25), L={"xy2d": 320, "heis2d": 320, "xy3d": 96, "heis3d": 96}),
    ]
    for key in ("ising2d", "ising3d", "potts3_2d", "potts8_2d", "xy2d", "xy3d", "heis3d"):
        s = SYSTEMS[key]
        first_order = key == "potts8_2d"
        out.append(Scene(
            f"mosaic_{key}", "mosaic", [key], n_frames=FM, local_sweeps=2, cluster=False,
            t_reduced=MOSAIC_T_FIRST_ORDER if first_order else MOSAIC_T,
            title=s.title,
            subtitle=(
                "Sixteen temperatures around T$_c$, each an equilibrium state evolving under local "
                "(single-spin) dynamics. " + (
                    "This transition is first order: there is no critical point. Ordered and disordered "
                    "phases coexist near T$_c$ and are separated by sharp interfaces; the order parameter "
                    "jumps instead of vanishing continuously."
                    if first_order else
                    "Near T$_c$ domains of all sizes appear and relax very slowly (critical slowing down); "
                    "far from T$_c$ the dynamics is fast.")),
        ))
    return out


# --------------------------------------------------------------------------- #
# Scans
# --------------------------------------------------------------------------- #
def run_scans(preset: str, data_dir: Path, systems=None, force: bool = False) -> None:
    from ..gpu.scan import gpu_temperature_scan

    p = PRESETS[preset]
    data_dir.mkdir(parents=True, exist_ok=True)
    for key in systems or SYSTEMS:
        s = SYSTEMS[key]
        temps = temperature_grid(*s.T_range, p["nT"], s.Tc, s.scan_sharpness)
        for L in s.scan_L[preset]:
            path = data_dir / f"scan_{key}_L{L}.npz"
            if path.exists() and not force:
                continue
            chains = int(np.clip(p["target_sites"] // (p["nT"] * L**s.dim), 1, 32))
            res = gpu_temperature_scan(s.model, L, s.dim, temps, n_equil=p["n_equil"], n_measure=p["n_measure"],
                                       n_chains=chains, seed=hash((key, L)) % 2**31, **s.params)
            res.save(path)


def plot_scans(preset: str, data_dir: Path, fig_dir: Path) -> None:
    """Static finite-size-scaling figures, one per system."""
    import matplotlib

    matplotlib.use("Agg")
    from ..plotting import plot_scan
    from .render import load_scans

    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, s in SYSTEMS.items():
        scans = load_scans(key, data_dir, preset)
        if not scans:
            continue
        obs = ["energy", "magnetization", "specific_heat", "susceptibility", "binder"]
        if "stiffness" in scans[0].mean:
            obs.append("stiffness")
        fig, axes = plot_scan(scans, observables=obs, Tc=s.Tc, title=f"{s.title}: finite-size scaling")
        fig.savefig(fig_dir / f"scan_{key}.png", dpi=130)
        matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------------- #
def make_videos(preset: str, out: Path, only=None, workers: int | None = None, seed: int = 0) -> list[Path]:
    from .render import concat_segments, render_chunk

    data_dir, video_dir = out / "data", out / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/dev/shm") if Path("/dev/shm").is_dir() else out
    work = tmp / f"spinmodels_frames_{os.getpid()}"
    seg_dir = out / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    todo = [sc for sc in scenes(preset) if not only or sc.name in only]
    produced = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: list[Future] = []
        per_scene: dict[str, list[tuple[int, Future]]] = {}
        for sc in todo:
            t0 = time.perf_counter()
            per_scene[sc.name] = []
            frame0 = 0
            for idx, chunk in enumerate(write_chunks(sc, work, seed=seed)):
                n = len(np.load(chunk)["T"])
                seg = seg_dir / f"{sc.name}_{idx:04d}.mp4"
                fut = pool.submit(render_chunk, sc.to_dict(), str(chunk), str(seg), str(data_dir), preset, frame0)
                per_scene[sc.name].append((idx, fut))
                pending.append(fut)
                frame0 += n
                # Back-pressure: keep RAM-disk usage bounded.
                while sum(not f.done() for f in pending) > 2 * workers:
                    time.sleep(0.05)
            print(f"[produce] {sc.name}: {sc.n_frames} frames simulated in {time.perf_counter() - t0:.1f}s",
                  flush=True)
        for sc in todo:
            segs = [Path(f.result()) for _, f in sorted(per_scene[sc.name])]
            dest = video_dir / f"{sc.name}.mp4"
            concat_segments(segs, dest)
            for sgm in segs:
                sgm.unlink()
            produced.append(dest)
            print(f"[video] {dest}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(seg_dir, ignore_errors=True)
    return produced


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m spinmodels.video", description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: output/<preset>)")
    ap.add_argument("--only", nargs="*", help="scene names to render (default: all)")
    ap.add_argument("--workers", type=int, default=None, help="render processes")
    ap.add_argument("--skip-scans", action="store_true")
    ap.add_argument("--rescan", action="store_true", help="recompute cached scans")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    args = ap.parse_args(argv)

    if args.list:
        for sc in scenes(args.preset):
            print(f"{sc.name:32s} {sc.kind:8s} {', '.join(sc.systems)}")
        return
    out = args.out or Path("output") / args.preset
    t0 = time.perf_counter()
    if not args.skip_scans:
        run_scans(args.preset, out / "data", force=args.rescan)
        plot_scans(args.preset, out / "data", out / "figures")
        print(f"[scans] done in {time.perf_counter() - t0:.0f}s", flush=True)
    make_videos(args.preset, out, only=args.only, workers=args.workers)
    print(f"[done] {time.perf_counter() - t0:.0f}s total", flush=True)
