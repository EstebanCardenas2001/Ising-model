"""GPU side of the video pipeline: run the simulations and emit frame chunks.

A *scene* is one output video. Its simulations run on the GPU; every
``chunk`` frames the producer writes an ``.npz`` (to RAM disk) holding
compact uint8 images plus per-frame scalars, which the renderer turns into
video segments in parallel worker processes.

Display images are *gauge fixed*: cluster updates and spontaneous symmetry
breaking pick an arbitrary global orientation, so each image is shown
relative to the current order-parameter direction (majority Ising sign,
majority Potts state, magnetization direction for XY/Heisenberg). Global
symmetry operations then do not recolor the whole picture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cupy as cp
import numpy as np

from ..gpu import GPU_MODELS, GPUIsing, GPUModel, GPUPotts, GPUXY
from .catalog import SYSTEMS, System, cooling_path


@dataclass
class Scene:
    name: str
    kind: str  # "single" | "compare" | "mosaic"
    systems: list[str]
    n_frames: int = 1500
    fps: int = 30
    title: str = ""
    subtitle: str = ""
    #: compare scenes: absolute temperature range swept (hot -> cold).
    T_range: tuple[float, float] | None = None
    #: mosaic scenes: temperatures as fractions of Tc.
    t_reduced: tuple[float, ...] = ()
    #: lattice size override (compare / mosaic scenes).
    L: dict = field(default_factory=dict)
    local_sweeps: int = 2
    cluster: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Gauge-fixed display images
# --------------------------------------------------------------------------- #
class Imager:
    """Turns replica ``r`` of a GPU model into a gauge-fixed uint8 image.

    2D lattices are returned whole, 3D lattices as the first plane, 1D chains
    as a single row. The gauge reference is only updated when the order
    parameter is clearly nonzero, so it does not jitter in the disordered phase.
    """

    def __init__(self, model: GPUModel):
        self.model = model
        self.threshold = 4.0 / np.sqrt(model.N)
        n = model.R
        self.sign = np.ones(n)
        self.state = np.zeros(n, dtype=int)
        self.direction = np.zeros((n, max(model.n_components, 1)))
        self.direction[:, -1 if model.n_components == 3 else 0] = 1.0

    def _plane(self, r: int) -> cp.ndarray:
        x = self.model.spins[r].reshape(self.model.shape + self.model.spins.shape[2:])
        while x.ndim > (2 if self.model.n_components == 1 or isinstance(self.model, GPUPotts) else 3):
            x = x[0]
        return x

    def image(self, r: int, raw: np.ndarray, gauge: bool = True) -> np.ndarray:
        m = self.model
        mvec = raw[r, 2 : 2 + m.n_components] / m.N
        x = self._plane(r)
        if isinstance(m, GPUIsing):
            if gauge and abs(mvec[0]) > self.threshold:
                self.sign[r] = np.sign(mvec[0])
            img = (x * int(self.sign[r]) < 0).astype(cp.uint8)  # 0 = majority
        elif isinstance(m, GPUPotts):
            counts = cp.asnumpy(m.counts()[r])
            top2 = np.sort(counts)[-2:] / m.N
            if top2[1] - top2[0] > self.threshold:
                self.state[r] = int(np.argmax(counts))
            img = ((x.astype(cp.int16) - self.state[r]) % m.q).astype(cp.uint8)  # 0 = majority
        else:
            norm = np.linalg.norm(mvec)
            if norm > self.threshold:
                self.direction[r] = mvec / norm
            if isinstance(m, GPUXY):
                ref = np.arctan2(self.direction[r, 1], self.direction[r, 0])
                theta = cp.arctan2(x[..., 1], x[..., 0]) - ref
                img = ((theta % (2 * np.pi)) * (256 / (2 * np.pi))).astype(cp.uint8)  # 0 = aligned
            else:
                proj = x @ cp.asarray(self.direction[r], dtype=cp.float32)
                img = cp.clip((proj + 1) * 127.5, 0, 255).astype(cp.uint8)  # 255 = aligned
        return cp.asnumpy(img)


def _make_model(system: System, L: int, temperatures, seed: int) -> GPUModel:
    return GPU_MODELS[system.model](L, system.dim, temperatures, seed=seed, **system.params)


def _step(model: GPUModel, cluster: bool, local_sweeps: int) -> None:
    if cluster:
        model.sweep("swendsen_wang")
    if local_sweeps:
        model.sweep("local", local_sweeps)


def _equilibrate(model: GPUModel, n: int) -> None:
    for _ in range(n):
        model.sweep("swendsen_wang")
        model.sweep("metropolis_adapt" if model.n_components > 1 and not isinstance(model, GPUPotts) else "local")


# --------------------------------------------------------------------------- #
# Scene runners: yield one dict of arrays per frame
# --------------------------------------------------------------------------- #
class _Track:
    """One displayed system following a temperature path."""

    def __init__(self, system: System, L: int, path: np.ndarray, seed: int, cluster: bool, local_sweeps: int):
        self.system, self.path = system, path
        self.cluster = cluster and system.dim > 1
        self.local_sweeps = local_sweeps
        self.model = _make_model(system, L, [path[0]], seed)
        self.imager = Imager(self.model)
        _equilibrate(self.model, 300)
        if system.dim == 1:
            # Space-time diagram: a scrolling window of chain configurations.
            self.rows_per_frame = 8
            self.history = np.zeros((L, L), dtype=np.uint8)
            for _ in range(L // self.rows_per_frame):
                self._spacetime_frame()

    def _spacetime_frame(self) -> np.ndarray:
        rows = []
        for _ in range(self.rows_per_frame):
            self.model.sweep("local")
            # No gauge fixing: rows must stay comparable over time, and local
            # dynamics never flips the whole chain at once.
            rows.append(self.imager.image(0, self.model.measure(), gauge=False))
        self.history = np.vstack([self.history[self.rows_per_frame :], np.array(rows)])
        return self.history

    def frame(self, f: int) -> tuple[np.ndarray, float, float]:
        self.model.set_temperatures([self.path[f]])
        if self.system.dim == 1:
            img = self._spacetime_frame()
            raw = self.model.measure()
        else:
            _step(self.model, self.cluster, self.local_sweeps)
            raw = self.model.measure()
            img = self.imager.image(0, raw)
        e = (-self.model.J * raw[0, 0] - self.model.h * raw[0, 1]) / self.model.N
        m = np.linalg.norm(raw[0, 2 : 2 + self.model.n_components]) / self.model.N
        return img, e, m


def scene_paths(scene: Scene) -> dict[str, np.ndarray]:
    """Temperature per frame for each system of a single/compare scene."""
    if scene.kind == "single":
        s = SYSTEMS[scene.systems[0]]
        return {s.key: cooling_path(*s.video_range, scene.n_frames, s.Tc)}
    T_hi, T_lo = scene.T_range
    path = cooling_path(T_hi, T_lo, scene.n_frames, None)
    return {k: path for k in scene.systems}


def run_scene(scene: Scene, seed: int = 0):
    """Generator of per-frame dicts for ``scene`` (see module docstring)."""
    if scene.kind in ("single", "compare"):
        paths = scene_paths(scene)
        tracks = []
        for i, key in enumerate(scene.systems):
            s = SYSTEMS[key]
            L = scene.L.get(key, s.display_L)
            tracks.append(_Track(s, L, paths[key], seed + i, scene.cluster, scene.local_sweeps))
        for f in range(scene.n_frames):
            out = {"T": np.array([paths[k][f] for k in scene.systems])}
            es, ms = [], []
            for i, tr in enumerate(tracks):
                img, e, m = tr.frame(f)
                out[f"image{i}"] = img
                es.append(e)
                ms.append(m)
            out["e"], out["m"] = np.array(es), np.array(ms)
            yield out
    elif scene.kind == "mosaic":
        s = SYSTEMS[scene.systems[0]]
        temps = np.asarray(scene.t_reduced) * s.Tc
        L = scene.L.get(s.key, 240 if s.dim == 2 else 96)
        model = _make_model(s, L, temps, seed)
        imager = Imager(model)
        _equilibrate(model, 400)
        for _ in range(scene.n_frames):
            model.sweep("local", scene.local_sweeps)
            raw = model.measure()
            out = {
                "T": temps,
                "images": np.stack([imager.image(r, raw) for r in range(model.R)]),
                "e": (-model.J * raw[:, 0] - model.h * raw[:, 1]) / model.N,
                "m": np.linalg.norm(raw[:, 2 : 2 + model.n_components], axis=1) / model.N,
            }
            yield out
    else:
        raise ValueError(scene.kind)


def write_chunks(scene: Scene, outdir: Path, chunk: int = 60, seed: int = 0):
    """Run ``scene`` and save frames in chunks; yields each chunk's path."""
    outdir.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    index = 0
    for frame in run_scene(scene, seed):
        buf.append(frame)
        if len(buf) == chunk:
            yield _save(buf, outdir / f"{scene.name}_{index:04d}.npz")
            buf, index = [], index + 1
    if buf:
        yield _save(buf, outdir / f"{scene.name}_{index:04d}.npz")


def _save(frames: list[dict], path: Path) -> Path:
    np.savez(path, **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
    return path
