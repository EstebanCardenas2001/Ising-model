"""CPU side of the video pipeline: draw frames with Matplotlib (Agg) and
encode them with ffmpeg (NVENC on the GPU when available).

Each worker renders one chunk of frames into one video segment; segments
are concatenated losslessly at the end. Figures are built once per chunk and
only the changing artists are updated per frame.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure

from ..simulation import ScanResult
from .catalog import SYSTEMS, System

W, H, DPI = 1920, 1080, 100

# Dark theme (validated dark-mode steps of the categorical palette).
BG = "#121211"
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#8a8984"
GRID = "#2e2e2c"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MARKER = "#ffffff"

_RC = {
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "legend.labelcolor": INK2,
}

LABELS = {
    "magnetization": r"order parameter $\langle|m|\rangle$",
    "susceptibility": r"susceptibility $\chi$",
    "specific_heat": r"specific heat $C$",
    "binder": r"Binder cumulant $U_4$",
    "energy": r"energy $\langle E\rangle/N$",
    "stiffness": r"spin stiffness $\Upsilon$",
    "histogram": r"energy histogram $P(E/N)$",
    "zoom": "zoom: spins and vortices",
}


# --------------------------------------------------------------------------- #
# Colour lookup tables for lattice images
# --------------------------------------------------------------------------- #
def _lut_from_cmap(cmap, n=256) -> np.ndarray:
    return (np.array([cmap(i / (n - 1))[:3] for i in range(n)]) * 255).astype(np.uint8)


@lru_cache(maxsize=None)
def lut(system_key: str) -> np.ndarray:
    s = SYSTEMS[system_key]
    if s.model == "ising":
        cols = SERIES[:2]
    elif s.model == "potts":
        cols = SERIES[: s.params["q"]]
    elif s.model == "xy":
        return _lut_from_cmap(matplotlib.colormaps["twilight"])
    else:  # Heisenberg: diverging blue <-> gray <-> red, value 255 = aligned (red)
        cmap = LinearSegmentedColormap.from_list("div", ["#3987e5", "#383835", "#e66767"])
        return _lut_from_cmap(cmap)
    table = np.zeros((256, 3), dtype=np.uint8)
    table[: len(cols)] = (np.array([to_rgb(c) for c in cols]) * 255).astype(np.uint8)
    return table


def colorize(system_key: str, img: np.ndarray) -> np.ndarray:
    return lut(system_key)[img]


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _encoder() -> list[str]:
    """NVENC (GPU) H.264 if this ffmpeg/GPU supports it, else libx264."""
    test = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
         "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    if test.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0",
                "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


class SegmentWriter:
    def __init__(self, path: Path, fps: int):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *_encoder(), "-pix_fmt", "yuv420p", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, canvas: FigureCanvasAgg) -> None:
        canvas.draw()
        self.proc.stdin.write(canvas.buffer_rgba())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def concat_segments(segments: list[Path], out: Path) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    listing.unlink()


# --------------------------------------------------------------------------- #
# Shared drawing helpers
# --------------------------------------------------------------------------- #
def load_scans(system_key: str, data_dir: Path, preset: str) -> list[ScanResult]:
    s = SYSTEMS[system_key]
    out = []
    for L in s.scan_L[preset]:
        path = data_dir / f"scan_{system_key}_L{L}.npz"
        if path.exists():
            out.append(ScanResult.load(path))
    return out


def _new_figure() -> tuple[Figure, FigureCanvasAgg]:
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig, FigureCanvasAgg(fig)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _image_axes(fig, rect) -> "matplotlib.axes.Axes":
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def _curve(ax, res: ScanResult, key: str, color: str, label: str | None = None, x_scale: float = 1.0):
    T = res.temperatures / x_scale
    y, err = res.mean[key], res.error[key]
    ax.fill_between(T, y - err, y + err, color=color, alpha=0.25, lw=0)
    ax.plot(T, y, color=color, lw=2, label=label)


def vortices(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plaquette winding numbers of an XY angle image (uint8, 256 = 2π).
    Returns (row, col) arrays of vortex (+1) and antivortex (−1) plaquette centres."""
    theta = img.astype(np.float64) * (2 * np.pi / 256)

    def d(a, b):
        return (b - a + np.pi) % (2 * np.pi) - np.pi

    t00, t01 = theta, np.roll(theta, -1, 1)
    t11, t10 = np.roll(t01, -1, 0), np.roll(theta, -1, 0)
    w = np.rint((d(t00, t01) + d(t01, t11) + d(t11, t10) + d(t10, t00)) / (2 * np.pi)).astype(int)
    return np.argwhere(w > 0) + 0.5, np.argwhere(w < 0) + 0.5


# --------------------------------------------------------------------------- #
# Single-system scene
# --------------------------------------------------------------------------- #
class SingleScene:
    """Big lattice view + equilibrium curves with a moving temperature cursor."""

    def __init__(self, scene: dict, data_dir: Path, preset: str):
        self.scene = scene
        self.s: System = SYSTEMS[scene["systems"][0]]
        s = self.s
        self.scans = load_scans(s.key, data_dir, preset)
        self.fig, self.canvas = _new_figure()
        fig = self.fig
        T_hi, T_lo = s.video_range
        self.T_lo, self.T_hi = T_lo, T_hi

        fig.text(0.02, 0.955, scene["title"] or s.title, fontsize=30, weight="bold", color=INK)
        fig.text(0.02, 0.918, scene["subtitle"], fontsize=14, color=INK2)

        self.ax_img = _image_axes(fig, [0.02, 0.035, 0.46, 0.86])
        self.im = None
        self.T_text = fig.text(0.515, 0.94, "", fontsize=28, color=INK, weight="bold")
        self.phase_title = fig.text(0.515, 0.118, "", fontsize=20, color=INK, weight="bold")
        self.phase_text = fig.text(0.515, 0.095, "", fontsize=14, color=INK2, va="top", wrap=True)
        self.phase_text._get_wrap_line_width = lambda: 0.46 * W  # wrap inside the right column

        self.cursors, self.dots, self.hist_ax, self.zoom_ax = [], {}, None, None
        x0, y0, w, h, gx, gy = 0.535, 0.60, 0.122, 0.26, 0.04, 0.36
        Tc = s.Tc
        for idx, key in enumerate(s.panels):
            col, row = idx % 3, idx // 3
            ax = fig.add_axes([x0 + col * (w + gx), y0 - row * gy, w, h])
            _style(ax)
            ax.set_title(LABELS[key], fontsize=13, loc="left")
            if key == "histogram":
                self.hist_ax = ax
                ax.set_yticks([])
                ax.set_xlabel("E/N")
                continue
            if key == "zoom":
                self.zoom_ax = ax
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if key in s.log_panels:
                ax.set_yscale("log")
                ax.set_title(LABELS[key] + " (log)", fontsize=13, loc="left", color=INK)
            for i, res in enumerate(self.scans):
                _curve(ax, res, key, SERIES[i], f"L = {res.L}")
            if key == "stiffness":
                Tg = np.linspace(T_lo, T_hi, 50)
                ax.plot(Tg, 2 * Tg / np.pi, color=INK2, ls="--", lw=1.5, label=r"$2T/\pi$")
                ax.legend(loc="upper right", fontsize=10)
            if Tc:
                ax.axvline(Tc, color=MUTED, ls=":", lw=1.5)
            ax.set_xlim(T_lo, T_hi)
            ax.set_xlabel("T / J")
            self.cursors.append(ax.axvline(T_hi, color=INK, lw=1.5, alpha=0.9))
            if key in ("magnetization", "energy"):
                (self.dots[key],) = ax.plot([], [], "o", ms=9, color=MARKER, mec=BG, mew=2,
                                            label=f"live, L = {s.display_L}", zorder=5)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=10)
        self._hist_index = None

    def _set_histogram(self, T: float) -> None:
        if not self.scans or self.hist_ax is None:
            return
        Ts = self.scans[0].temperatures
        j = int(np.argmin(np.abs(Ts - T)))
        ax = self.hist_ax
        samples = [r.extras["energy_samples"][:, j].astype(np.float64) for r in self.scans]
        if j != self._hist_index:
            self._hist_index = j
            ax.cla()
            _style(ax)
            ax.set_yticks([])
            ax.set_xlabel("E/N")
            lo = min(x.min() for x in samples)
            hi = max(x.max() for x in samples)
            pad = 0.05 * (hi - lo + 1e-3)
            self._bins = np.linspace(lo - pad, hi + pad, 60)
            self._stairs = [ax.stairs(np.zeros(59), self._bins, color=SERIES[i], lw=2, fill=False)
                            for i in range(len(samples))]
            ax.set_xlim(self._bins[0], self._bins[-1])
        # Single-histogram reweighting (Ferrenberg-Swendsen) from the nearest
        # scan temperature T_j to the live T: weights exp(-(1/T - 1/T_j) N E/N).
        top = 0.0
        for res, x, st in zip(self.scans, samples, self._stairs):
            logw = -(1.0 / T - 1.0 / Ts[j]) * res.N * x
            w = np.exp(logw - logw.max())
            hist, _ = np.histogram(x, bins=self._bins, weights=w, density=True)
            st.set_data(hist)
            top = max(top, hist.max())
        ax.set_ylim(0, 1.1 * top)
        ax.set_title(f"P(E/N) at T = {T:.3f}", fontsize=13, loc="left", color=INK)

    def _set_zoom(self, img: np.ndarray) -> None:
        ax, n = self.zoom_ax, 32
        ax.cla()
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(LABELS["zoom"], fontsize=13, loc="left", color=INK)
        sub = img[:n, :n]
        ax.imshow(colorize(self.s.key, sub), interpolation="nearest", alpha=0.55)
        theta = sub.astype(float) * (2 * np.pi / 256)
        yy, xx = np.mgrid[:n, :n]
        ax.quiver(xx, yy, np.cos(theta), -np.sin(theta), color=INK, pivot="middle", scale=1.1,
                  scale_units="xy", width=0.006, headwidth=3)
        plus, minus = vortices(sub)
        for pts, c, mk in ((plus, SERIES[1], "o"), (minus, SERIES[0], "s")):
            ok = (pts[:, 0] < n - 1) & (pts[:, 1] < n - 1)
            ax.plot(pts[ok, 1], pts[ok, 0], mk, ms=11, mfc="none", mec=c, mew=2.5)
        ax.set_xlim(-0.5, n - 1.5)
        ax.set_ylim(n - 1.5, -0.5)

    def draw(self, chunk: dict, i: int) -> None:
        s = self.s
        T = float(chunk["T"][i, 0])
        img = chunk["image0"][i]
        rgb = colorize(s.key, img)
        if self.im is None:
            self.im = self.ax_img.imshow(rgb, interpolation="nearest")
            if s.dim == 1:
                self.ax_img.set_ylabel("time  ↓  (newest row at the bottom)", color=INK2)
                self.ax_img.set_xlabel("position along the chain", color=INK2)
            if self.zoom_ax is not None:
                self.ax_img.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5), 32, 32, fill=False,
                                                                   ec=INK, lw=1.5))
        else:
            self.im.set_data(rgb)
        Tc = s.Tc
        ratio = f"    T/T$_c$ = {T / Tc:.3f}" if Tc else ""
        self.T_text.set_text(f"T = {T:.3f} J{ratio}")
        head, body = s.phase(T)
        self.phase_title.set_text(head)
        self.phase_text.set_text(body)
        for c in self.cursors:
            c.set_xdata([T, T])
        if "magnetization" in self.dots:
            self.dots["magnetization"].set_data([T], [chunk["m"][i, 0]])
        if "energy" in self.dots:
            self.dots["energy"].set_data([T], [chunk["e"][i, 0]])
        self._set_histogram(T)
        if self.zoom_ax is not None:
            self._set_zoom(img)


# --------------------------------------------------------------------------- #
# Comparison scene
# --------------------------------------------------------------------------- #
class CompareScene:
    """Several systems side by side at the same absolute temperature."""

    def __init__(self, scene: dict, data_dir: Path, preset: str):
        self.scene = scene
        self.systems = [SYSTEMS[k] for k in scene["systems"]]
        n = len(self.systems)
        self.fig, self.canvas = _new_figure()
        fig = self.fig
        T_hi, T_lo = scene["T_range"]
        fig.text(0.02, 0.95, scene["title"], fontsize=28, weight="bold", color=INK)
        fig.text(0.02, 0.915, scene["subtitle"], fontsize=14, color=INK2)
        self.T_text = fig.text(0.98, 0.95, "", fontsize=28, weight="bold", color=INK, ha="right")

        gap = 0.012
        size_w = (0.96 - gap * (n - 1)) / n
        size = min(size_w, 0.50 * W / H)  # square tiles, height <= 50% of frame
        total = n * size + (n - 1) * gap
        left = (1 - total) / 2
        self.ims, self.tile_labels, self.state_labels = [None] * n, [], []
        self.axes = []
        for i, s in enumerate(self.systems):
            x = left + i * (size + gap)
            h = size * W / H
            ax = _image_axes(fig, [x, 0.88 - h, size, h])
            self.axes.append(ax)
            fig.text(x, 0.885, s.short, fontsize=17, weight="bold", color=SERIES[i])
            tc = f"T$_c$ = {s.Tc:.3f}" if s.Tc else "no transition"
            fig.text(x + size, 0.885, tc, fontsize=12, color=INK2, ha="right")
            self.state_labels.append(fig.text(x, 0.86 - h, "", fontsize=13, color=INK2, va="top"))

        self.cursors, self.dots = [], []
        plot_specs = [("magnetization", 0.06), ("specific_heat", 0.54)]
        plot_top = min(0.86 - size * W / H - 0.09, 0.40)
        for key, x in plot_specs:
            ax = fig.add_axes([x, 0.08, 0.41, plot_top - 0.08])
            _style(ax)
            ax.set_title(LABELS[key] + "  (largest scanned L; shaded: ±1σ)", fontsize=13, loc="left")
            for i, s in enumerate(self.systems):
                scans = load_scans(s.key, data_dir, preset)
                if scans:
                    _curve(ax, scans[-1], key, SERIES[i], s.short)
                if s.Tc and T_lo < s.Tc < T_hi:
                    ax.axvline(s.Tc, color=SERIES[i], ls=":", lw=1.5)
            ax.set_xlim(T_lo, T_hi)
            ax.set_xscale("log")
            if key == "specific_heat":
                # First-order peaks (8-state Potts) grow like L^d: a log axis keeps
                # the continuous transitions visible next to them.
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.03)
                ax.set_title(LABELS[key] + "  (log scale; largest scanned L)", fontsize=13, loc="left", color=INK)
            ticks = [t for t in (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 6) if T_lo <= t <= T_hi]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("T / J  (log scale; dotted: T$_c$)")
            self.cursors.append(ax.axvline(T_hi, color=INK, lw=1.5))
            if key == "magnetization":
                ax.legend(loc="upper right", ncols=1, fontsize=11)
                for i in range(n):
                    (dot,) = ax.plot([], [], "o", ms=9, color=SERIES[i], mec=INK, mew=1.5, zorder=5)
                    self.dots.append(dot)

    def draw(self, chunk: dict, i: int) -> None:
        T = float(chunk["T"][i, 0])
        self.T_text.set_text(f"T = {T:.3f} J")
        for k, s in enumerate(self.systems):
            rgb = colorize(s.key, chunk[f"image{k}"][i])
            if self.ims[k] is None:
                self.ims[k] = self.axes[k].imshow(rgb, interpolation="nearest")
            else:
                self.ims[k].set_data(rgb)
            self.state_labels[k].set_text(_short_state(s, T))
            self.dots[k].set_data([T], [chunk["m"][i, k]])
        for c in self.cursors:
            c.set_xdata([T, T])


def _short_state(s: System, T: float) -> str:
    if s.Tc is None:
        return "no order at T > 0" + (" (ξ finite)" if s.dim == 1 else " (Mermin–Wagner)")
    t = T / s.Tc
    if s.key == "xy2d":
        return "free vortices" if t > 1.05 else ("BKT transition" if t > 0.97 else "quasi-long-range order")
    if s.key == "potts8_2d" and 0.99 < t < 1.01:
        return "first-order transition (coexistence)"
    return "disordered" if t > 1.04 else ("critical" if t > 0.97 else "ordered")


# --------------------------------------------------------------------------- #
# Mosaic scene
# --------------------------------------------------------------------------- #
class MosaicScene:
    """A 4x4 grid of fixed temperatures around T_c with local dynamics."""

    def __init__(self, scene: dict, data_dir: Path, preset: str):
        self.scene = scene
        self.s = s = SYSTEMS[scene["systems"][0]]
        self.t = np.asarray(scene["t_reduced"])
        self.fig, self.canvas = _new_figure()
        fig = self.fig
        n = int(np.sqrt(len(self.t)))
        side = 0.94 * H / W
        cell = side / n
        self.ims, self.axes = [None] * len(self.t), []
        for k, t in enumerate(self.t):
            r, c = divmod(k, n)
            x = 0.012 + c * cell
            y = 0.97 - (r + 1) * cell * W / H
            ax = _image_axes(fig, [x + 0.002, y + 0.004, cell - 0.004, cell * W / H - 0.008])
            self.axes.append(ax)
            ax.text(0.03, 0.97, f"{t:.2f} T$_c$", transform=ax.transAxes, fontsize=13, color=INK,
                    va="top", weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.7))
        xr = 0.012 + side + 0.03
        fig.text(xr, 0.93, scene["title"] or s.title, fontsize=28, weight="bold", color=INK)
        fig.text(xr, 0.895, scene["subtitle"], fontsize=14, color=INK2, va="top", wrap=True)
        txt = fig.texts[-1]
        txt._get_wrap_line_width = lambda: (1 - xr - 0.02) * W

        ax = fig.add_axes([xr + 0.03, 0.12, 1 - xr - 0.06, 0.48])
        _style(ax)
        ax.set_title(LABELS["magnetization"] + "  (curves: equilibrium scans)", fontsize=13, loc="left")
        for i, res in enumerate(load_scans(s.key, data_dir, preset)):
            _curve(ax, res, "magnetization", SERIES[i], f"L = {res.L}", x_scale=s.Tc)
        ax.axvline(1.0, color=MUTED, ls=":", lw=1.5)
        ax.set_xlim(self.t.min() - 0.03, self.t.max() + 0.03)
        ax.set_xlabel("T / T$_c$")
        (self.dots,) = ax.plot([], [], "o", ms=8, color=MARKER, mec=BG, mew=1.5, zorder=5,
                               label="live tiles")
        ax.legend(loc="upper right", fontsize=11)
        self.time_text = fig.text(xr, 0.035, "", fontsize=13, color=INK2)

    def draw(self, chunk: dict, i: int) -> None:
        for k in range(len(self.t)):
            rgb = colorize(self.s.key, chunk["images"][i, k])
            if self.ims[k] is None:
                self.ims[k] = self.axes[k].imshow(rgb, interpolation="nearest")
            else:
                self.ims[k].set_data(rgb)
        self.dots.set_data(self.t, chunk["m"][i])
        frame = int(chunk["frame0"]) + i
        self.time_text.set_text(f"Monte Carlo time: {frame * self.scene['local_sweeps']:,} sweeps")


SCENE_TYPES = {"single": SingleScene, "compare": CompareScene, "mosaic": MosaicScene}


def render_chunk(scene: dict, chunk_path: str, segment_path: str, data_dir: str, preset: str,
                 frame0: int, delete_chunk: bool = True) -> str:
    """Render one chunk of frames to a video segment (runs in a worker process)."""
    matplotlib.rcParams.update(_RC)
    data = dict(np.load(chunk_path))
    data["frame0"] = frame0
    view = SCENE_TYPES[scene["kind"]](scene, Path(data_dir), preset)
    writer = SegmentWriter(Path(segment_path), scene["fps"])
    n = len(data["T"])
    for i in range(n):
        view.draw(data, i)
        writer.write(view.canvas)
    writer.close()
    if delete_chunk:
        Path(chunk_path).unlink()
    return segment_path
