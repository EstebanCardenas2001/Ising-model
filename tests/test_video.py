import numpy as np
import pytest

from spinmodels.video.catalog import SYSTEMS, cooling_path, temperature_grid
from spinmodels.video.render import vortices


def test_vortex_detection_on_a_vortex_pair():
    L = 32
    y, x = np.mgrid[:L, :L].astype(float)
    # Vortex at plaquette centre (10.5, 10.5), antivortex at (20.5, 20.5).
    theta = np.arctan2(y - 10.5, x - 10.5) - np.arctan2(y - 20.5, x - 20.5)
    img = ((theta % (2 * np.pi)) * 256 / (2 * np.pi)).astype(np.uint8)
    plus, minus = vortices(img)
    assert [tuple(p) for p in plus] == [(10.5, 10.5)]
    assert [tuple(p) for p in minus] == [(20.5, 20.5)]


def test_temperature_grid_is_denser_near_tc():
    T = temperature_grid(1.0, 4.0, 41, Tc=2.269)
    assert T[0] == pytest.approx(1.0) and T[-1] == pytest.approx(4.0)
    gaps = np.diff(T)
    assert np.all(gaps > 0)
    assert gaps[np.argmin(np.abs(T[:-1] - 2.269))] < 0.6 * gaps.mean()
    assert gaps[np.argmin(np.abs(T[:-1] - 2.269))] < gaps[0] / 3


def test_cooling_path_holds_and_is_monotonic():
    path = cooling_path(3.0, 1.0, 300, Tc=2.0, hold=20)
    assert len(path) == 300
    assert np.all(path[:20] == path[0]) and np.all(path[-20:] == path[-1])
    assert np.all(np.diff(path) <= 0)


def test_every_system_has_a_caption():
    for s in SYSTEMS.values():
        lo, hi = sorted(s.video_range)
        for T in np.linspace(lo, hi, 7):
            head, body = s.phase(T)
            assert head and body
