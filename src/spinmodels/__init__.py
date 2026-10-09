"""Monte Carlo simulation of classical lattice spin models."""

from .lattice import HypercubicLattice
from .models import (
    MODELS,
    HeisenbergModel,
    IsingModel,
    LatticeModel,
    PottsModel,
    VectorModel,
    XYModel,
)
from .observables import OBSERVABLES, binder_cumulant, integrated_autocorr_time, jackknife, thermodynamics
from .simulation import ScanResult, TimeSeries, parallel_tempering, run, temperature_scan

__version__ = "0.2.0"

__all__ = [
    "HypercubicLattice",
    "LatticeModel",
    "IsingModel",
    "PottsModel",
    "VectorModel",
    "XYModel",
    "HeisenbergModel",
    "MODELS",
    "OBSERVABLES",
    "binder_cumulant",
    "integrated_autocorr_time",
    "jackknife",
    "thermodynamics",
    "ScanResult",
    "TimeSeries",
    "run",
    "temperature_scan",
    "parallel_tempering",
]
