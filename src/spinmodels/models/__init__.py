from .base import LatticeModel
from .ising import IsingModel
from .potts import PottsModel
from .vector import HeisenbergModel, VectorModel, XYModel

#: Registry used by the command-line interface.
MODELS = {
    "ising": IsingModel,
    "potts": PottsModel,
    "xy": XYModel,
    "heisenberg": HeisenbergModel,
}

__all__ = [
    "LatticeModel",
    "IsingModel",
    "PottsModel",
    "VectorModel",
    "XYModel",
    "HeisenbergModel",
    "MODELS",
]
