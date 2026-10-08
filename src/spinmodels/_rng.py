"""Seeding for Numba's internal random number generator.

Numba-compiled code draws from its own Mersenne Twister state, independent of
NumPy's global state and of ``np.random.Generator`` objects. It must therefore
be seeded from inside a jitted function.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)
