"""Union-find helpers shared by the Swendsen-Wang kernels."""

import numpy as np
from numba import njit


@njit(cache=True)
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


@njit(cache=True)
def union(parent, a, b):
    ra = find(parent, a)
    rb = find(parent, b)
    if ra < rb:
        parent[rb] = ra
    elif rb < ra:
        parent[ra] = rb


@njit(cache=True)
def flatten(parent):
    """Point every site directly at its root; return the largest cluster fraction."""
    N = parent.shape[0]
    size = np.zeros(N, dtype=np.int64)
    largest = 0
    for i in range(N):
        r = find(parent, i)
        parent[i] = r
        size[r] += 1
        if size[r] > largest:
            largest = size[r]
    return largest / N
