import time
from typing import Optional, Tuple

import numpy as np


class LegacyRandom:
    multiplier = 0x5DEECE66D
    addend = 0xB
    mask = (1 << 48) - 1

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = time.time_ns()
        self.seed = (int(seed) ^ self.multiplier) & self.mask

    def next(self, bits: int) -> int:
        self.seed = (self.seed * self.multiplier + self.addend) & self.mask
        return self.seed >> (48 - bits)

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & (bound - 1)) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) >= 0:
                return val


def init_parameters(
    max_aid: int,
    max_bid: int,
    max_cid: int,
    rank: int,
    initscale: float,
    initscale2: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = LegacyRandom(seed)
    scale = 1000
    r = rank

    U = np.zeros((r + 1, max_aid + 1, r + 1), dtype=np.float64)
    V = np.zeros((r + 1, max_bid + 1, r + 1), dtype=np.float64)
    W = np.zeros((r + 1, max_cid + 1, r + 1), dtype=np.float64)

    D = np.zeros((max_aid + 1, r + 1), dtype=np.float64)
    E = np.zeros((max_bid + 1, r + 1), dtype=np.float64)
    F = np.zeros((max_cid + 1, r + 1), dtype=np.float64)

    for aid in range(1, max_aid + 1):
        for p in range(1, r + 1):
            for q in range(0, r + 1):
                U[p, aid, q] = initscale * rng.next_int(scale) / scale
            D[aid, p] = initscale2 * rng.next_int(scale) / scale

    for bid in range(1, max_bid + 1):
        for p in range(1, r + 1):
            for q in range(0, r + 1):
                V[p, bid, q] = initscale * rng.next_int(scale) / scale
            E[bid, p] = initscale2 * rng.next_int(scale) / scale

    for cid in range(1, max_cid + 1):
        for p in range(1, r + 1):
            for q in range(0, r + 1):
                W[p, cid, q] = initscale * rng.next_int(scale) / scale
            F[cid, p] = initscale2 * rng.next_int(scale) / scale

    return U, V, W, D, E, F
