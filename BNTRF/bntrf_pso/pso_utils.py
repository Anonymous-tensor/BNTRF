from typing import Optional, Tuple

import numpy as np


def init_pso_arrays(
    population: int,
    train_round: int,
    min_lambda_reg: float,
    max_lambda_reg: float,
    min_lambda_b: float,
    max_lambda_b: float,
    velocity_ratio: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(None if seed is None else int(seed))

    min_x = np.asarray([min_lambda_reg, min_lambda_b], dtype=np.float64)
    max_x = np.asarray([max_lambda_reg, max_lambda_b], dtype=np.float64)
    max_v = velocity_ratio * (max_x - min_x)
    min_v = -max_v

    px = np.zeros((population, 2), dtype=np.float64)
    pv = np.zeros((population, 2), dtype=np.float64)
    for p in range(population):
        for h in range(2):
            px[p, h] = min_x[h] + rng.random() * (max_x[h] - min_x[h])

    rand1 = rng.random((train_round + 1, population)).astype(np.float64)
    rand2 = rng.random((train_round + 1, population)).astype(np.float64)
    return px, pv, min_x, max_x, min_v, max_v, rand1, rand2
