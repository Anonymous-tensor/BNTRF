from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    train: str = "data/D1/tr.txt"
    valid: str = "data/D1/va.txt"
    test: str = "data/D1/te.txt"
    separator: str = "::"

    rank: int = 5
    train_round: int = 1000
    initscale: float = 0.25
    initscale2: float = 0.10
    threshold: int = 2
    errorgap: float = 1e-5

    population: int = 5
    c1: float = 2.0
    c2: float = 2.0
    w: float = 0.724
    alpha: float = 0.5

    min_lambda_reg: float = 0.01
    max_lambda_reg: float = 0.10
    min_lambda_b: float = 0.01
    max_lambda_b: float = 0.10
    velocity_ratio: float = 0.2

    seed: Optional[int] = None
    pso_seed: Optional[int] = None
    log10_value_plus_one: bool = False
    print_every: int = 1

    use_current_validation_fitness: bool = True
