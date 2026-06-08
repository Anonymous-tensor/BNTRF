import time
from dataclasses import asdict

from .config import Config
from .data_utils import get_index_ranges, read_tensor_file
from .initialize import init_parameters
from .kernels import train_kernel
from .pso_utils import init_pso_arrays


class BNTRFPSOTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.result = None
        self.factors = None
        self.index_info = None

    def fit(self):
        cfg = self.config

        train_indices, train_values = read_tensor_file(
            cfg.train, cfg.separator, cfg.log10_value_plus_one
        )
        valid_indices, valid_values = read_tensor_file(
            cfg.valid, cfg.separator, cfg.log10_value_plus_one
        )
        test_indices, test_values = read_tensor_file(
            cfg.test, cfg.separator, cfg.log10_value_plus_one
        )

        min_ids, max_ids = get_index_ranges(train_indices, valid_indices, test_indices)
        max_aid, max_bid, max_cid = max_ids
        self.index_info = {"min_ids": min_ids, "max_ids": max_ids}

        U, V, W, D, E, F = init_parameters(
            max_aid=max_aid,
            max_bid=max_bid,
            max_cid=max_cid,
            rank=cfg.rank,
            initscale=cfg.initscale,
            initscale2=cfg.initscale2,
            seed=cfg.seed,
        )

        px, pv, min_x, max_x, min_v, max_v, rand1, rand2 = init_pso_arrays(
            population=cfg.population,
            train_round=cfg.train_round,
            min_lambda_reg=cfg.min_lambda_reg,
            max_lambda_reg=cfg.max_lambda_reg,
            min_lambda_b=cfg.min_lambda_b,
            max_lambda_b=cfg.max_lambda_b,
            velocity_ratio=cfg.velocity_ratio,
            seed=cfg.pso_seed,
        )

        start = time.perf_counter()
        result = train_kernel(
            train_indices,
            train_values,
            valid_indices,
            valid_values,
            test_indices,
            test_values,
            U,
            V,
            W,
            D,
            E,
            F,
            cfg.rank,
            max_aid,
            max_bid,
            max_cid,
            cfg.train_round,
            cfg.threshold,
            cfg.errorgap,
            cfg.print_every,
            px,
            pv,
            min_x,
            max_x,
            min_v,
            max_v,
            rand1,
            rand2,
            cfg.c1,
            cfg.c2,
            cfg.w,
            cfg.alpha,
            cfg.use_current_validation_fitness,
        )
        elapsed = time.perf_counter() - start

        (
            valid_rmse_history,
            valid_mae_history,
            test_rmse_history,
            test_mae_history,
            best_lambda_history,
            min_rmse_round,
            min_mae_round,
            stop_round,
            final_px,
            final_pv,
            g_best_value,
            g_best,
        ) = result

        self.result = {
            "config": asdict(cfg),
            "elapsed_sec": elapsed,
            "stop_round": int(stop_round),
            "best_lambda_reg": float(g_best_value[0]),
            "best_lambda_b": float(g_best_value[1]),
            "g_best": float(g_best),
            "min_rmse_round": int(min_rmse_round),
            "min_mae_round": int(min_mae_round),
            "test_min_rmse": float(test_rmse_history[min_rmse_round]),
            "test_min_mae": float(test_mae_history[min_mae_round]),
            "valid_rmse_history": valid_rmse_history,
            "valid_mae_history": valid_mae_history,
            "test_rmse_history": test_rmse_history,
            "test_mae_history": test_mae_history,
            "best_lambda_history": best_lambda_history,
            "final_px": final_px,
            "final_pv": final_pv,
        }
        self.factors = {"U": U, "V": V, "W": W, "D": D, "E": E, "F": F}
        return self.result
