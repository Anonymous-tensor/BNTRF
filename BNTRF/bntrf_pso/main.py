import argparse

from .config import Config
from .trainer import BNTRFPSOTrainer


def build_config(args) -> Config:
    return Config(
        train=args.train,
        valid=args.valid,
        test=args.test,
        separator=args.separator,
        rank=args.rank,
        train_round=args.train_round,
        initscale=args.initscale,
        initscale2=args.initscale2,
        threshold=args.threshold,
        errorgap=args.errorgap,
        population=args.population,
        c1=args.c1,
        c2=args.c2,
        w=args.w,
        alpha=args.alpha,
        min_lambda_reg=args.min_lambda_reg,
        max_lambda_reg=args.max_lambda_reg,
        min_lambda_b=args.min_lambda_b,
        max_lambda_b=args.max_lambda_b,
        velocity_ratio=args.velocity_ratio,
        seed=args.seed,
        pso_seed=args.pso_seed,
        log10_value_plus_one=args.log10_value_plus_one,
        print_every=args.print_every,
        use_current_validation_fitness=not args.fixed_initial_fitness,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="BNTRF with PSO-based hyperparameter adaptation")
    parser.add_argument("--train", default="data/D1/tr.txt")
    parser.add_argument("--valid", default="data/D1/va.txt")
    parser.add_argument("--test", default="data/D1/te.txt")
    parser.add_argument("--separator", default="::")

    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--train-round", type=int, default=1000)
    parser.add_argument("--initscale", type=float, default=0.25)
    parser.add_argument("--initscale2", type=float, default=0.10)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--errorgap", type=float, default=1e-5)

    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--c1", type=float, default=2.0)
    parser.add_argument("--c2", type=float, default=2.0)
    parser.add_argument("--w", type=float, default=0.724)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--min-lambda-reg", type=float, default=0.01)
    parser.add_argument("--max-lambda-reg", type=float, default=0.10)
    parser.add_argument("--min-lambda-b", type=float, default=0.01)
    parser.add_argument("--max-lambda-b", type=float, default=0.10)
    parser.add_argument("--velocity-ratio", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pso-seed", type=int, default=None)
    parser.add_argument("--log10-value-plus-one", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--fixed-initial-fitness", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    trainer = BNTRFPSOTrainer(cfg)
    result = trainer.fit()

    print(f"Elapsed time: {result['elapsed_sec']:.6f}s")
    print(f"Stop round: {result['stop_round']}")
    print(f"Best lambda_reg: {result['best_lambda_reg']}")
    print(f"Best lambda_b: {result['best_lambda_b']}")
    print(f"Test min RMSE: {result['test_min_rmse']} at round {result['min_rmse_round']}")
    print(f"Test min MAE: {result['test_min_mae']} at round {result['min_mae_round']}")


if __name__ == "__main__":
    main()
