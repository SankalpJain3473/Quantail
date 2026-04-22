"""
run_backtest.py
===============
Full backtest: Quantail vs Delta Hedge vs No Hedge.

Usage:
  python run_backtest.py
  python run_backtest.py --train 200 --test 100 --plot
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.backtest import Backtester


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",  type=int,  default=200)
    p.add_argument("--test",   type=int,  default=100)
    p.add_argument("--plot",   action="store_true", default=True)
    p.add_argument("--output", type=str,
                   default="/mnt/user-data/outputs/quantail_backtest.png")
    return p.parse_args()


def main():
    args = parse_args()

    backtester = Backtester(env_config={
        "S0": 100.0, "K": 100.0, "T": 0.25,
        "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
        "n_steps": 60, "lam": 0.01,
    })

    print("\nRunning Quantail (train + test)...")
    backtester.run_quantail(n_train=args.train, n_test=args.test)

    print("\nRunning Delta Hedge baseline...")
    backtester.run_delta_hedge(n_episodes=args.test)

    print("\nRunning No Hedge baseline...")
    backtester.run_no_hedge(n_episodes=args.test)

    backtester.compare()

    if args.plot:
        backtester.plot_results(save_path=args.output)


if __name__ == "__main__":
    main()
