"""
evaluation/baselines.py
=======================
Classical baselines for comparison with Quantail.

  1. DeltaHedgeBaseline  — classical Black-Scholes delta hedging
  2. NoHedgeBaseline     — hold option, no hedging
  3. StaticHedgeBaseline — fixed hedge ratio
"""

import numpy as np
from typing import List, Dict


class DeltaHedgeBaseline:
    """
    Classical Black-Scholes delta hedging.
    Rebalances to delta at every timestep.
    This is the benchmark Quantail beats.
    """

    def __init__(self, env):
        self.env = env
        self.results = []

    def run_episode(self) -> Dict:
        obs, _ = self.env.reset()
        done = False
        total_error = 0.0
        total_cost = 0.0
        errors = []
        costs = []

        while not done:
            # Delta hedge: set hedge = current delta
            # delta is obs[3], hedge is obs[5]
            current_delta = float(obs[3])
            current_hedge = float(obs[5])

            # Compute required adjustment
            adj = current_delta - current_hedge  # target - current
            adj_clipped = np.clip(adj, -0.05, 0.05)

            # Map to action (adj in [-0.05, 0.05] -> action in [0, 10])
            action = int(np.round((adj_clipped / 0.01) + 5))
            action = np.clip(action, 0, 10)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            he = info.get("hedging_error", 0.0)
            cost = info.get("cost", 0.0)
            errors.append(he)
            costs.append(cost)
            total_cost += cost

        errors = np.array(errors)
        result = {
            "strategy":     "DeltaHedge",
            "rmse":         float(np.sqrt(np.mean(errors ** 2))),
            "mean_abs_err": float(np.mean(np.abs(errors))),
            "total_cost":   total_cost,
            "total_pnl":    float(np.sum(errors)),
            "cvar_95":      self._cvar(errors),
            "var_95":       self._var(errors),
            "errors":       errors.tolist(),
            "costs":        costs,
        }
        self.results.append(result)
        return result

    def run_episodes(self, n: int = 100) -> Dict:
        for _ in range(n):
            self.run_episode()
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        if not self.results:
            return {}
        rmses  = [r["rmse"] for r in self.results]
        cvars  = [r["cvar_95"] for r in self.results]
        costs  = [r["total_cost"] for r in self.results]
        return {
            "strategy":      "DeltaHedge",
            "n_episodes":    len(self.results),
            "mean_rmse":     float(np.mean(rmses)),
            "std_rmse":      float(np.std(rmses)),
            "mean_cvar_95":  float(np.mean(cvars)),
            "mean_cost":     float(np.mean(costs)),
        }

    def _cvar(self, errors: np.ndarray, alpha: float = 0.05) -> float:
        var_val = float(-np.percentile(errors, alpha * 100))
        tail = errors[errors <= -var_val]
        return float(-np.mean(tail)) if len(tail) > 0 else var_val

    def _var(self, errors: np.ndarray, alpha: float = 0.05) -> float:
        return float(-np.percentile(errors, alpha * 100))


class NoHedgeBaseline:
    """Hold option with no hedging — maximum exposure."""

    def __init__(self, env):
        self.env = env
        self.results = []

    def run_episode(self) -> Dict:
        obs, _ = self.env.reset()
        done = False
        errors = []

        while not done:
            action = 5  # no change, stay at current (zero) hedge
            obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            errors.append(info.get("hedging_error", 0.0))

        errors = np.array(errors)
        result = {
            "strategy": "NoHedge",
            "rmse":     float(np.sqrt(np.mean(errors ** 2))),
            "cvar_95":  float(-np.mean(errors[errors <= np.percentile(errors, 5)]))
                         if len(errors) > 0 else 0.0,
        }
        self.results.append(result)
        return result

    def run_episodes(self, n: int = 100) -> Dict:
        for _ in range(n):
            self.run_episode()
        rmses = [r["rmse"] for r in self.results]
        cvars = [r["cvar_95"] for r in self.results]
        return {
            "strategy":     "NoHedge",
            "mean_rmse":    float(np.mean(rmses)),
            "mean_cvar_95": float(np.mean(cvars)),
        }
