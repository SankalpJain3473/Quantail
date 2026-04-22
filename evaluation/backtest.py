"""
evaluation/backtest.py
======================
Full backtest engine comparing Quantail vs baselines.

Metrics:
  - Hedging RMSE
  - CVaR@95%
  - Sharpe ratio
  - Sortino ratio
  - Transaction cost ratio
  - Win rate vs delta hedge
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.heston_env import HestonEnv
from envs.hybrid_env import HybridEnv
from agents.agents import HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent
from coordinator.wasserstein_coordinator import WassersteinCoordinator
from evaluation.baselines import DeltaHedgeBaseline, NoHedgeBaseline
from risk.coherent_risk import cvar, var


class Backtester:
    """
    Full backtest engine for Quantail vs baselines.
    """

    def __init__(self, env_config: Optional[Dict] = None):
        self.env_config = env_config or {}
        self.results = {}

    # ─────────────────────────────────────────────────────────────────────
    def run_quantail(
        self,
        n_train: int = 200,
        n_test: int = 100,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        verbose: bool = True,
    ) -> Dict:
        """Train and test the full Quantail multi-agent system."""

        env = HybridEnv(**{k:v for k,v in self.env_config.items() if k not in ["kappa","theta","xi","rho"]})

        # Initialize agents
        agents = {
            "HedgingAgent":        HedgingAgent(seed=42, n_qubits=13),
            "RiskAgent":           RiskAgent(seed=43, n_qubits=13),
            "QuantumExplorerAgent":QuantumExplorerAgent(seed=44, n_qubits=13),
            "RegimeAgent":         RegimeAgent(seed=45, n_qubits=13),
            "AlphaAgent":          AlphaAgent(seed=46, n_qubits=13),
        }

        coordinator = WassersteinCoordinator(
            agent_names=list(agents.keys()),
            weights={"HedgingAgent": 0.40, "RiskAgent": 0.25,
                     "QuantumExplorerAgent": 0.15, "RegimeAgent": 0.10, "AlphaAgent": 0.10},
        )

        # ── Training phase ────────────────────────────────────────────
        if verbose:
            print(f"\n{'='*60}")
            print(f"  QUANTAIL TRAINING  ({n_train} episodes)")
            print(f"{'='*60}")

        train_returns = []
        for ep in range(n_train):
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * ep / n_train,
            )
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                # Coordinator selects action
                action, coord_info = coordinator.coordinate(obs, agents, epsilon)

                # Environment step
                obs_next, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Each agent computes its own reward and stores experience
                for name, agent in agents.items():
                    agent_reward = agent.compute_reward(info)
                    agent.store(obs, action, agent_reward, obs_next, done)
                    ep_reward += agent_reward / len(agents)

                obs = obs_next

            # End episode for all agents
            for agent in agents.values():
                agent.end_episode()

            # Update policies
            for agent in agents.values():
                agent.update(batch_size=min(32, len(agent.buffer)))

            # Update coordinator weights based on performance
            performances = {
                name: agent.get_stats().get("mean_return", 0.0)
                for name, agent in agents.items()
            }
            coordinator.update_weights(performances)

            train_returns.append(ep_reward)

            if verbose and (ep + 1) % 50 == 0:
                recent = np.mean(train_returns[-50:])
                print(f"  Episode {ep+1:4d}/{n_train} | "
                      f"Avg return: {recent:8.4f} | "
                      f"Epsilon: {epsilon:.3f}")

        # ── Testing phase ──────────────────────────────────────────────
        if verbose:
            print(f"\n{'='*60}")
            print(f"  QUANTAIL TESTING   ({n_test} episodes)")
            print(f"{'='*60}")

        test_errors   = []
        test_costs    = []
        test_returns  = []
        test_cvars    = []
        episode_errors = []

        for ep in range(n_test):
            obs, _ = env.reset()
            done = False
            ep_errors = []
            ep_costs  = []

            while not done:
                action, _ = coordinator.coordinate(obs, agents, epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_errors.append(info.get("hedging_error", 0.0))
                ep_costs.append(info.get("cost", 0.0))

            ep_errors = np.array(ep_errors)
            ep_rmse = float(np.sqrt(np.mean(ep_errors ** 2)))
            ep_cvar = cvar(-ep_errors) if len(ep_errors) > 1 else 0.0

            test_errors.append(ep_rmse)
            test_costs.append(sum(ep_costs))
            test_cvars.append(ep_cvar)
            episode_errors.append(ep_errors)

        all_errors = np.concatenate(episode_errors)
        all_returns = -np.array(test_errors)

        result = {
            "strategy":     "Quantail",
            "n_test":       n_test,
            "mean_rmse":    float(np.mean(test_errors)),
            "std_rmse":     float(np.std(test_errors)),
            "mean_cvar_95": float(np.mean(test_cvars)),
            "mean_cost":    float(np.mean(test_costs)),
            "sharpe":       float(np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)),
            "sortino":      self._sortino(all_returns),
            "episode_rmses": test_errors,
            "episode_cvars": test_cvars,
            "train_returns": train_returns,
            "agents_stats":  {name: agent.get_stats() for name, agent in agents.items()},
            "coord_stats":   coordinator.get_coordination_stats(),
        }

        self.results["Quantail"] = result

        if verbose:
            self._print_results(result)

        return result

    # ─────────────────────────────────────────────────────────────────────
    def run_delta_hedge(self, n_episodes: int = 100, verbose: bool = True) -> Dict:
        """Run classical delta hedge baseline."""
        env = HybridEnv(**{k:v for k,v in self.env_config.items() if k not in ["kappa","theta","xi","rho"]})
        baseline = DeltaHedgeBaseline(env)
        result = baseline.run_episodes(n_episodes)
        self.results["DeltaHedge"] = result
        if verbose:
            self._print_results(result)
        return result

    def run_no_hedge(self, n_episodes: int = 100, verbose: bool = True) -> Dict:
        """Run no-hedge baseline."""
        env = HybridEnv(**{k:v for k,v in self.env_config.items() if k not in ["kappa","theta","xi","rho"]})
        baseline = NoHedgeBaseline(env)
        result = baseline.run_episodes(n_episodes)
        self.results["NoHedge"] = result
        return result

    # ─────────────────────────────────────────────────────────────────────
    def compare(self) -> Dict:
        """Compare all strategies and print summary table."""
        if not self.results:
            print("No results yet. Run strategies first.")
            return {}

        print(f"\n{'='*70}")
        print(f"  BACKTEST COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Strategy':<20} {'RMSE':>10} {'CVaR@95%':>12} {'Sharpe':>10} {'Cost':>10}")
        print(f"{'-'*70}")

        for name, res in self.results.items():
            rmse  = res.get("mean_rmse",    res.get("rmse", 0))
            cv    = res.get("mean_cvar_95", res.get("cvar_95", 0))
            sharpe= res.get("sharpe", 0)
            cost  = res.get("mean_cost", res.get("total_cost", 0))
            print(f"{name:<20} {rmse:>10.4f} {cv:>12.4f} {sharpe:>10.4f} {cost:>10.4f}")

        print(f"{'='*70}")

        # Improvement of Quantail over DeltaHedge
        if "Quantail" in self.results and "DeltaHedge" in self.results:
            q = self.results["Quantail"]
            d = self.results["DeltaHedge"]
            rmse_imp = (d["mean_rmse"] - q["mean_rmse"]) / d["mean_rmse"] * 100
            cvar_imp = (d["mean_cvar_95"] - q["mean_cvar_95"]) / d["mean_cvar_95"] * 100
            print(f"\n  Quantail vs Delta Hedge:")
            print(f"    RMSE improvement:    {rmse_imp:+.1f}%")
            print(f"    CVaR@95% improvement:{cvar_imp:+.1f}%")

        return self.results

    # ─────────────────────────────────────────────────────────────────────
    def plot_results(self, save_path: str = "/mnt/user-data/outputs/quantail_backtest.png"):
        """Generate comparison plots."""
        if not self.results:
            return

        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        colors = {
            "Quantail":  "#534AB7",
            "DeltaHedge": "#18608A",
            "NoHedge":   "#993C1D",
        }

        # ── Plot 1: Training curve ────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        if "Quantail" in self.results:
            train_r = self.results["Quantail"].get("train_returns", [])
            if train_r:
                window = 20
                smoothed = np.convolve(train_r, np.ones(window)/window, mode='valid')
                ax1.plot(range(len(smoothed)), smoothed,
                         color="#534AB7", lw=2, label="Quantail (smoothed)")
                ax1.fill_between(range(len(smoothed)),
                                 smoothed - np.std(smoothed),
                                 smoothed + np.std(smoothed),
                                 alpha=0.15, color="#534AB7")
        ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax1.set_title("Training Learning Curve", fontweight='bold', fontsize=12)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: RMSE comparison ───────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        strategies = []
        rmses = []
        cols = []
        for name, res in self.results.items():
            if "mean_rmse" in res or "rmse" in res:
                strategies.append(name)
                rmses.append(res.get("mean_rmse", res.get("rmse", 0)))
                cols.append(colors.get(name, "gray"))
        bars = ax2.bar(strategies, rmses, color=cols, alpha=0.8, edgecolor='white')
        ax2.set_title("Hedging RMSE\n(lower is better)", fontweight='bold')
        ax2.set_ylabel("RMSE")
        for bar, val in zip(bars, rmses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        # ── Plot 3: CVaR comparison ───────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        strategies_c = []
        cvars_v = []
        cols_c = []
        for name, res in self.results.items():
            cv = res.get("mean_cvar_95", res.get("cvar_95", 0))
            if cv > 0:
                strategies_c.append(name)
                cvars_v.append(cv)
                cols_c.append(colors.get(name, "gray"))
        bars2 = ax3.bar(strategies_c, cvars_v, color=cols_c, alpha=0.8, edgecolor='white')
        ax3.set_title("CVaR@95%\n(lower = better tail risk)", fontweight='bold')
        ax3.set_ylabel("CVaR")
        for bar, val in zip(bars2, cvars_v):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        # ── Plot 4: Sharpe comparison ─────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        sharpes = []
        strat_s = []
        cols_s = []
        for name, res in self.results.items():
            s = res.get("sharpe", 0)
            if s != 0:
                strat_s.append(name)
                sharpes.append(s)
                cols_s.append(colors.get(name, "gray"))
        if strat_s:
            bars3 = ax4.bar(strat_s, sharpes, color=cols_s, alpha=0.8, edgecolor='white')
            ax4.set_title("Sharpe Ratio\n(higher is better)", fontweight='bold')
            ax4.set_ylabel("Sharpe")
            ax4.axhline(0, color="gray", lw=0.8)

        # ── Plot 5: RMSE distribution ─────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 0:2])
        for name, res in self.results.items():
            rmse_list = res.get("episode_rmses", [])
            if rmse_list:
                ax5.hist(rmse_list, bins=20, alpha=0.6,
                         color=colors.get(name, "gray"),
                         label=name, edgecolor='white')
        ax5.set_title("RMSE Distribution Across Episodes", fontweight='bold')
        ax5.set_xlabel("RMSE")
        ax5.set_ylabel("Count")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # ── Plot 6: CVaR distribution ─────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 2])
        for name, res in self.results.items():
            cvar_list = res.get("episode_cvars", [])
            if cvar_list:
                ax6.hist(cvar_list, bins=20, alpha=0.6,
                         color=colors.get(name, "gray"),
                         label=name, edgecolor='white')
        ax6.set_title("CVaR@95% Distribution", fontweight='bold')
        ax6.set_xlabel("CVaR")
        ax6.set_ylabel("Count")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Title
        fig.suptitle(
            "Quantail — Distributional Quantum RL vs Classical Baselines\n"
            "Sankalp Jain & Veronica Koval | Columbia University",
            fontsize=13, fontweight='bold', y=0.98
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved to: {save_path}")
        return save_path

    # ─────────────────────────────────────────────────────────────────────
    def _print_results(self, result: Dict):
        name = result.get("strategy", "Unknown")
        print(f"\n  {name}:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"    {k:<20}: {v:.4f}")
            elif isinstance(v, int):
                print(f"    {k:<20}: {v}")

    def _sortino(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float(np.mean(returns) / 1e-8 * np.sqrt(252))
        return float(np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(252))
