"""
train.py
========
Main training script for Quantail.

Usage:
  python train.py
  python train.py --episodes 500 --steps 60
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from envs.heston_env import HestonEnv
from agents.agents import HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent
from coordinator.wasserstein_coordinator import WassersteinCoordinator
from risk.coherent_risk import check_coherence_axioms
from quantum.vqc_policy import VQCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train Quantail agents")
    parser.add_argument("--episodes",  type=int,   default=200)
    parser.add_argument("--steps",     type=int,   default=60)
    parser.add_argument("--S0",        type=float, default=100.0)
    parser.add_argument("--K",         type=float, default=100.0)
    parser.add_argument("--kappa",     type=float, default=2.0)
    parser.add_argument("--theta",     type=float, default=0.04)
    parser.add_argument("--xi",        type=float, default=0.3)
    parser.add_argument("--rho",       type=float, default=-0.7)
    parser.add_argument("--lam",       type=float, default=0.01)
    parser.add_argument("--seed",      type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("  QUANTAIL — Distributional Quantum RL for Hedging")
    print("  Sankalp Jain & Veronica Koval | Columbia University")
    print("=" * 60)

    # ── Environment ───────────────────────────────────────────────────
    env = HestonEnv(
        S0=args.S0, K=args.K, n_steps=args.steps,
        kappa=args.kappa, theta=args.theta,
        xi=args.xi, rho=args.rho, lam=args.lam,
    )
    print(f"\nEnvironment: HestonEnv")
    print(f"  S0={args.S0}, K={args.K}, T=0.25 (3mo)")
    print(f"  kappa={args.kappa}, theta={args.theta}, xi={args.xi}, rho={args.rho}")
    print(f"  Feller: 2*kappa*theta={2*args.kappa*args.theta:.3f} > xi^2={args.xi**2:.3f}")

    # ── VQC expressivity info ─────────────────────────────────────────
    vqc_info = VQCPolicy(n_qubits=8, n_layers=3).expressivity_info()
    print(f"\nVQC Policy (Theorem 7.1 — Schuld et al. 2021):")
    print(f"  Qubits: {vqc_info['n_qubits']}, Layers: {vqc_info['n_layers']}")
    print(f"  VQC Fourier modes:     O(2^n) = {vqc_info['vqc_fourier_modes']}")
    print(f"  Classical MLP modes:   O(poly(n)) ~ {vqc_info['mlp_fourier_modes']}")
    print(f"  Expressivity ratio:    {vqc_info['expressivity_ratio']:.1f}x")

    # ── Coherence axiom check ─────────────────────────────────────────
    print(f"\nCoherence Axiom Verification (Definition 8.1 — Artzner 1999):")
    rng = np.random.RandomState(42)
    X = rng.normal(-0.01, 0.05, 1000)
    Y = rng.normal(-0.02, 0.04, 1000)
    check_coherence_axioms(X, Y, verbose=True)

    # ── Agents ────────────────────────────────────────────────────────
    agents = {
        "HedgingAgent":   HedgingAgent(seed=args.seed),
        "RiskAgent":      RiskAgent(seed=args.seed + 1),
        "PortfolioAgent": PortfolioAgent(seed=args.seed + 2),
        "AlphaAgent":     AlphaAgent(seed=args.seed + 3),
    }

    coordinator = WassersteinCoordinator(
        agent_names=list(agents.keys()),
        weights={"HedgingAgent": 0.40, "RiskAgent": 0.30,
                 "PortfolioAgent": 0.20, "AlphaAgent": 0.10},
    )

    print(f"\nAgents initialized:")
    for name in agents:
        print(f"  - {name} (VQC policy, 8 qubits, 3 layers)")
    print(f"  - WassersteinCoordinator (barycenter consensus)")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nTraining for {args.episodes} episodes...")
    print(f"{'─'*60}")

    ep_returns = []
    ep_errors  = []

    for ep in range(args.episodes):
        epsilon = max(0.05, 0.5 - 0.45 * ep / args.episodes)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_errors = []

        while not done:
            action, coord_info = coordinator.coordinate(obs, agents, epsilon)
            obs_next, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            for name, agent in agents.items():
                r = agent.compute_reward(info)
                agent.store(obs, action, r, obs_next, done)
                ep_reward += r / len(agents)

            step_errors.append(abs(info.get("hedging_error", 0.0)))
            obs = obs_next

        for agent in agents.values():
            agent.end_episode()
            agent.update(batch_size=min(32, len(agent.buffer)))

        ep_returns.append(ep_reward)
        ep_errors.append(np.mean(step_errors) if step_errors else 0.0)

        # Logging
        if (ep + 1) % 50 == 0:
            recent_ret = np.mean(ep_returns[-50:])
            recent_err = np.mean(ep_errors[-50:])
            print(f"  Ep {ep+1:4d}/{args.episodes} | "
                  f"Return: {recent_ret:8.4f} | "
                  f"Avg |error|: {recent_err:.4f} | "
                  f"eps: {epsilon:.3f}")

    # ── Final stats ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — Agent Statistics")
    print(f"{'='*60}")
    for name, agent in agents.items():
        stats = agent.get_stats()
        print(f"\n  {name}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k:<20}: {v:.4f}")

    coord_stats = coordinator.get_coordination_stats()
    print(f"\n  Coordinator:")
    print(f"    Total decisions: {coord_stats.get('total_decisions', 0)}")
    print(f"    Veto rate:       {coord_stats.get('veto_rate', 0):.3f}")
    print(f"    Mean CVaR:       {coord_stats.get('mean_cvar', 0):.4f}")
    print(f"    Weights:         {coord_stats.get('weights', {})}")

    print(f"\nTraining complete. Run 'python run_backtest.py' for full comparison.")


if __name__ == "__main__":
    main()
