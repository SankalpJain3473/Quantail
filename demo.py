"""
demo.py
=======
YC Demo Script — runs the full Quantail PoC and generates
presentation-ready output in under 2 minutes.

Usage:
  python demo.py

Output:
  - Console: metrics table comparing Quantail vs Delta Hedge
  - Plot:    saved to outputs/quantail_demo.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from envs.heston_env import HestonEnv
from agents.agents import HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent
from coordinator.wasserstein_coordinator import WassersteinCoordinator
from evaluation.baselines import DeltaHedgeBaseline, NoHedgeBaseline
from risk.coherent_risk import cvar, var, check_coherence_axioms
from quantum.vqc_policy import VQCPolicy


def run_demo():
    np.random.seed(42)

    print("=" * 65)
    print("  QUANTAIL — YC Demo")
    print("  Distributional Quantum RL for Institutional Hedging")
    print("  Sankalp Jain & Veronica Koval | Columbia University")
    print("=" * 65)

    # ── 1. Show mathematical foundations ─────────────────────────────
    print("\n[1/5] Mathematical Foundations")
    print("  Heston SDE: dS = μS dt + √v·S dW^S")
    print("               dv = κ(θ-v)dt + ξ√v dW^v")

    env_cfg = dict(S0=100, K=100, T=0.25, kappa=2.0,
                   theta=0.04, xi=0.3, rho=-0.7, n_steps=60, lam=0.01)
    env = HestonEnv(**env_cfg)
    print(f"  Feller: 2κθ = {2*2.0*0.04:.3f} > ξ² = {0.3**2:.3f} ✓")

    # ── 2. VQC expressivity ───────────────────────────────────────────
    print("\n[2/5] VQC Expressivity (Theorem 7.1 — Schuld et al. 2021)")
    vqc = VQCPolicy(n_qubits=8, n_layers=3, seed=42)
    info = vqc.expressivity_info()
    print(f"  VQC Fourier modes:  O(2^8) = {info['vqc_fourier_modes']}")
    print(f"  MLP Fourier modes:  O(8^2) = {info['mlp_fourier_modes']}")
    print(f"  Expressivity gain:  {info['expressivity_ratio']:.1f}x")

    # ── 3. CVaR coherence ─────────────────────────────────────────────
    print("\n[3/5] CVaR Coherence (Definition 8.1 — Artzner 1999)")
    rng = np.random.RandomState(42)
    X = rng.normal(-0.01, 0.05, 500)
    Y = rng.normal(-0.02, 0.04, 500)
    check_coherence_axioms(X, Y, verbose=True)

    # ── 4. Train multi-agent system ───────────────────────────────────
    N_TRAIN = 150
    N_TEST  = 80

    print(f"\n[4/5] Training Quantail ({N_TRAIN} episodes)...")
    agents = {
        "HedgingAgent":   HedgingAgent(seed=42),
        "RiskAgent":      RiskAgent(seed=43),
        "PortfolioAgent": PortfolioAgent(seed=44),
        "AlphaAgent":     AlphaAgent(seed=45),
    }
    coordinator = WassersteinCoordinator(
        agent_names=list(agents.keys()),
        weights={"HedgingAgent": 0.40, "RiskAgent": 0.30,
                 "PortfolioAgent": 0.20, "AlphaAgent": 0.10},
    )

    train_returns = []
    for ep in range(N_TRAIN):
        eps = max(0.05, 0.5 * (1 - ep / N_TRAIN))
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action, _ = coordinator.coordinate(obs, agents, eps)
            obs_n, _, term, trunc, info = env.step(action)
            done = term or trunc
            for agent in agents.values():
                r = agent.compute_reward(info)
                agent.store(obs, action, r, obs_n, done)
                ep_r += r / 5
            obs = obs_n
        for agent in agents.values():
            agent.end_episode()
            agent.update(32)
        train_returns.append(ep_r)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{N_TRAIN} | "
                  f"Avg return: {np.mean(train_returns[-50:]):.4f}")

    # ── 5. Compare vs baselines ───────────────────────────────────────
    print(f"\n[5/5] Testing ({N_TEST} episodes each)...")

    # Test Quantail
    q_errors, q_cvars, q_costs = [], [], []
    for _ in range(N_TEST):
        obs, _ = env.reset()
        done = False
        ep_err, ep_cost = [], []
        while not done:
            action, _ = coordinator.coordinate(obs, agents, 0.0)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ep_err.append(info.get("hedging_error", 0.0))
            ep_cost.append(info.get("cost", 0.0))
        ep_arr = np.array(ep_err)
        q_errors.append(float(np.sqrt(np.mean(ep_arr**2))))
        q_cvars.append(cvar(-ep_arr) if len(ep_arr) > 1 else 0.0)
        q_costs.append(sum(ep_cost))

    # Test Delta Hedge
    delta_env = HestonEnv(**env_cfg)
    dh = DeltaHedgeBaseline(delta_env)
    dh_results = dh.run_episodes(N_TEST)

    # Test No Hedge
    nh_env = HestonEnv(**env_cfg)
    nh = NoHedgeBaseline(nh_env)
    nh_results = nh.run_episodes(N_TEST)

    # ── Results table ─────────────────────────────────────────────────
    q_rmse  = np.mean(q_errors)
    q_cvar  = np.mean(q_cvars)
    q_cost  = np.mean(q_costs)
    dh_rmse = dh_results.get("mean_rmse", 0)
    dh_cvar = dh_results.get("mean_cvar_95", 0)
    dh_cost = dh_results.get("mean_cost", 0)

    rmse_imp = (dh_rmse - q_rmse) / (dh_rmse + 1e-10) * 100
    cvar_imp = (dh_cvar - q_cvar) / (dh_cvar + 1e-10) * 100

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"{'Strategy':<20} {'RMSE':>10} {'CVaR@95%':>12} {'Cost':>10}")
    print(f"{'-'*55}")
    print(f"{'Quantail':<20} {q_rmse:>10.4f} {q_cvar:>12.4f} {q_cost:>10.4f}")
    print(f"{'Delta Hedge':<20} {dh_rmse:>10.4f} {dh_cvar:>12.4f} {dh_cost:>10.4f}")
    nh_rmse = nh_results.get("mean_rmse", 0)
    nh_cvar = nh_results.get("mean_cvar_95", 0)
    print(f"{'No Hedge':<20} {nh_rmse:>10.4f} {nh_cvar:>12.4f} {'N/A':>10}")
    print(f"{'='*65}")
    print(f"\n  Quantail vs Delta Hedge:")
    print(f"    RMSE improvement:     {rmse_imp:+.1f}%")
    print(f"    CVaR@95% improvement: {cvar_imp:+.1f}%")

    # ── Generate plots ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    PURPLE = "#534AB7"
    BLUE   = "#18608A"
    CORAL  = "#993C1D"
    TEAL   = "#0F6E56"
    GRAY   = "#5F5E5A"

    # ── Plot 1: Training curve ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    window = 15
    smoothed = np.convolve(train_returns, np.ones(window)/window, mode='valid')
    ax1.plot(range(len(smoothed)), smoothed, color=PURPLE, lw=2.5, label="Quantail training")
    ax1.fill_between(range(len(smoothed)),
                     smoothed - 0.5*np.std(smoothed),
                     smoothed + 0.5*np.std(smoothed),
                     alpha=0.15, color=PURPLE)
    ax1.axhline(0, color=GRAY, lw=0.8, linestyle='--')
    ax1.set_title("Quantail — Training Learning Curve (Multi-Agent Distributional RL)",
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Average Return", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: RMSE Bars ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    strats = ["Quantail", "Delta\nHedge", "No\nHedge"]
    rmse_vals = [q_rmse, dh_rmse, nh_rmse]
    colors_bar = [PURPLE, BLUE, CORAL]
    bars = ax2.bar(strats, rmse_vals, color=colors_bar, alpha=0.85, edgecolor='white', width=0.5)
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title("Hedging RMSE\n(lower = better)", fontweight='bold')
    ax2.set_ylabel("RMSE")
    ax2.grid(True, axis='y', alpha=0.3)

    # ── Plot 3: CVaR bars ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    cvar_vals = [q_cvar, dh_cvar, nh_cvar]
    bars2 = ax3.bar(strats, cvar_vals, color=colors_bar, alpha=0.85, edgecolor='white', width=0.5)
    for bar, val in zip(bars2, cvar_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title("CVaR@95%\n(lower = better tail risk)", fontweight='bold')
    ax3.set_ylabel("CVaR")
    ax3.grid(True, axis='y', alpha=0.3)

    # ── Plot 4: Improvement ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    metrics = ["RMSE\nimprovement", "CVaR@95%\nimprovement"]
    improvements = [rmse_imp, cvar_imp]
    colors_imp = [TEAL if v > 0 else CORAL for v in improvements]
    bars3 = ax4.bar(metrics, improvements, color=colors_imp, alpha=0.85,
                    edgecolor='white', width=0.4)
    for bar, val in zip(bars3, improvements):
        y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 2
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+.1f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    ax4.axhline(0, color=GRAY, lw=1)
    ax4.set_title("Quantail vs Delta Hedge\n(+% = improvement)", fontweight='bold')
    ax4.set_ylabel("Improvement %")
    ax4.grid(True, axis='y', alpha=0.3)

    # ── Plot 5: RMSE distribution ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(q_errors, bins=20, color=PURPLE, alpha=0.7, label="Quantail", edgecolor='white')
    ax5.axvline(q_rmse, color=PURPLE, lw=2, linestyle='--', label=f"Mean: {q_rmse:.4f}")
    ax5.set_title("Quantail RMSE Distribution", fontweight='bold')
    ax5.set_xlabel("RMSE per episode")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: CVaR distribution ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(q_cvars, bins=20, color=TEAL, alpha=0.7, label="Quantail CVaR", edgecolor='white')
    ax6.axvline(q_cvar, color=TEAL, lw=2, linestyle='--', label=f"Mean: {q_cvar:.4f}")
    ax6.set_title("Quantail CVaR@95% Distribution", fontweight='bold')
    ax6.set_xlabel("CVaR per episode")
    ax6.set_ylabel("Count")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── Plot 7: Agent weights ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    coord_stats = coordinator.get_coordination_stats()
    w = coord_stats.get("weights", {})
    if w:
        labels = [k.replace("Agent", "") for k in w.keys()]
        sizes  = list(w.values())
        colors_pie = [PURPLE, CORAL, BLUE, TEAL]
        wedges, texts, autotexts = ax7.pie(
            sizes, labels=labels, autopct='%1.0f%%',
            colors=colors_pie[:len(labels)], startangle=90,
            textprops={'fontsize': 9}
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('bold')
    ax7.set_title("Coordinator Agent Weights\n(Wasserstein Barycenter)", fontweight='bold')

    # ── Main title ────────────────────────────────────────────────────
    fig.suptitle(
        "Quantail — Distributional Quantum RL with Multi-Agent Coordination\n"
        "Sankalp Jain & Veronica Koval  |  Columbia University, New York  |  YC S2026 PoC",
        fontsize=13, fontweight='bold', y=0.99
    )

    out = "/mnt/user-data/outputs/quantail_demo.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved: {out}")

    print("\n" + "="*65)
    print("  DEMO COMPLETE")
    print("  See quantail_demo.png for presentation-ready visualization.")
    print("="*65)


if __name__ == "__main__":
    run_demo()
