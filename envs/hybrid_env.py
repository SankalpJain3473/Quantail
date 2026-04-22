"""
envs/hybrid_env.py
==================
Heston-Bates-Hamilton Hybrid Environment

Combines three models into one:
  1. Heston SDE         — stochastic volatility, leverage effect
  2. Bates (1996)       — Poisson price jumps (flash crashes, earnings)
  3. Hamilton RS        — Markov regime switching (calm/stressed/crash)

Each regime has its own calibrated parameter set. The system
transitions between regimes via a hidden Markov chain — agents
never see the regime directly, only its effects on prices.

This fixes all 5 structural gaps in the original Heston:
  ✓ Price jumps / flash crash       (Bates component)
  ✓ Regime transitions mid-session  (Hamilton component)
  ✓ Fat tails / excess kurtosis     (jumps add kurtosis)
  ✓ Correlation breakdown in crisis (rho differs per regime)
  ✓ Vol clustering                  (regime-specific kappa/xi)

Observation (13-dim — extended from 10):
  [0]  moneyness         S/K
  [1]  variance          v_t
  [2]  tau               time to expiry
  [3]  delta             BS delta
  [4]  gamma             scaled gamma
  [5]  hedge             current hedge ratio
  [6]  pnl               normalized P&L
  [7]  cost_frac         cost used
  [8]  realized_vol      20-step rolling vol
  [9]  drawdown          from peak
  [10] jump_intensity    recent jump frequency  ← NEW
  [11] regime_vol        current vol regime     ← NEW
  [12] time_in_regime    steps in current regime← NEW

VQC gains: 2^13 = 8192 Fourier modes (vs 1024 previously)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
from typing import Tuple, Optional


# ── Regime definitions ──────────────────────────────────────────────────────
REGIMES = {
    0: "calm",        # low vol, trending
    1: "stressed",    # elevated vol, mean-reverting
    2: "crisis",      # high vol, spike, jumps frequent
    3: "recovery",    # post-crisis, vol falling
}

# Per-regime Heston + Bates parameters
# (kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)
REGIME_PARAMS = {
    0: dict(kappa=3.0,  theta=0.04,  xi=0.25, rho=-0.6,  lam_j=0.2,  mu_j=-0.01, sig_j=0.02),  # calm
    1: dict(kappa=2.0,  theta=0.08,  xi=0.40, rho=-0.7,  lam_j=0.5,  mu_j=-0.02, sig_j=0.04),  # stressed
    2: dict(kappa=1.5,  theta=0.25,  xi=0.60, rho=-0.85, lam_j=2.0,  mu_j=-0.05, sig_j=0.08),  # crisis
    3: dict(kappa=4.0,  theta=0.06,  xi=0.30, rho=-0.65, lam_j=0.3,  mu_j=-0.01, sig_j=0.03),  # recovery
}

# Markov transition matrix (rows = from, cols = to)
# High probability of staying, low of switching
TRANSITION_MATRIX = np.array([
    [0.97, 0.02, 0.005, 0.005],   # calm     → mostly stays calm
    [0.10, 0.80, 0.08,  0.02 ],   # stressed → can go calm or crisis
    [0.05, 0.20, 0.65,  0.10 ],   # crisis   → can move to recovery/stressed
    [0.15, 0.10, 0.05,  0.70 ],   # recovery → often back to calm/stressed
])


class HybridEnv(gym.Env):
    """
    Heston-Bates-Hamilton Hybrid Gymnasium Environment.

    Handles every edge case the original Heston misses:
      - Jumps from Bates component
      - Regime transitions from Hamilton component
      - Fat tails from compound effect of jumps + regimes
      - Correlation breakdown: each regime has its own rho
      - Vol clustering: regime switches cause vol bursts

    Backward compatible with the 5-agent system.
    Drop-in replacement for HestonEnv.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0: float = 100.0,
        K:  float = 100.0,
        T:  float = 0.25,
        r:  float = 0.05,
        n_steps: int = 60,
        lam: float = 0.005,         # transaction cost weight
        spread: float = 0.001,
        cost_budget: float = 5.0,
        initial_regime: int = 0,    # start in calm regime
        regime_aware: bool = True,  # expose regime features in obs
        stress_test: Optional[str] = None,  # force a scenario
    ):
        super().__init__()

        self.S0 = S0
        self.K  = K
        self.T  = T
        self.r  = r
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.lam = lam
        self.spread = spread
        self.cost_budget = cost_budget
        self.initial_regime = initial_regime
        self.regime_aware = regime_aware
        self.stress_test = stress_test

        # 11 discrete actions
        self.action_space = spaces.Discrete(11)
        # 13-dim observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # State (initialized in reset)
        self.S = None
        self.v = None
        self.hedge = None
        self.step_idx = None
        self.pnl = None
        self.total_cost = None
        self.V = None
        self.delta = None
        self.regime = None

        # Tracking
        self._price_history = []
        self._peak_value = 1.0
        self._jump_count = 0
        self._jump_window = []
        self._time_in_regime = 0
        self.episode_history = []

    # ──────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Handle stress test scenarios
        if self.stress_test == "flash_crash":
            self.initial_regime = 2  # start in crisis
        elif self.stress_test == "vol_spike":
            self.initial_regime = 1
        elif self.stress_test == "recovery":
            self.initial_regime = 3

        self.S = self.S0
        self.regime = self.initial_regime
        params = REGIME_PARAMS[self.regime]
        self.v = params['theta']   # start at regime long-run vol

        self.hedge = 0.0
        self.step_idx = 0
        self.pnl = 0.0
        self.total_cost = 0.0
        self._price_history = [self.S]
        self._peak_value = 1.0
        self._jump_count = 0
        self._jump_window = []
        self._time_in_regime = 0
        self.episode_history = []

        tau = self.T
        self.V = self._option_price(self.S, self.v, tau)
        self.delta = self._bs_delta(self.S, self.v, tau)

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action)

        adj = (action - 5) * 0.01
        new_hedge = float(np.clip(self.hedge + adj, -1.0, 1.0))
        delta_h = abs(new_hedge - self.hedge)
        cost = self.spread * delta_h * self.S
        self.hedge = new_hedge
        self.total_cost += cost

        S_old = self.S
        V_old = self.V

        # ── Step 1: Regime transition ─────────────────────────────────
        trans_probs = TRANSITION_MATRIX[self.regime]
        new_regime = int(np.random.choice(4, p=trans_probs))
        if new_regime != self.regime:
            self._time_in_regime = 0
            self.regime = new_regime
        else:
            self._time_in_regime += 1

        params = REGIME_PARAMS[self.regime]

        # ── Step 2: Heston SDE with regime params ────────────────────
        dt = self.dt
        vp = max(self.v, 0)
        z1, z2 = np.random.randn(), np.random.randn()
        dW_S = z1
        dW_v = params['rho'] * z1 + np.sqrt(max(1 - params['rho']**2, 0)) * z2

        # Log-Euler for price (continuous part)
        log_S = np.log(self.S) + (self.r - 0.5 * vp) * dt + np.sqrt(vp * dt) * dW_S

        # Full-truncation Euler for variance
        self.v = max(
            self.v + params['kappa'] * (params['theta'] - vp) * dt
            + params['xi'] * np.sqrt(vp * dt) * dW_v,
            0.0
        )

        # ── Step 3: Jump component (Bates) ───────────────────────────
        lam_dt = params['lam_j'] * dt
        n_jumps = np.random.poisson(lam_dt)
        jump_factor = 0.0

        if n_jumps > 0:
            for _ in range(n_jumps):
                y = np.random.normal(params['mu_j'], params['sig_j'])
                jump_factor += y
            self._jump_count += n_jumps
            self._jump_window.append(n_jumps)
        else:
            self._jump_window.append(0)

        if len(self._jump_window) > 20:
            self._jump_window.pop(0)

        # Apply jump to log-price
        log_S += jump_factor
        self.S = float(np.exp(log_S))
        self.S = np.clip(self.S, 1.0, 10000.0)

        # ── Step 4: Option valuation and hedging error ────────────────
        self.step_idx += 1
        tau = max(self.T - self.step_idx * self.dt, 1e-8)
        self.V = self._option_price(self.S, self.v, tau)
        self.delta = self._bs_delta(self.S, self.v, tau)
        gamma = self._bs_gamma(self.S, self.v, tau)

        dV = self.V - V_old
        dS = self.S - S_old
        hedging_error = dV - self.hedge * dS
        reward = -(hedging_error ** 2) - self.lam * cost
        self.pnl += reward

        # Track price history
        self._price_history.append(self.S)
        if len(self._price_history) > 22:
            self._price_history.pop(0)

        # Drawdown
        pv = 1.0 + self.pnl / (self.K * 100 + 1e-8)
        if pv > self._peak_value:
            self._peak_value = pv

        self.episode_history.append({
            "step": self.step_idx,
            "S": self.S, "v": self.v,
            "regime": self.regime,
            "regime_name": REGIMES[self.regime],
            "n_jumps": n_jumps,
            "hedge": self.hedge,
            "hedging_error": hedging_error,
            "cost": cost, "reward": reward,
            "pnl": self.pnl,
        })

        terminated = self.step_idx >= self.n_steps
        truncated  = self.total_cost > self.cost_budget

        sigma = np.sqrt(max(self.v, 1e-8))
        info = {
            "hedging_error": hedging_error,
            "cost": cost,
            "total_cost": self.total_cost,
            "S": self.S,
            "v": self.v,
            "delta": self.delta,
            "gamma": gamma,
            "tau": tau,
            "iv": sigma,
            "spread": self.spread,
            "regime": self.regime,
            "regime_name": REGIMES[self.regime],
            "n_jumps": n_jumps,
            "jump_count": self._jump_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        tau   = max(self.T - self.step_idx * self.dt, 1e-8)
        gamma = self._bs_gamma(self.S, self.v, tau)
        sigma = np.sqrt(max(self.v, 1e-8))

        # Realized vol
        if len(self._price_history) >= 2:
            lr  = np.diff(np.log(np.maximum(self._price_history, 1e-8)))
            realized_vol = float(np.std(lr) * np.sqrt(252))
        else:
            realized_vol = sigma

        # Drawdown
        pv = 1.0 + self.pnl / (self.K * 100 + 1e-8)
        drawdown = float(np.clip(
            (self._peak_value - pv) / (self._peak_value + 1e-8), 0, 1
        ))

        # Jump intensity estimate (recent jump frequency)
        jump_intensity = float(np.mean(self._jump_window)) if self._jump_window else 0.0

        # Regime vol signal (normalized)
        regime_vol = float(np.clip(np.sqrt(self.v) / 0.5, 0, 1))

        # Time in current regime (normalized)
        time_in_regime = float(np.clip(self._time_in_regime / 20.0, 0, 1))

        return np.array([
            self.S / self.K,                                          # [0]  moneyness
            float(np.clip(self.v, 0, 1)),                            # [1]  variance
            tau / self.T,                                             # [2]  time
            float(np.clip(self.delta, 0, 1)),                        # [3]  delta
            float(np.clip(gamma * self.S, 0, 5)),                    # [4]  gamma
            float(np.clip(self.hedge, -1, 1)),                       # [5]  hedge
            float(np.clip(self.pnl / 100, -5, 5)),                   # [6]  pnl
            float(np.clip(self.total_cost / self.cost_budget, 0, 1)),# [7]  cost
            float(np.clip(realized_vol, 0, 2)),                      # [8]  realized vol
            float(drawdown),                                          # [9]  drawdown
            float(np.clip(jump_intensity * 10, 0, 1)),               # [10] jump intensity
            float(regime_vol),                                        # [11] regime vol
            float(time_in_regime),                                    # [12] time in regime
        ], dtype=np.float32)

    # ── Black-Scholes helpers ─────────────────────────────────────────────
    def _option_price(self, S, v, tau):
        if tau < 1e-8:
            return max(S - self.K, 0.0)
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return float(max(S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2), 0))

    def _bs_delta(self, S, v, tau):
        if tau < 1e-8:
            return 1.0 if S > self.K else 0.0
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        return float(norm.cdf(d1))

    def _bs_gamma(self, S, v, tau):
        if tau < 1e-8:
            return 0.0
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(tau)))

    def render(self, mode="human"):
        tau = max(self.T - self.step_idx * self.dt, 0)
        print(
            f"Step {self.step_idx:3d} | S={self.S:7.2f} | "
            f"v={self.v:.4f} | regime={REGIMES[self.regime]:<10} | "
            f"hedge={self.hedge:.3f} | pnl={self.pnl:.4f}"
        )


# ── Stress-test scenarios ─────────────────────────────────────────────────
class FlashCrashEnv(HybridEnv):
    """Forces a flash crash mid-episode — tests tail risk handling."""
    def __init__(self, crash_at_step: int = 20, **kwargs):
        super().__init__(stress_test="flash_crash", **kwargs)
        self.crash_at_step = crash_at_step

    def step(self, action):
        # Force a jump at the crash step
        if self.step_idx == self.crash_at_step - 1:
            self.S *= 0.92   # 8% instant drop
            self._jump_count += 1
        return super().step(action)


class VolSpikeEnv(HybridEnv):
    """Forces a volatility spike — tests regime-switching handling."""
    def __init__(self, spike_at_step: int = 15, **kwargs):
        super().__init__(stress_test="vol_spike", **kwargs)
        self.spike_at_step = spike_at_step

    def step(self, action):
        if self.step_idx == self.spike_at_step - 1:
            self.v = min(self.v * 5, 0.64)  # vol quintuples
            self.regime = 2  # force crisis regime
        return super().step(action)


class EarningsEventEnv(HybridEnv):
    """Simulates a scheduled earnings event — large jump at known step."""
    def __init__(self, event_step: int = 30, surprise_magnitude: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        self.event_step = event_step
        self.surprise   = surprise_magnitude

    def step(self, action):
        if self.step_idx == self.event_step - 1:
            direction = 1 if np.random.random() > 0.5 else -1
            self.S *= (1 + direction * self.surprise)
            self.v  = min(self.v * 3, 0.50)
        return super().step(action)


class OvernightGapEnv(HybridEnv):
    """Simulates overnight gap risk — price jumps between sessions."""
    def __init__(self, gap_steps: list = None, **kwargs):
        super().__init__(**kwargs)
        self.gap_steps = gap_steps or [20, 40]

    def step(self, action):
        if self.step_idx in [g - 1 for g in self.gap_steps]:
            gap = np.random.normal(0, 0.015)   # ~1.5% overnight gap
            self.S *= np.exp(gap)
        return super().step(action)
