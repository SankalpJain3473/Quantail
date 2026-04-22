"""
envs/heston_env.py
==================
Heston Stochastic Volatility Gym Environment

Mathematical Foundation (from paper):
  Definition 4.1 — Heston Dynamics:
    dS_t = mu*S_t*dt + sqrt(v_t)*S_t*dW^S_t
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW^v_t
    d<W^S, W^v>_t = rho*dt

  Lemma 4.2 — Discrete Hedging Error:
    epsilon_i = Delta_V_i - Delta_i * Delta_S_i
    E[epsilon_i^2] = O(dt^2)

  Corollary 4.1 — RL Objective:
    L(pi) = E[sum_i epsilon_i^2] + lambda * E[TotalCost]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm


class HestonEnv(gym.Env):
    """
    Options hedging environment under Heston stochastic volatility.

    Observation (10-dim):
      [0] S/K          — moneyness
      [1] v            — current variance
      [2] tau/T        — normalized time to expiry
      [3] delta        — BS delta (feature, not policy)
      [4] gamma        — BS gamma
      [5] hedge        — current hedge ratio in [-1, 1]
      [6] pnl          — normalized running P&L
      [7] cost_used    — fraction of cost budget used
      [8] realized_vol — 20-step rolling realized volatility  [NEW]
      [9] drawdown     — current drawdown from peak           [NEW]

    Action: integer in {0,...,10}
      maps to hedge adjustment in {-0.05, -0.04, ..., 0.04, 0.05}

    Reward:
      -epsilon_i^2 - lambda * transaction_cost
      (Corollary 4.1: minimize hedging error + cost jointly)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 0.25,          # 3-month option
        r: float = 0.05,          # risk-free rate
        # Heston parameters
        kappa: float = 2.0,       # mean reversion speed
        theta: float = 0.04,      # long-run variance (20% vol)
        xi: float = 0.3,          # vol-of-vol
        rho: float = -0.7,        # price-vol correlation (leverage effect)
        v0: float = 0.04,         # initial variance
        # Trading parameters
        n_steps: int = 60,        # daily rebalancing for 3 months
        lam: float = 0.01,        # transaction cost weight (lambda)
        spread: float = 0.001,    # bid-ask half-spread
        cost_budget: float = 5.0, # max total transaction cost
    ):
        super().__init__()

        # Store all parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.lam = lam
        self.spread = spread
        self.cost_budget = cost_budget

        # Feller condition check (Definition 4.1, ensures v_t > 0 a.s.)
        feller = 2 * kappa * theta
        xi_sq = xi ** 2
        assert feller > xi_sq, (
            f"Feller condition violated: 2*kappa*theta={feller:.4f} "
            f"<= xi^2={xi_sq:.4f}. Increase kappa or theta, or reduce xi."
        )

        # Spaces
        # 11 discrete actions: -5% to +5% delta adjustment in 1% steps
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.S = None
        self.v = None
        self.hedge = None
        self.step_idx = None
        self.pnl = None
        self.total_cost = None
        self.V = None
        self.delta = None
        self.episode_history = []

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.S = self.S0
        self.v = self.v0
        self.hedge = 0.0
        self.step_idx = 0
        self.pnl = 0.0
        self.total_cost = 0.0
        self.episode_history = []

        tau = self.T
        self.V = self._option_price(self.S, self.v, tau)
        self.delta = self._bs_delta(self.S, self.v, tau)

        # New tracking for 10-dim obs
        self._price_history = [self.S]
        self._peak_value    = 1.0

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Execute one hedging step.

        action: int in [0, 10]
          0  -> adjust hedge by -0.05 (reduce by 5%)
          5  -> no change
          10 -> adjust hedge by +0.05 (increase by 5%)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Map action to hedge adjustment
        adj = (action - 5) * 0.01
        new_hedge = float(np.clip(self.hedge + adj, -1.0, 1.0))

        # Transaction cost: spread * |delta_hedge| * S
        delta_h = abs(new_hedge - self.hedge)
        cost = self.spread * delta_h * self.S
        self.hedge = new_hedge
        self.total_cost += cost

        # Simulate one Heston step (Euler-Maruyama, Definition 4.1)
        S_old = self.S
        V_old = self.V
        self.S, self.v = self._euler_maruyama_step(self.S, self.v)
        self.step_idx += 1

        # Compute new option value and greeks
        tau = max(self.T - self.step_idx * self.dt, 1e-8)
        self.V = self._option_price(self.S, self.v, tau)
        self.delta = self._bs_delta(self.S, self.v, tau)
        gamma = self._bs_gamma(self.S, self.v, tau)

        # Hedging error (Lemma 4.2)
        dV = self.V - V_old
        dS = self.S - S_old
        hedging_error = dV - self.hedge * dS

        # Reward: -epsilon^2 - lambda*cost (Corollary 4.1)
        reward = -(hedging_error ** 2) - self.lam * cost
        self.pnl += reward

        # Record history
        self.episode_history.append({
            "step": self.step_idx,
            "S": self.S,
            "v": self.v,
            "hedge": self.hedge,
            "hedging_error": hedging_error,
            "cost": cost,
            "reward": reward,
            "pnl": self.pnl,
            "delta": self.delta,
            "gamma": gamma,
        })

        terminated = self.step_idx >= self.n_steps
        truncated = self.total_cost > self.cost_budget

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
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────
    def _euler_maruyama_step(self, S: float, v: float):
        """
        Euler-Maruyama discretization of Heston SDEs (Definition 4.1).
        Uses full truncation: v_plus = max(v, 0) to ensure positivity.
        Uses log-Euler for S to ensure S > 0 exactly.
        """
        dt = self.dt
        v_plus = max(v, 0.0)

        # Correlated Brownian motions
        z1 = np.random.standard_normal()
        z2 = np.random.standard_normal()
        dW_S = z1
        dW_v = self.rho * z1 + np.sqrt(max(1 - self.rho ** 2, 0)) * z2

        # Asset price (log-Euler, ensures S > 0)
        S_new = S * np.exp(
            (self.r - 0.5 * v_plus) * dt
            + np.sqrt(v_plus * dt) * dW_S
        )

        # Variance (full truncation Euler-Maruyama)
        v_new = max(
            v + self.kappa * (self.theta - v_plus) * dt
            + self.xi * np.sqrt(v_plus * dt) * dW_v,
            0.0,
        )

        return S_new, v_new

    # ─────────────────────────────────────────────────────────────────────
    def _option_price(self, S: float, v: float, tau: float) -> float:
        """Black-Scholes call price using current variance as proxy."""
        if tau < 1e-8:
            return max(S - self.K, 0.0)
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * tau) / (
            sigma * np.sqrt(tau)
        )
        d2 = d1 - sigma * np.sqrt(tau)
        price = S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        return max(price, 0.0)

    def _bs_delta(self, S: float, v: float, tau: float) -> float:
        """Black-Scholes delta."""
        if tau < 1e-8:
            return 1.0 if S > self.K else 0.0
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * tau) / (
            sigma * np.sqrt(tau)
        )
        return float(norm.cdf(d1))

    def _bs_gamma(self, S: float, v: float, tau: float) -> float:
        """Black-Scholes gamma."""
        if tau < 1e-8:
            return 0.0
        sigma = np.sqrt(max(v, 1e-8))
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * tau) / (
            sigma * np.sqrt(tau)
        )
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(tau)))

    # ─────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        tau = max(self.T - self.step_idx * self.dt, 1e-8)
        gamma = self._bs_gamma(self.S, self.v, tau)

        # Realized volatility (20-step rolling)
        if not hasattr(self, '_price_history'):
            self._price_history = [self.S]
            self._peak_value = 1.0
        self._price_history.append(self.S)
        if len(self._price_history) > 21:
            self._price_history = self._price_history[-21:]
        if len(self._price_history) >= 2:
            log_rets = np.diff(np.log(np.maximum(self._price_history, 1e-8)))
            realized_vol = float(np.std(log_rets) * np.sqrt(252))
        else:
            realized_vol = np.sqrt(max(self.v, 0))

        # Drawdown from peak
        portfolio_val = 1.0 + self.pnl / (self.K * 100 + 1e-8)
        if portfolio_val > self._peak_value:
            self._peak_value = portfolio_val
        drawdown = float(np.clip(
            (self._peak_value - portfolio_val) / (self._peak_value + 1e-8),
            0.0, 1.0
        ))

        return np.array([
            self.S / self.K,                                   # [0] moneyness
            float(np.clip(self.v, 0, 1)),                      # [1] variance
            tau / self.T,                                      # [2] time
            float(np.clip(self.delta, 0, 1)),                  # [3] delta
            float(np.clip(gamma * self.S, 0, 5)),              # [4] gamma
            float(np.clip(self.hedge, -1, 1)),                 # [5] hedge
            float(np.clip(self.pnl / 100, -5, 5)),             # [6] pnl
            float(np.clip(self.total_cost / self.cost_budget, 0, 1)),  # [7] cost
            float(np.clip(realized_vol, 0, 2)),                # [8] realized vol NEW
            float(drawdown),                                   # [9] drawdown NEW
        ], dtype=np.float32)

    def render(self, mode="human"):
        tau = max(self.T - self.step_idx * self.dt, 0)
        print(
            f"Step {self.step_idx:3d} | "
            f"S={self.S:7.2f} K={self.K} tau={tau:.3f} | "
            f"v={self.v:.4f} delta={self.delta:.3f} | "
            f"hedge={self.hedge:.3f} cost={self.total_cost:.4f} | "
            f"pnl={self.pnl:.4f}"
        )
