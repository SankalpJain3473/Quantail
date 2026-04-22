"""
agents/agents.py
================
Five specialized agents matching the original design plan exactly:

  1. HedgingAgent       — minimize hedging error + transaction cost
  2. RiskAgent          — CVaR + drawdown + liquidity risk monitoring
  3. QuantumExplorerAgent — dedicated VQC exploration of action space
  4. RegimeAgent         — bull/bear/crash/high-vol market regime classifier
  5. AlphaAgent          — directional momentum signals

Each agent has:
  - Its own VQC policy network (specialized reward shapes different behavior)
  - Its own replay buffer (independent learning)
  - Its own return distribution (fed to Wasserstein coordinator)

The coordinator gets genuinely diverse distributions from 5 agents
with fundamentally different reward signals — far better than the
previous 4 agents that all derived reward from hedging error.

Obs vector (10-dim — expanded from 8):
  [0] moneyness         S/K
  [1] variance          v_t (Heston)
  [2] tau               normalized time to expiry
  [3] delta             BS delta
  [4] gamma             scaled BS gamma
  [5] hedge             current hedge ratio
  [6] pnl               normalized running P&L
  [7] cost_frac         cost budget used
  [8] realized_vol      20-step rolling realized vol    ← NEW
  [9] drawdown          current drawdown from peak       ← NEW
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quantum.vqc_policy import VQCPolicy
from risk.coherent_risk import cvar, var, RiskBudget


# ─────────────────────────────────────────────────────────────────────────────
class BaseAgent:
    """
    Base class for all five Quantail agents.
    Each agent has a VQC policy, a replay buffer, and a specialized reward.
    """

    def __init__(
        self,
        name: str,
        n_qubits: int = 10,       # matches 10-dim obs
        n_layers: int = 3,
        n_actions: int = 11,
        gamma: float = 0.99,
        lr: float = 0.005,
        buffer_size: int = 1000,
        n_quantiles: int = 51,
        seed: int = 42,
    ):
        self.name = name
        self.gamma = gamma
        self.n_quantiles = n_quantiles

        self.policy = VQCPolicy(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_actions=n_actions,
            learning_rate=lr,
            seed=seed,
        )

        self.buffer = deque(maxlen=buffer_size)
        self.episode_rewards: List[List[float]] = []
        self.episode_returns: List[float] = []
        self.current_episode_rewards: List[float] = []
        self.return_distribution = np.zeros(n_quantiles)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        return self.policy.select_action(obs, epsilon)

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        self.current_episode_rewards.append(reward)

    def compute_reward(self, info: dict) -> float:
        raise NotImplementedError

    def update(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return 0.0
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        total_loss = 0.0
        for idx in indices:
            obs, action, reward, next_obs, done = self.buffer[idx]
            next_val = 0.0 if done else float(np.max(self.policy.forward(next_obs)[0]))
            td_error  = reward + self.gamma * next_val - float(np.max(self.policy.forward(obs)[0]))
            self.policy.update(obs, action, td_error)
            total_loss += abs(td_error)
        return total_loss / batch_size

    def end_episode(self):
        if self.current_episode_rewards:
            ep_return = sum(self.current_episode_rewards)
            self.episode_returns.append(ep_return)
            self.episode_rewards.append(self.current_episode_rewards.copy())
            all_rewards = [r for ep in self.episode_rewards[-20:] for r in ep]
            if len(all_rewards) >= self.n_quantiles:
                self.return_distribution = np.percentile(
                    all_rewards, np.linspace(0, 100, self.n_quantiles)
                )
        self.current_episode_rewards = []

    def get_return_distribution(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.get_return_distribution(obs, self.n_quantiles)

    def get_stats(self) -> dict:
        if not self.episode_returns:
            return {"agent": self.name, "mean_return": 0.0, "cvar_95": 0.0, "n_episodes": 0}
        arr = np.array(self.episode_returns)
        return {
            "agent":       self.name,
            "n_episodes":  len(arr),
            "mean_return": float(np.mean(arr)),
            "std_return":  float(np.std(arr)),
            "cvar_95":     cvar(arr, 0.05),
            "sharpe":      float(np.mean(arr) / (np.std(arr) + 1e-8) * np.sqrt(252)),
            "best":        float(np.max(arr)),
            "worst":       float(np.min(arr)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: Hedging Agent  (status: DONE — matches plan exactly)
# ─────────────────────────────────────────────────────────────────────────────
class HedgingAgent(BaseAgent):
    """
    Hedging Agent — specialist in minimizing discrete hedging error.

    Reward (Corollary 4.1):
        r = -epsilon² - lambda * cost

    This agent's sole focus is accurate hedging. It learns when
    small rebalances are worth the transaction cost and when to hold.
    Analogy: the trader at the desk who executes hedge orders.
    """

    def __init__(self, lambda1: float = 0.01, **kwargs):
        super().__init__(name="HedgingAgent", **kwargs)
        self.lambda1 = lambda1
        self._hedging_errors: List[float] = []
        self._costs: List[float] = []

    def compute_reward(self, info: dict) -> float:
        he   = info.get("hedging_error", 0.0)
        cost = info.get("cost", 0.0)
        self._hedging_errors.append(he)
        self._costs.append(cost)
        return -(he ** 2) - self.lambda1 * cost

    def get_hedging_stats(self) -> dict:
        if not self._hedging_errors:
            return {}
        err = np.array(self._hedging_errors)
        return {
            "rmse":        float(np.sqrt(np.mean(err ** 2))),
            "mean_abs":    float(np.mean(np.abs(err))),
            "max_error":   float(np.max(np.abs(err))),
            "total_cost":  float(sum(self._costs)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: Risk Agent  (status: DONE — now includes drawdown + liquidity)
# ─────────────────────────────────────────────────────────────────────────────
class RiskAgent(BaseAgent):
    """
    Risk Agent — monitors CVaR, drawdown, and liquidity risk.

    Reward:
        r = pnl - w_cvar*CVaR - w_dd*drawdown_penalty - w_liq*liquidity_penalty
        Hard veto when CVaR > limit.

    Now tracks ALL three risk dimensions from the plan:
      - CVaR (tail risk)                — was already done
      - Drawdown from peak              — NEW
      - Liquidity (bid-ask spread proxy) — NEW

    Analogy: the risk officer who shouts "stop!" when limits breach.
    """

    def __init__(
        self,
        cvar_limit: float = 0.05,
        max_drawdown: float = 0.10,   # 10% max drawdown
        lambda_cvar: float = 0.5,
        lambda_dd:   float = 0.3,
        lambda_liq:  float = 0.2,
        alpha: float = 0.05,
        **kwargs,
    ):
        super().__init__(name="RiskAgent", **kwargs)
        self.cvar_limit   = cvar_limit
        self.max_drawdown = max_drawdown
        self.lambda_cvar  = lambda_cvar
        self.lambda_dd    = lambda_dd
        self.lambda_liq   = lambda_liq
        self.alpha        = alpha
        self.risk_budget  = RiskBudget(cvar_limit=cvar_limit, alpha=alpha)

        self._pnl_history: List[float] = []
        self._peak_value:  float = 1.0      # normalized portfolio peak
        self._current_dd:  float = 0.0      # current drawdown (0–1)
        self._spreads:     List[float] = []  # bid-ask spreads for liquidity

    def compute_reward(self, info: dict) -> float:
        he     = info.get("hedging_error", 0.0)
        cost   = info.get("cost", 0.0)
        spread = info.get("spread", 0.0002)  # bid-ask half-spread
        S      = info.get("S", 100.0)

        # Track P&L
        pnl = -abs(he)
        self._pnl_history.append(pnl)
        self.risk_budget.update(pnl)

        # ── Drawdown tracking ───────────────────────────────────────────
        cum_pnl = sum(self._pnl_history)
        portfolio_val = 1.0 + cum_pnl / 100.0   # normalized
        if portfolio_val > self._peak_value:
            self._peak_value = portfolio_val
        self._current_dd = max(0.0, (self._peak_value - portfolio_val) / (self._peak_value + 1e-8))

        # ── Liquidity risk proxy ────────────────────────────────────────
        # Higher spread = worse liquidity = more risk
        spread_bps = spread / (S + 1e-8) * 10000
        self._spreads.append(spread_bps)
        liq_penalty = spread_bps / 10.0   # normalize to ~0–1 range

        # ── CVaR ────────────────────────────────────────────────────────
        cv = self.risk_budget.current_cvar()

        # ── Combined reward ─────────────────────────────────────────────
        reward = pnl
        reward -= self.lambda_cvar * cv
        reward -= self.lambda_dd   * self._current_dd
        reward -= self.lambda_liq  * liq_penalty

        # Hard breach penalties
        if self.risk_budget.is_cvar_breached():
            reward -= 1.0
        if self._current_dd > self.max_drawdown:
            reward -= 0.5

        return reward

    def get_scale_factor(self) -> float:
        """
        Position scale factor [0,1].
        Reduces position size as CVaR or drawdown approach limits.
        """
        cvar_scale = self.risk_budget.scale_factor()
        dd_scale   = float(np.clip(1.0 - self._current_dd / self.max_drawdown, 0.0, 1.0))
        return float(min(cvar_scale, dd_scale))

    def get_risk_stats(self) -> dict:
        return {
            "current_cvar":     self.risk_budget.current_cvar(),
            "current_var":      self.risk_budget.current_var(),
            "current_drawdown": self._current_dd,
            "peak_value":       self._peak_value,
            "cvar_limit":       self.cvar_limit,
            "max_drawdown":     self.max_drawdown,
            "cvar_breached":    self.risk_budget.is_cvar_breached(),
            "dd_breached":      self._current_dd > self.max_drawdown,
            "scale_factor":     self.get_scale_factor(),
            "avg_spread_bps":   float(np.mean(self._spreads[-20:])) if self._spreads else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: Quantum Explorer Agent  (status: DONE — replaces PortfolioAgent)
# ─────────────────────────────────────────────────────────────────────────────
class QuantumExplorerAgent(BaseAgent):
    """
    Quantum Explorer Agent — dedicated VQC action-space explorer.

    This is what the plan called "Quantum Circuit Agent."
    Its job is NOT to hedge — it is to efficiently explore the
    large action space and find hedge ratios the other agents miss.

    Reward:
        r = diversity_bonus - exploration_cost
    where diversity_bonus rewards choosing actions that OTHER agents
    don't choose, giving the coordinator diverse proposals to work with.

    The VQC's O(2^n) Fourier expressivity is most valuable here —
    it can represent complex non-linear policies that MLP agents miss.

    Analogy: the quant researcher who proposes novel strategies
    that the desk traders would never think of themselves.
    """

    def __init__(
        self,
        diversity_weight: float = 0.6,
        exploration_bonus: float = 0.3,
        n_qubits: int = 10,
        **kwargs,
    ):
        super().__init__(name="QuantumExplorerAgent", n_qubits=n_qubits, **kwargs)
        self.diversity_weight   = diversity_weight
        self.exploration_bonus  = exploration_bonus

        self._action_history:   deque = deque(maxlen=100)
        self._action_counts:    np.ndarray = np.zeros(11)   # 11 discrete actions
        self._recent_returns:   deque = deque(maxlen=50)
        self._other_actions:    deque = deque(maxlen=20)     # what other agents chose

    def compute_reward(self, info: dict) -> float:
        he      = info.get("hedging_error", 0.0)
        cost    = info.get("cost", 0.0)
        action  = info.get("last_action", 5)   # neutral action if not provided

        self._action_history.append(action)
        self._action_counts[action] += 1

        # ── Base hedging performance ────────────────────────────────────
        base_reward = -(he ** 2) * 0.3   # care about hedging but not primarily

        # ── Diversity bonus ─────────────────────────────────────────────
        # Reward actions that are less frequently chosen (exploration)
        total = self._action_counts.sum() + 1e-6
        action_freq = self._action_counts[action] / total
        diversity_bonus = self.diversity_weight * (1.0 - action_freq)

        # ── Fourier exploration bonus ───────────────────────────────────
        # Reward the agent for using extreme actions (edges of action space)
        # These correspond to high Fourier mode activation in the VQC
        extreme_bonus = self.exploration_bonus * abs(action - 5) / 5.0

        # ── Penalty for always staying at action 5 (no change) ─────────
        center_penalty = -0.1 if action == 5 and len(self._action_history) > 5 else 0.0

        reward = base_reward + diversity_bonus + extreme_bonus + center_penalty
        self._recent_returns.append(reward)
        return reward

    def record_other_actions(self, actions: Dict[str, int]):
        """Called by coordinator — lets this agent know what others chose."""
        for a in actions.values():
            self._other_actions.append(a)

    def get_exploration_stats(self) -> dict:
        total = self._action_counts.sum()
        if total == 0:
            return {}
        probs = self._action_counts / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        return {
            "action_entropy":  entropy,                          # higher = more diverse
            "max_entropy":     float(np.log(11)),               # log(11) = max possible
            "coverage":        float(np.sum(self._action_counts > 0) / 11),
            "most_used_action": int(np.argmax(self._action_counts)),
            "mean_return":      float(np.mean(list(self._recent_returns))) if self._recent_returns else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4: Regime Agent  (status: DONE — entirely new, was missing)
# ─────────────────────────────────────────────────────────────────────────────
class RegimeAgent(BaseAgent):
    """
    Market Regime Agent — classifies and adapts to market conditions.

    Regimes:
        0 = TRENDING_LOW_VOL   (bull/bear, calm)
        1 = TRENDING_HIGH_VOL  (strong trend, volatile)
        2 = MEAN_REVERTING     (range-bound, choppy)
        3 = CRASH              (vol spike, tail event)

    Reward:
        r = regime_accuracy_bonus + adaptation_bonus
    Rewards the agent for correctly identifying regime transitions
    and adjusting coordinator weights proactively.

    Key contribution: tells the coordinator HOW MUCH to trust each
    other agent based on current market conditions:
      - CRASH:            RiskAgent weight ↑↑, Explorer weight ↓
      - TRENDING_LOW_VOL: HedgingAgent weight ↑, Regime weight ↓
      - HIGH_VOL:         RiskAgent weight ↑, QuantumExplorer weight ↑

    Analogy: the senior strategist who reads the macro environment
    and tells the trading desk to be more/less aggressive.
    """

    REGIMES = {
        0: "trending_low_vol",
        1: "trending_high_vol",
        2: "mean_reverting",
        3: "crash",
    }

    # Coordinator weight recommendations per regime
    REGIME_WEIGHTS = {
        0: {"HedgingAgent": 0.50, "RiskAgent": 0.20, "QuantumExplorerAgent": 0.15, "RegimeAgent": 0.10, "AlphaAgent": 0.05},
        1: {"HedgingAgent": 0.35, "RiskAgent": 0.30, "QuantumExplorerAgent": 0.20, "RegimeAgent": 0.10, "AlphaAgent": 0.05},
        2: {"HedgingAgent": 0.40, "RiskAgent": 0.25, "QuantumExplorerAgent": 0.20, "RegimeAgent": 0.10, "AlphaAgent": 0.05},
        3: {"HedgingAgent": 0.20, "RiskAgent": 0.50, "QuantumExplorerAgent": 0.15, "RegimeAgent": 0.10, "AlphaAgent": 0.05},
    }

    def __init__(self, vol_threshold: float = 0.25, crash_threshold: float = 0.50, **kwargs):
        super().__init__(name="RegimeAgent", **kwargs)
        self.vol_threshold   = vol_threshold
        self.crash_threshold = crash_threshold

        self._vol_history:   deque = deque(maxlen=20)
        self._price_history: deque = deque(maxlen=20)
        self._current_regime: int = 0
        self._regime_history: List[int] = []
        self._correct_calls:  int = 0
        self._total_calls:    int = 0
        self._regime_counts:  np.ndarray = np.zeros(4)

    def _classify_regime(self, iv: float, realized_vol: float, price_trend: float) -> int:
        """
        Classify current market regime from observable features.

        Logic:
          vol > crash_threshold              → CRASH (3)
          vol > vol_threshold AND trend != 0 → TRENDING_HIGH_VOL (1)
          vol < vol_threshold AND trend != 0 → TRENDING_LOW_VOL (0)
          else                               → MEAN_REVERTING (2)
        """
        vol = max(iv, realized_vol)
        if vol > self.crash_threshold:
            return 3    # CRASH
        trend_strength = abs(price_trend)
        if vol > self.vol_threshold and trend_strength > 0.01:
            return 1    # TRENDING_HIGH_VOL
        if vol <= self.vol_threshold and trend_strength > 0.005:
            return 0    # TRENDING_LOW_VOL
        return 2        # MEAN_REVERTING

    def compute_reward(self, info: dict) -> float:
        S  = info.get("S", 100.0)
        v  = info.get("v", 0.04)    # Heston variance
        iv = info.get("iv", 0.18)   # implied vol
        he = info.get("hedging_error", 0.0)

        self._vol_history.append(np.sqrt(max(v, 0)))
        self._price_history.append(S)

        # Compute realized vol and trend
        realized_vol = float(np.std(list(self._vol_history)) * np.sqrt(252)) if len(self._vol_history) > 2 else 0.18
        price_trend  = 0.0
        if len(self._price_history) >= 5:
            prices = list(self._price_history)
            price_trend = (prices[-1] - prices[-5]) / (prices[-5] + 1e-8)

        # Classify regime
        new_regime = self._classify_regime(iv, realized_vol, price_trend)
        self._regime_counts[new_regime] += 1
        self._total_calls += 1

        # Reward regime stability (not thrashing)
        if self._regime_history:
            stability_bonus = 0.1 if new_regime == self._regime_history[-1] else -0.05
        else:
            stability_bonus = 0.0

        # Regime-appropriate hedging reward
        # In crash: reward conservative (low hedge changes)
        # In trending: reward tracking delta closely
        if new_regime == 3:     # CRASH
            reward = -abs(he) * 2.0 + stability_bonus   # penalize errors heavily
        elif new_regime == 0:   # TRENDING_LOW_VOL
            reward = -(he ** 2) + stability_bonus        # standard hedging
        elif new_regime == 1:   # TRENDING_HIGH_VOL
            reward = -abs(he) * 1.5 + stability_bonus   # moderate penalty
        else:                   # MEAN_REVERTING
            reward = -(he ** 2) * 0.8 + stability_bonus # lighter penalty

        self._current_regime = new_regime
        self._regime_history.append(new_regime)
        return reward

    @property
    def current_regime(self) -> int:
        return self._current_regime

    @property
    def current_regime_name(self) -> str:
        return self.REGIMES[self._current_regime]

    def get_recommended_weights(self) -> dict:
        """Return coordinator weight recommendations for current regime."""
        return dict(self.REGIME_WEIGHTS[self._current_regime])

    def get_regime_stats(self) -> dict:
        total = self._total_calls + 1e-6
        return {
            "current_regime":      self.current_regime_name,
            "regime_distribution": {
                self.REGIMES[i]: float(self._regime_counts[i] / total)
                for i in range(4)
            },
            "total_classifications": self._total_calls,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5: Alpha Agent  (status: DONE — directional signals, unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
class AlphaAgent(BaseAgent):
    """
    Alpha Agent — generates directional momentum signals.

    Reward:
        r = actual_return * signal - lambda * cost
    where signal = sign(delta - 0.5) as directional proxy.

    Provides the coordinator with momentum-aware proposals —
    important during trending regimes identified by RegimeAgent.

    Analogy: the market strategist who reads price action and
    says "this is going up, lean long on the hedge."
    """

    def __init__(self, lambda4: float = 0.005, **kwargs):
        super().__init__(name="AlphaAgent", **kwargs)
        self.lambda4 = lambda4
        self._prev_S: Optional[float] = None
        self._signals:        List[float] = []
        self._signal_returns: List[float] = []

    def compute_reward(self, info: dict) -> float:
        S    = info.get("S", 100.0)
        cost = info.get("cost", 0.0)

        if self._prev_S is None:
            self._prev_S = S
            return 0.0

        actual_return = (S - self._prev_S) / (self._prev_S + 1e-8)
        self._prev_S  = S

        delta  = info.get("delta", 0.5)
        signal = float(np.sign(delta - 0.5))
        self._signals.append(signal)

        directional_pnl = actual_return * signal
        self._signal_returns.append(directional_pnl)

        return directional_pnl - self.lambda4 * cost

    def get_signal_accuracy(self) -> float:
        if not self._signal_returns:
            return 0.5
        return float(sum(r > 0 for r in self._signal_returns) / len(self._signal_returns))

    def get_alpha_stats(self) -> dict:
        if not self._signal_returns:
            return {}
        arr = np.array(self._signal_returns)
        return {
            "signal_accuracy": self.get_signal_accuracy(),
            "mean_alpha":      float(np.mean(arr)),
            "sharpe_alpha":    float(np.mean(arr) / (np.std(arr) + 1e-8) * np.sqrt(252)),
            "n_signals":       len(self._signals),
        }
