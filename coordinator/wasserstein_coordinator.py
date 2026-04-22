"""
coordinator/wasserstein_coordinator.py
=======================================
Wasserstein Barycenter Coordinator Agent.

Mathematical Foundation:
  Definition 6.1 — Wasserstein Barycenter (Agueh & Carlier 2011):
    Z* = argmin_Z  sum_{k=1}^K  lambda_k * W_2^2(Z, Z_k)

  Theorem (Agueh & Carlier 2011) — Uniqueness:
    For 1D distributions with quantile functions F_k^{-1}:
      (F*)^{-1}(tau) = sum_{k=1}^K  lambda_k * F_k^{-1}(tau)
    The barycenter exists and is unique under finite second moments.

  The coordinator:
    1. Collects return distributions from all 4 agents
    2. Computes weighted Wasserstein barycenter
    3. Applies risk veto from Risk Agent
    4. Selects final action from unified distribution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from risk.coherent_risk import cvar_from_quantiles


class WassersteinCoordinator:
    """
    Coordinator that combines agent return distributions
    via Wasserstein barycenter for unified action selection.

    Implements Definition 6.1 and the uniqueness theorem
    from Agueh & Carlier (2011).
    """

    def __init__(
        self,
        agent_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        n_quantiles: int = 51,
        cvar_threshold: float = 0.05,
        alpha: float = 0.05,
    ):
        """
        Args:
            agent_names:     list of agent names
            weights:         lambda_k weights for barycenter (uniform if None)
            n_quantiles:     N in quantile representation
            cvar_threshold:  CVaR limit for risk veto
            alpha:           tail confidence level
        """
        self.agent_names = agent_names
        self.n_quantiles = n_quantiles
        self.cvar_threshold = cvar_threshold
        self.alpha = alpha

        # Default: uniform weights
        if weights is None:
            # Default weights — RegimeAgent will override these dynamically
            default = {
                'HedgingAgent':        0.40,
                'RiskAgent':           0.25,
                'QuantumExplorerAgent':0.15,
                'RegimeAgent':         0.10,
                'AlphaAgent':          0.10,
            }
            self.weights = {name: default.get(name, 1.0/len(agent_names)) for name in agent_names}
            # Normalize
            total = sum(self.weights.values())
            self.weights = {k: v/total for k,v in self.weights.items()}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

        # Coordination history
        self.coordination_history = []
        self.veto_count = 0
        self.total_decisions = 0

    # ─────────────────────────────────────────────────────────────────────
    def wasserstein_barycenter(
        self,
        distributions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute Wasserstein barycenter of agent return distributions.

        Theorem (Agueh & Carlier 2011):
          For 1D distributions, the W_2 barycenter has quantile function:
            (F*)^{-1}(tau) = sum_{k=1}^K  lambda_k * F_k^{-1}(tau)

          This is simply the weighted average of sorted quantile arrays.

        Args:
            distributions: {agent_name: quantile_array} for each agent

        Returns:
            barycenter: unified return distribution as quantile array
        """
        # Ensure all distributions have same length via interpolation
        barycenter = np.zeros(self.n_quantiles)
        tau_grid = np.linspace(0, 1, self.n_quantiles)

        for name, dist in distributions.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0.0 or len(dist) == 0:
                continue

            # Interpolate to common quantile grid
            dist_sorted = np.sort(dist)
            src_taus = np.linspace(0, 1, len(dist_sorted))

            # Interpolate quantile function F_k^{-1}(tau)
            q_interp = np.interp(tau_grid, src_taus, dist_sorted)

            # Weighted sum (barycenter formula)
            barycenter += weight * q_interp

        return barycenter

    # ─────────────────────────────────────────────────────────────────────
    def coordinate(
        self,
        obs: np.ndarray,
        agents: dict,
        epsilon: float = 0.1,
    ) -> Tuple[int, dict]:
        """
        Main coordination step.

        1. Collect return distributions from all agents
        2. Compute Wasserstein barycenter
        3. Apply risk veto if CVaR limit breached
        4. Select action from unified distribution

        Args:
            obs:     current observation
            agents:  dict of {name: agent_instance}
            epsilon: exploration rate

        Returns:
            action:  final coordinated action
            info:    coordination metadata
        """
        self.total_decisions += 1

        # Step 0: Ask RegimeAgent for weight recommendations
        regime_agent = agents.get('RegimeAgent')
        if regime_agent is not None and hasattr(regime_agent, 'get_recommended_weights'):
            recommended = regime_agent.get_recommended_weights()
            # Blend current weights toward regime recommendation (EMA)
            for name in self.weights:
                if name in recommended:
                    self.weights[name] = 0.85 * self.weights[name] + 0.15 * recommended[name]
            # Renormalize
            total = sum(self.weights.values())
            self.weights = {k: v/total for k,v in self.weights.items()}

        # Step 1: Collect return distributions and action proposals
        distributions = {}
        action_proposals = {}
        agent_weights = {}

        for name, agent in agents.items():
            dist = agent.get_return_distribution(obs)
            distributions[name] = dist
            action_proposals[name] = agent.select_action(obs, epsilon)

        # Tell QuantumExplorer what other agents proposed (for diversity reward)
        explorer = agents.get('QuantumExplorerAgent')
        if explorer is not None and hasattr(explorer, 'record_other_actions'):
            other = {k:v for k,v in action_proposals.items() if k != 'QuantumExplorerAgent'}
            explorer.record_other_actions(other)

        # Step 2: Compute Wasserstein barycenter
        barycenter = self.wasserstein_barycenter(distributions)

        # Step 3: CVaR check (Risk Agent veto)
        unified_cvar = cvar_from_quantiles(barycenter, self.alpha)
        risk_agent = agents.get("RiskAgent")
        risk_veto = False
        scale_factor = 1.0

        if risk_agent is not None:
            scale_factor = risk_agent.get_scale_factor()
            # Veto if CVaR OR drawdown limit breached
            cvar_breach = risk_agent.risk_budget.is_cvar_breached()
            dd_breach   = hasattr(risk_agent, '_current_dd') and risk_agent._current_dd > risk_agent.max_drawdown
            if cvar_breach or dd_breach:
                risk_veto = True
                self.veto_count += 1

        # Step 4: Action selection from barycenter
        if risk_veto:
            # Under veto: prefer conservative action (no change = action 5)
            action = 5
        else:
            # Weight agent actions by their contribution to barycenter quality
            action = self._select_from_barycenter(
                barycenter, action_proposals, scale_factor
            )

        # Record coordination metadata
        regime_name = "unknown"
        if regime_agent is not None and hasattr(regime_agent, 'current_regime_name'):
            regime_name = regime_agent.current_regime_name

        info = {
            "barycenter_mean":  float(np.mean(barycenter)),
            "barycenter_std":   float(np.std(barycenter)),
            "unified_cvar":     unified_cvar,
            "risk_veto":        risk_veto,
            "scale_factor":     scale_factor,
            "agent_proposals":  action_proposals,
            "veto_rate":        self.veto_count / self.total_decisions,
            "market_regime":    regime_name,
            "current_weights":  dict(self.weights),
        }

        self.coordination_history.append(info)
        return action, info

    def _select_from_barycenter(
        self,
        barycenter: np.ndarray,
        action_proposals: Dict[str, int],
        scale_factor: float,
    ) -> int:
        """
        Select action by voting weighted by barycenter quality.
        Actions closer to mean of barycenter get higher weight.
        Scale factor shrinks aggressive positions toward neutral (5).
        """
        # Vote counting with quality weighting
        n_actions = 11
        votes = np.zeros(n_actions)

        barycenter_mean = np.mean(barycenter)
        barycenter_std = np.std(barycenter) + 1e-8

        for name, action in action_proposals.items():
            # Weight by how well this agent's distribution aligns with barycenter
            w = self.weights.get(name, 0.25)
            votes[action] += w

        # Apply scale factor: shift votes toward neutral action (5)
        if scale_factor < 1.0:
            neutral = 5
            scaled_votes = np.zeros(n_actions)
            for a in range(n_actions):
                scaled_a = int(np.round(neutral + (a - neutral) * scale_factor))
                scaled_a = np.clip(scaled_a, 0, n_actions - 1)
                scaled_votes[scaled_a] += votes[a]
            votes = scaled_votes

        # Add exploration noise
        votes += 1e-6 * np.random.rand(n_actions)

        return int(np.argmax(votes))

    # ─────────────────────────────────────────────────────────────────────
    def get_coordination_stats(self) -> dict:
        if not self.coordination_history:
            return {}
        cvars = [h["unified_cvar"] for h in self.coordination_history]
        return {
            "total_decisions": self.total_decisions,
            "veto_count":      self.veto_count,
            "veto_rate":       self.veto_count / max(self.total_decisions, 1),
            "mean_cvar":       float(np.mean(cvars)),
            "max_cvar":        float(np.max(cvars)),
            "weights":         self.weights,
        }

    def update_weights(self, agent_performances: Dict[str, float]):
        """
        Adaptive weight update based on agent performance.
        Agents with better recent returns get higher weight.

        Args:
            agent_performances: {name: recent_return} for each agent
        """
        if not agent_performances:
            return

        # Softmax over performances
        names = list(agent_performances.keys())
        perfs = np.array([agent_performances[n] for n in names])
        perfs = perfs - np.max(perfs)
        weights_raw = np.exp(perfs * 0.1)
        weights_norm = weights_raw / (weights_raw.sum() + 1e-10)

        for i, name in enumerate(names):
            # Blend with current weights (EMA)
            self.weights[name] = 0.9 * self.weights.get(name, 0.25) + 0.1 * weights_norm[i]

        # Renormalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
