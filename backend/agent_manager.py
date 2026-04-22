"""
backend/agent_manager.py
=========================
Manages trained VQC agents for the FastAPI backend.

This is the missing link — connects the real QR-DQN + VQC policy
network to the live trading engine, replacing the hardcoded formula
in backend/main.py.

Each user session gets agents that:
  1. Are pre-trained on Heston SDE (background training on startup)
  2. Continue learning online as the session runs (online RL)
  3. Make decisions via the actual VQC policy forward pass
  4. Are coordinated by the Wasserstein barycenter

The key difference from the old hardcoded approach:
  OLD: hedge_adj = (delta - hedge) * 0.5   <-- formula, not learned
  NEW: action = vqc.forward(obs) -> policy -> hedge_adj  <-- learned
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import os
import warnings
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# Defer imports to avoid slow startup
_agents_module     = None
_coordinator_module = None
_env_module         = None


def _import_ml():
    global _agents_module, _coordinator_module, _env_module
    if _agents_module is None:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from agents.agents import HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent
        from coordinator.wasserstein_coordinator import WassersteinCoordinator
        from envs.heston_env import HestonEnv
        from envs.hybrid_env import HybridEnv
        _agents_module      = (HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent)
        _coordinator_module = WassersteinCoordinator
        _env_module         = HybridEnv


# ── How many episodes to pre-train on startup ──────────────────────────────
PRETRAIN_EPISODES = int(os.environ.get("PRETRAIN_EPISODES", "60"))


class AgentSession:
    """
    One set of trained VQC agents per user session.
    Agents are pre-trained on Heston SDE, then continue
    learning online during the live session.
    """

    def __init__(self, user_id: str, seed: int = 42):
        _import_ml()
        HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent = _agents_module
        WassersteinCoordinator = _coordinator_module

        self.user_id    = user_id
        self.trained    = False
        self.step_count = 0

        # Initialize the five VQC agents (10-qubit, matches 10-dim obs)
        self.agents = {
            "HedgingAgent":        HedgingAgent(seed=seed,   lr=0.015, n_qubits=13, n_layers=3),
            "RiskAgent":           RiskAgent(seed=seed+1,    lr=0.015, n_qubits=13, n_layers=3),
            "QuantumExplorerAgent":QuantumExplorerAgent(seed=seed+2, lr=0.015, n_qubits=13, n_layers=3),
            "RegimeAgent":         RegimeAgent(seed=seed+3,  lr=0.015, n_qubits=13, n_layers=3),
            "AlphaAgent":          AlphaAgent(seed=seed+4,   lr=0.015, n_qubits=13, n_layers=3),
        }

        # Wasserstein coordinator — regime-aware weights
        self.coordinator = WassersteinCoordinator(
            agent_names=list(self.agents.keys()),
            weights={
                "HedgingAgent":        0.40,
                "RiskAgent":           0.25,
                "QuantumExplorerAgent":0.15,
                "RegimeAgent":         0.10,
                "AlphaAgent":          0.10,
            },
            n_quantiles=51,
            cvar_threshold=0.04,
            alpha=0.05,
        )

        # Online learning buffer
        self._last_obs   = None
        self._last_action = None
        self._episode_steps = []

    def pretrain(self, n_episodes: int = PRETRAIN_EPISODES) -> dict:
        """
        Train agents on Heston SDE before going live.
        This is the offline training phase.

        Returns training stats.
        """
        _import_ml()
        HybridEnv = _env_module

        env = HybridEnv(
            S0=100, K=100, T=0.25,
            n_steps=20, lam=0.005,
        )

        returns = []
        for ep in range(n_episodes):
            eps = max(0.05, 0.5 * (1 - ep / n_episodes))
            obs, _ = env.reset()
            done = False
            ep_r  = 0.0

            while not done:
                action, _ = self.coordinator.coordinate(obs, self.agents, eps)
                obs_n, _, t, tr, info = env.step(action)
                done = t or tr

                for agent in self.agents.values():
                    r = agent.compute_reward(info)
                    agent.store(obs, action, r, obs_n, done)
                    ep_r += r / 4

                obs = obs_n

            for agent in self.agents.values():
                agent.end_episode()
                agent.update(batch_size=min(32, len(agent.buffer)))

            returns.append(ep_r)

            # Adaptive weight updates
            perfs = {
                name: agent.get_stats().get("mean_return", 0.0)
                for name, agent in self.agents.items()
            }
            self.coordinator.update_weights(perfs)

        self.trained = True

        # Save weights to disk so next boot is instant
        try:
            from backend.model_store import save_weights
            save_weights(self.agents, self.coordinator, version="base")
        except Exception as e:
            print(f"Could not save weights: {e}")

        recent = returns[-20:] if len(returns) >= 20 else returns
        return {
            "episodes":    n_episodes,
            "mean_return": float(np.mean(recent)),
            "final_weights": dict(self.coordinator.weights),
            "agent_stats": {
                name: agent.get_stats()
                for name, agent in self.agents.items()
            },
        }

    def decide(self, obs: np.ndarray) -> Tuple[int, dict]:
        """
        Make a trading decision using the VQC policy.

        This is the core call — replaces the hardcoded formula.
        Every live trade goes through the actual learned policy.

        Returns:
            action:   discrete action [0-10] -> hedge adj [-0.05, +0.05]
            info:     coordinator metadata (CVaR, veto, weights, proposals)
        """
        # Greedy if trained, slight exploration if not
        epsilon = 0.0 if self.trained else 0.15

        action, coord_info = self.coordinator.coordinate(
            obs, self.agents, epsilon=epsilon
        )

        self._last_obs    = obs.copy()
        self._last_action = action
        self.step_count  += 1

        return action, coord_info

    def learn_online(self, obs: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        """
        Online learning step — agents update from live experience.
        This is what makes the system adaptive in production.
        Called after every trade with the actual reward received.
        """
        if self._last_obs is None or self._last_action is None:
            return

        # Store experience in each agent's replay buffer
        for agent in self.agents.values():
            agent.store(self._last_obs, self._last_action, reward, next_obs, done)

        # Update every 10 steps (avoid overhead on every step)
        if self.step_count % 10 == 0:
            for agent in self.agents.values():
                if len(agent.buffer) >= 16:
                    agent.update(batch_size=16)

        if done:
            for agent in self.agents.values():
                agent.end_episode()
            perfs = {
                name: agent.get_stats().get("mean_return", 0.0)
                for name, agent in self.agents.items()
            }
            self.coordinator.update_weights(perfs)

    def get_vqc_info(self, obs: np.ndarray) -> dict:
        """
        Get detailed VQC internals for dashboard display.
        Shows exactly what each agent's quantum circuit produced.
        """
        info = {}
        for name, agent in self.agents.items():
            probs, measurements = agent.policy.forward(obs)
            dist = agent.get_return_distribution(obs)
            action = agent.select_action(obs, epsilon=0.0)
            info[name] = {
                "action":       int(action),
                "hedge_adj":    float((action - 5) * 0.01),
                "top_prob":     float(probs.max()),
                "action_probs": probs.round(4).tolist(),
                "measurements": measurements.round(4).tolist(),
                "cvar_dist":    float(-np.percentile(dist, 5)),
                "dist_mean":    float(np.mean(dist)),
                "dist_std":     float(np.std(dist)),
            }
        info["coordinator_weights"] = dict(self.coordinator.weights)
        info["veto_rate"]           = self.coordinator.get_coordination_stats().get("veto_rate", 0)
        info["expressivity"]        = self.agents["HedgingAgent"].policy.expressivity_info()
        return info


# ── Global agent manager ───────────────────────────────────────────────────
class AgentManager:
    """
    Manages one AgentSession per user.
    Handles background pre-training on startup.
    """

    def __init__(self):
        self._sessions: Dict[str, AgentSession] = {}
        self._training_tasks: Dict[str, asyncio.Task] = {}
        self._pretrained_base: Optional[AgentSession] = None
        self._base_ready = False

    async def startup_pretrain(self):
        """
        Pre-train a base agent session on startup.
        Tries to load saved weights first — only trains if no weights found.
        """
        from backend.model_store import weights_exist, load_weights, weights_info
        base = AgentSession(user_id="__base__", seed=42)

        if weights_exist("base"):
            info = weights_info("base")
            print(f"Loading saved weights from {info['saved_at']}...")
            if load_weights(base.agents, base.coordinator, "base"):
                base.trained = True
                self._pretrained_base = base
                self._base_ready = True
                print(f"Weights loaded instantly — skipping pre-training")
                return

        print(f"Pre-training VQC agents ({PRETRAIN_EPISODES} episodes)...")
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, base.pretrain, PRETRAIN_EPISODES)
        self._pretrained_base = base
        self._base_ready      = True
        print(f"Pre-training complete. Mean return: {stats['mean_return']:.4f}")
        print(f"Agent weights: {stats['final_weights']}")

    def get_or_create(self, user_id: str) -> AgentSession:
        """Get existing session or create new one from pre-trained base."""
        if user_id not in self._sessions:
            self._sessions[user_id] = AgentSession(user_id=user_id, seed=hash(user_id) % 10000)
            # If base is ready, copy its trained weights
            if self._base_ready and self._pretrained_base:
                self._copy_weights_from_base(self._sessions[user_id])
        return self._sessions[user_id]

    def _copy_weights_from_base(self, target: AgentSession):
        """Copy trained VQC parameters from base to a new user session."""
        import copy
        src = self._pretrained_base
        for name in target.agents:
            # Copy VQC policy parameters (10-qubit)
            target.agents[name].policy.params   = src.agents[name].policy.params.copy()
            target.agents[name].policy.W_out    = src.agents[name].policy.W_out.copy()
            target.agents[name].policy.b_out    = src.agents[name].policy.b_out.copy()
        # Copy coordinator weights
        target.coordinator.weights = dict(src.coordinator.weights)
        target.trained = True

    def reset_session(self, user_id: str):
        """Reset a user's agents to base pre-trained state."""
        if user_id in self._sessions:
            del self._sessions[user_id]

    def is_ready(self) -> bool:
        return self._base_ready


# Singleton
agent_manager = AgentManager()
