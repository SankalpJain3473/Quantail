"""
backend/model_store.py
======================
Saves and loads trained VQC agent weights to/from disk.

Without this, agents retrain from scratch every time the
server restarts — losing all learned knowledge.

With this:
  - First boot: trains for PRETRAIN_EPISODES, saves weights
  - Subsequent boots: loads in <1 second, starts immediately
  - Weights persist across deployments
  - Each user session clones from the saved base

Storage format: numpy .npz files (lightweight, no ML framework needed)
Location: ./model_weights/ directory (configurable via env var)
"""

import os
import numpy as np
from datetime import datetime
from typing import Optional

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "./model_weights")


def ensure_dir():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)


def weights_path(version: str = "base") -> str:
    return os.path.join(WEIGHTS_DIR, f"quantail_agents_{version}.npz")


def save_weights(agents: dict, coordinator, version: str = "base") -> str:
    """
    Save all agent VQC weights and coordinator state to disk.

    Args:
        agents:      dict of {name: agent_instance}
        coordinator: WassersteinCoordinator instance
        version:     name tag for the save file

    Returns:
        path to saved file
    """
    ensure_dir()

    save_data = {
        "saved_at":   np.array([datetime.utcnow().isoformat()]),
        "agent_names": np.array(list(agents.keys())),
        "coord_weights": np.array(list(coordinator.weights.values())),
        "coord_weight_names": np.array(list(coordinator.weights.keys())),
    }

    for name, agent in agents.items():
        key = name.replace(" ", "_")
        save_data[f"{key}_params"]   = agent.policy.params
        save_data[f"{key}_W_out"]    = agent.policy.W_out
        save_data[f"{key}_b_out"]    = agent.policy.b_out
        save_data[f"{key}_n_qubits"] = np.array([agent.policy.params.shape[1]])
        save_data[f"{key}_n_layers"] = np.array([agent.policy.params.shape[0]])

    path = weights_path(version)
    np.savez_compressed(path, **save_data)
    size_kb = os.path.getsize(path) / 1024
    print(f"Weights saved: {path} ({size_kb:.1f} KB)")
    return path


def load_weights(agents: dict, coordinator, version: str = "base") -> bool:
    """
    Load saved VQC weights into agent instances.

    Args:
        agents:      dict of {name: agent_instance} (will be modified in place)
        coordinator: WassersteinCoordinator instance
        version:     name tag of the save file to load

    Returns:
        True if loaded successfully, False if no save file found
    """
    path = weights_path(version)
    if not os.path.exists(path):
        print(f"No saved weights found at {path}")
        return False

    try:
        data = np.load(path, allow_pickle=True)
        saved_at = str(data["saved_at"][0])

        for name, agent in agents.items():
            key = name.replace(" ", "_")
            if f"{key}_params" in data:
                saved_params = data[f"{key}_params"]
                # Only load if architecture matches
                if saved_params.shape == agent.policy.params.shape:
                    agent.policy.params = saved_params.copy()
                    agent.policy.W_out  = data[f"{key}_W_out"].copy()
                    agent.policy.b_out  = data[f"{key}_b_out"].copy()
                else:
                    print(f"  Shape mismatch for {name}: saved {saved_params.shape} vs current {agent.policy.params.shape}")
                    return False

        # Load coordinator weights
        if "coord_weights" in data and "coord_weight_names" in data:
            names   = [str(n) for n in data["coord_weight_names"]]
            weights = list(data["coord_weights"])
            for n, w in zip(names, weights):
                if n in coordinator.weights:
                    coordinator.weights[n] = float(w)
            # Normalize
            total = sum(coordinator.weights.values())
            coordinator.weights = {k: v/total for k, v in coordinator.weights.items()}

        print(f"Weights loaded from {path} (saved at {saved_at})")
        return True

    except Exception as e:
        print(f"Failed to load weights: {e}")
        return False


def weights_exist(version: str = "base") -> bool:
    return os.path.exists(weights_path(version))


def weights_info(version: str = "base") -> dict:
    """Return metadata about saved weights."""
    path = weights_path(version)
    if not os.path.exists(path):
        return {"exists": False}
    data = np.load(path, allow_pickle=True)
    return {
        "exists":     True,
        "path":       path,
        "saved_at":   str(data["saved_at"][0]),
        "size_kb":    round(os.path.getsize(path) / 1024, 1),
        "agents":     [str(n) for n in data["agent_names"]],
    }
