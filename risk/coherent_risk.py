"""
risk/coherent_risk.py
=====================
Coherent Risk Measures — CVaR and supporting utilities.

Mathematical Foundation:
  Definition 8.1 (Artzner et al. 1999) — Coherent Risk Measure:
    rho satisfies: Monotonicity, Subadditivity, Homogeneity, Translation invariance.
    VaR fails Subadditivity => NOT coherent.
    CVaR satisfies all four => coherent.

  Proposition 8.1 (Rockafellar-Uryasev 2000) — CVaR Dual Form:
    CVaR_alpha(X) = min_z { z + (1/(1-alpha)) * E[max(-X - z, 0)] }
"""

import numpy as np
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
def var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Value-at-Risk at confidence level alpha.
    VaR_alpha = -quantile_alpha of the return distribution.

    NOTE: VaR is NOT coherent — fails subadditivity (Axiom A2).
    Included only as comparison baseline.

    Args:
        returns: 1D array of portfolio returns
        alpha:   tail probability (0.05 = 95% VaR)

    Returns:
        VaR value (positive = loss)
    """
    return float(-np.percentile(returns, alpha * 100))


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall) at confidence level alpha.
    CVaR_alpha = E[-R | R <= -VaR_alpha]

    CVaR IS coherent — satisfies all four Artzner axioms.
    Direct implementation: mean of worst alpha fraction of returns.

    Args:
        returns: 1D array of portfolio returns
        alpha:   tail probability (0.05 = 95% CVaR)

    Returns:
        CVaR value (positive = expected tail loss)
    """
    var_val = var(returns, alpha)
    tail_returns = returns[returns <= -var_val]
    if len(tail_returns) == 0:
        return var_val
    return float(-np.mean(tail_returns))


def cvar_from_quantiles(
    quantiles: np.ndarray, alpha: float = 0.05
) -> float:
    """
    CVaR directly from QR-DQN quantile outputs.

    Remark (Section 6.2 of paper):
      CVaR_alpha ~= (1/floor(alpha*N)) * sum_{i=1}^{floor(alpha*N)} theta_i
      where theta_i are sorted quantile values (lowest first).

    This is the key link between distributional RL (Pillar 3)
    and coherent risk (Pillar 2).

    Args:
        quantiles: sorted array of N quantile values (return distribution)
        alpha:     tail probability

    Returns:
        CVaR estimate from quantile representation
    """
    n = len(quantiles)
    n_tail = max(1, int(np.floor(alpha * n)))
    sorted_q = np.sort(quantiles)
    return float(-np.mean(sorted_q[:n_tail]))


def rockafellar_uryasev_cvar(
    returns: np.ndarray, alpha: float = 0.05
) -> float:
    """
    CVaR via Rockafellar-Uryasev dual representation (Proposition 8.1):
      CVaR_alpha(X) = min_z { z + (1/(1-alpha)) * E[max(-X - z, 0)] }

    This form is convex and differentiable in z — used for optimization.

    Args:
        returns: 1D array of portfolio returns
        alpha:   tail probability

    Returns:
        CVaR value
    """
    z_star = np.percentile(returns, alpha * 100)

    def objective(z):
        return z + (1.0 / (1.0 - alpha)) * np.mean(np.maximum(-returns - z, 0))

    # Grid search over z (simple; use cvxpy for production)
    z_values = np.linspace(np.min(returns), np.max(returns), 1000)
    obj_values = [objective(z) for z in z_values]
    return float(min(obj_values))


def cvar_reward_penalty(
    hedging_errors: np.ndarray,
    alpha: float = 0.05,
    weight: float = 0.1,
) -> float:
    """
    CVaR penalty term for reward shaping.

    Used by the Risk Agent to penalize tail outcomes:
      reward_penalty = -weight * CVaR_alpha(hedging_errors)

    Args:
        hedging_errors: array of hedging errors in current episode
        alpha:          tail confidence level
        weight:         penalty weight

    Returns:
        Negative CVaR penalty (negative = bad)
    """
    if len(hedging_errors) < 2:
        return 0.0
    cv = cvar(-hedging_errors, alpha)
    return -weight * cv


def check_coherence_axioms(
    returns_X: np.ndarray,
    returns_Y: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Verify coherence axioms for CVaR on given data.
    Educational function demonstrating why CVaR beats VaR.

    Returns dict with axiom check results.
    """
    rho = cvar
    eps = 1e-6

    # A1: Monotonicity — if X <= Y a.s., rho(X) >= rho(Y)
    X_le_Y = np.all(returns_X <= returns_Y)
    if X_le_Y:
        a1 = rho(returns_X, alpha) >= rho(returns_Y, alpha) - eps
    else:
        a1 = True  # condition not met, axiom vacuously holds

    # A2: Subadditivity — rho(X+Y) <= rho(X) + rho(Y)
    rho_sum = rho(returns_X + returns_Y, alpha)
    rho_X = rho(returns_X, alpha)
    rho_Y = rho(returns_Y, alpha)
    a2_cvar = rho_sum <= rho_X + rho_Y + eps
    a2_var  = var(returns_X + returns_Y, alpha) <= var(returns_X, alpha) + var(returns_Y, alpha) + eps

    # A3: Positive homogeneity — rho(2X) = 2*rho(X)
    a3 = abs(rho(2 * returns_X, alpha) - 2 * rho(returns_X, alpha)) < eps * 10

    # A4: Translation invariance — rho(X + m) = rho(X) - m
    m = 1.0
    a4 = abs(rho(returns_X + m, alpha) - (rho(returns_X, alpha) - m)) < eps * 10

    results = {
        "A1_monotonicity_CVaR":         a1,
        "A2_subadditivity_CVaR":        a2_cvar,
        "A2_subadditivity_VaR":         a2_var,
        "A3_homogeneity_CVaR":          a3,
        "A4_translation_inv_CVaR":      a4,
        "CVaR_is_coherent":             all([a1, a2_cvar, a3, a4]),
        "VaR_is_coherent":              a2_var,
    }

    if verbose:
        print("\n=== Coherence Axiom Check ===")
        print(f"  A1 Monotonicity (CVaR):        {'PASS' if a1       else 'FAIL'}")
        print(f"  A2 Subadditivity (CVaR):       {'PASS' if a2_cvar  else 'FAIL'}")
        print(f"  A2 Subadditivity (VaR):        {'PASS' if a2_var   else 'FAIL'} <-- VaR fails here")
        print(f"  A3 Homogeneity (CVaR):         {'PASS' if a3       else 'FAIL'}")
        print(f"  A4 Translation inv. (CVaR):    {'PASS' if a4       else 'FAIL'}")
        print(f"  CVaR is coherent:              {'YES'  if results['CVaR_is_coherent'] else 'NO'}")
        print(f"  VaR is coherent:               {'YES'  if results['VaR_is_coherent']  else 'NO'}")

    return results


class RiskBudget:
    """
    Portfolio risk budget manager.
    Tracks CVaR usage and enforces hard limits.
    """

    def __init__(
        self,
        cvar_limit: float = 0.05,   # max allowed CVaR (5% of portfolio)
        var_limit: float = 0.03,    # max allowed VaR
        alpha: float = 0.05,        # tail confidence level
        window: int = 20,           # rolling window for risk calculation
    ):
        self.cvar_limit = cvar_limit
        self.var_limit = var_limit
        self.alpha = alpha
        self.window = window
        self._returns_buffer = []

    def update(self, ret: float):
        self._returns_buffer.append(ret)
        if len(self._returns_buffer) > self.window:
            self._returns_buffer.pop(0)

    def current_cvar(self) -> float:
        if len(self._returns_buffer) < 2:
            return 0.0
        return cvar(np.array(self._returns_buffer), self.alpha)

    def current_var(self) -> float:
        if len(self._returns_buffer) < 2:
            return 0.0
        return var(np.array(self._returns_buffer), self.alpha)

    def is_cvar_breached(self) -> bool:
        return self.current_cvar() > self.cvar_limit

    def is_var_breached(self) -> bool:
        return self.current_var() > self.var_limit

    def scale_factor(self) -> float:
        """
        Returns position scale factor [0, 1].
        Reduces positions when approaching risk limits.
        """
        cv = self.current_cvar()
        if cv <= 0:
            return 1.0
        ratio = cv / self.cvar_limit
        return float(np.clip(1.0 - max(ratio - 0.8, 0) / 0.2, 0.0, 1.0))
