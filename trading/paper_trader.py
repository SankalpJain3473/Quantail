"""
trading/paper_trader.py
=======================
Paper Trading Engine for Quantail.

Connects the trained multi-agent system to real market data
and simulates order execution. This is the bridge between
the research PoC and live trading.

Flow:
  MarketFeed -> ObservationBuilder -> Agents -> Coordinator
      -> Signal -> PaperTrader -> Portfolio -> P&L tracking

No real money moves. Orders are executed at simulated fills
based on real bid/ask quotes.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trading.market_feed import MarketFeed, MarketSnapshot
from risk.coherent_risk import cvar, var, RiskBudget


@dataclass
class Position:
    """A single position in the portfolio."""
    symbol:     str
    qty:        float          # shares (can be fractional for hedge ratio)
    avg_price:  float
    current_price: float = 0.0
    position_type: str = 'stock'  # 'stock' or 'option'

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.qty * (self.current_price - self.avg_price)

    @property
    def cost_basis(self) -> float:
        return abs(self.qty * self.avg_price)


@dataclass
class Order:
    """A trade order."""
    symbol:     str
    qty:        float
    side:       str            # 'buy' or 'sell'
    order_type: str = 'market'
    fill_price: float = 0.0
    fill_time:  str = ''
    status:     str = 'pending'  # 'pending', 'filled', 'rejected'
    cost:       float = 0.0    # transaction cost


@dataclass
class Trade:
    """A completed trade record."""
    timestamp:  str
    symbol:     str
    qty:        float
    side:       str
    fill_price: float
    cost:       float
    hedge_before: float
    hedge_after:  float
    agent_action: int
    hedging_error: float = 0.0
    reason:     str = ''


class Portfolio:
    """
    Tracks positions, P&L, and risk metrics in real time.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.pnl_history: List[dict] = []
        self.risk_budget = RiskBudget(cvar_limit=2.0, alpha=0.05)
        self._daily_returns = []

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices of all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_return_pct(self) -> float:
        return (self.total_value - self.initial_cash) / self.initial_cash * 100

    def record_pnl(self, timestamp: str, snapshot: MarketSnapshot):
        """Record current P&L snapshot."""
        total = self.total_value
        if self.pnl_history:
            prev = self.pnl_history[-1]['total_value']
            daily_ret = (total - prev) / prev if prev > 0 else 0.0
            self._daily_returns.append(daily_ret)
            self.risk_budget.update(daily_ret)

        self.pnl_history.append({
            'timestamp':      timestamp,
            'total_value':    round(total, 2),
            'cash':           round(self.cash, 2),
            'unrealized_pnl': round(self.total_unrealized_pnl, 2),
            'return_pct':     round(self.total_return_pct, 4),
            'spot_price':     snapshot.price,
            'iv':             snapshot.iv,
            'vix':            snapshot.vix,
            'cvar_95':        round(self.current_cvar(), 4),
        })

    def current_cvar(self) -> float:
        if len(self._daily_returns) < 5:
            return 0.0
        return self.risk_budget.current_cvar()

    def current_sharpe(self) -> float:
        if len(self._daily_returns) < 5:
            return 0.0
        arr = np.array(self._daily_returns)
        return float(np.mean(arr) / (np.std(arr) + 1e-10) * np.sqrt(252))

    def get_summary(self) -> dict:
        returns = np.array(self._daily_returns) if self._daily_returns else np.array([0.0])
        return {
            'total_value':    round(self.total_value, 2),
            'cash':           round(self.cash, 2),
            'unrealized_pnl': round(self.total_unrealized_pnl, 2),
            'total_return':   round(self.total_return_pct, 2),
            'sharpe':         round(self.current_sharpe(), 3),
            'cvar_95':        round(self.current_cvar(), 4),
            'n_trades':       len(self.trades),
            'positions':      {s: {'qty': round(p.qty, 4),
                                   'value': round(p.market_value, 2),
                                   'pnl': round(p.unrealized_pnl, 2)}
                               for s, p in self.positions.items()},
        }


class PaperTrader:
    """
    Paper trading engine — connects Quantail agents to real market data.

    Each trading step:
      1. Fetch market snapshot (real or simulated prices)
      2. Build observation vector from market state
      3. Run agents through coordinator -> get action
      4. Translate action to hedge adjustment
      5. Simulate order fill at bid/ask
      6. Update portfolio and risk metrics
      7. Log everything for the dashboard
    """

    def __init__(
        self,
        symbol: str = 'SPY',
        option_strike: Optional[float] = None,
        option_expiry_days: int = 30,
        initial_shares: float = 100.0,    # underlying shares to hedge
        initial_cash: float = 100_000.0,
        spread_bps: float = 2.0,          # bid-ask half-spread in bps
        market_source: str = 'simulated', # 'yahoo', 'simulated'
        polygon_key: str = '',
    ):
        self.symbol = symbol
        self.option_expiry_days = option_expiry_days
        self.initial_shares = initial_shares
        self.spread_bps = spread_bps / 10000

        # Market feed
        self.feed = MarketFeed(symbol, source=market_source,
                               polygon_api_key=polygon_key)

        # Portfolio
        self.portfolio = Portfolio(initial_cash)

        # Trading state
        self.current_hedge = 0.0          # current hedge ratio in [-1, 1]
        self.option_value_prev = None
        self.step_count = 0
        self.session_start = datetime.now()

        # Get initial snapshot
        snap = self.feed.get_snapshot()
        self.option_strike = option_strike or snap.price  # ATM by default
        self.option_value_prev = self._compute_option_value(snap)

        # Agents (loaded/initialized externally)
        self.agents = None
        self.coordinator = None

        # History for dashboard
        self.trade_log: List[Trade] = []
        self.step_log: List[dict] = []

        print(f"\nPaperTrader ready:")
        print(f"  Symbol:   {symbol}")
        print(f"  Strike:   {self.option_strike:.2f}")
        print(f"  Expiry:   {option_expiry_days}d")
        print(f"  Shares:   {initial_shares}")
        print(f"  Cash:     ${initial_cash:,.0f}")
        print(f"  Source:   {self.feed.active_source}")

    def set_agents(self, agents: dict, coordinator):
        """Attach trained agents and coordinator."""
        self.agents = agents
        self.coordinator = coordinator
        print(f"  Agents:   {list(agents.keys())} + Coordinator")

    def _compute_option_value(self, snap: MarketSnapshot) -> float:
        """Compute ATM call option price from current market state."""
        from scipy.stats import norm
        S, K = snap.price, self.option_strike
        sigma = snap.iv
        r = snap.risk_free
        T = max(self.option_expiry_days / 365.0, 1e-6)

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-8)
        d2 = d1 - sigma*np.sqrt(T)
        return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

    def _build_observation(self, snap: MarketSnapshot) -> np.ndarray:
        """
        Build 8-dim observation vector from live market data.
        Matches exactly the HestonEnv observation space.
        """
        from scipy.stats import norm
        S, K = snap.price, self.option_strike
        sigma = snap.iv
        r = snap.risk_free
        T = max((self.option_expiry_days - self.step_count) / 365.0, 1e-6)

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-8)
        d2 = d1 - sigma*np.sqrt(T)

        delta = float(norm.cdf(d1))
        gamma = float(norm.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-8))
        option_value = float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

        # Variance proxy from IV
        v_proxy = (sigma ** 2)
        tau = T / (self.option_expiry_days / 365.0)

        # Running P&L
        pnl_norm = self.portfolio.total_return_pct / 10.0

        return np.array([
            S / K,                              # moneyness
            float(np.clip(v_proxy, 0, 1)),      # variance
            float(np.clip(tau, 0, 1)),          # time to expiry
            float(np.clip(delta, 0, 1)),        # delta
            float(np.clip(gamma * S, 0, 5)),    # scaled gamma
            float(np.clip(self.current_hedge, -1, 1)),  # hedge ratio
            float(np.clip(pnl_norm, -5, 5)),    # normalized pnl
            float(np.clip(len(self.trade_log)/100, 0, 1)),  # activity
        ], dtype=np.float32)

    def _action_to_hedge_change(self, action: int) -> float:
        """Map discrete action [0-10] to hedge adjustment [-0.05, +0.05]."""
        return (action - 5) * 0.01

    def _execute_order(
        self,
        snap: MarketSnapshot,
        new_hedge: float,
        action: int,
        obs: np.ndarray,
    ) -> Trade:
        """
        Simulate order execution with realistic fill model.
        Fill = mid ± (spread/2) depending on buy/sell.
        """
        delta_h = new_hedge - self.current_hedge
        qty = delta_h * self.initial_shares

        if abs(qty) < 0.001:
            return None

        side = 'buy' if qty > 0 else 'sell'

        # Fill price with spread model
        fill_price = snap.price + np.sign(qty) * snap.spread / 2

        # Transaction cost
        cost = abs(qty) * fill_price * self.spread_bps

        # Update portfolio cash
        self.portfolio.cash -= qty * fill_price + cost

        # Update position
        sym = self.symbol
        if sym in self.portfolio.positions:
            pos = self.portfolio.positions[sym]
            total_qty = pos.qty + qty
            if abs(total_qty) < 0.001:
                del self.portfolio.positions[sym]
            else:
                pos.avg_price = (pos.qty*pos.avg_price + qty*fill_price) / total_qty
                pos.qty = total_qty
                pos.current_price = snap.price
        else:
            if abs(qty) >= 0.001:
                self.portfolio.positions[sym] = Position(
                    symbol=sym,
                    qty=qty,
                    avg_price=fill_price,
                    current_price=snap.price,
                )

        # Compute hedging error
        option_value_now = self._compute_option_value(snap)
        dV = option_value_now - (self.option_value_prev or option_value_now)
        dS = 0.0
        if self.step_log:
            dS = snap.price - self.step_log[-1].get('price', snap.price)
        hedging_error = dV - self.current_hedge * dS
        self.option_value_prev = option_value_now

        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol=sym,
            qty=round(qty, 4),
            side=side,
            fill_price=round(fill_price, 2),
            cost=round(cost, 4),
            hedge_before=round(self.current_hedge, 4),
            hedge_after=round(new_hedge, 4),
            agent_action=action,
            hedging_error=round(hedging_error, 6),
            reason='quantail_signal',
        )

        self.current_hedge = new_hedge
        self.trade_log.append(trade)
        self.portfolio.trades.append(trade)
        return trade

    def step(self) -> dict:
        """
        Execute one trading step.
        Returns current state dict for dashboard.
        """
        self.step_count += 1
        snap = self.feed.get_snapshot()

        # Update portfolio prices
        self.portfolio.update_prices({self.symbol: snap.price})

        # Build observation
        obs = self._build_observation(snap)

        # Get agent action
        if self.agents and self.coordinator:
            action, coord_info = self.coordinator.coordinate(
                obs, self.agents, epsilon=0.0
            )
        else:
            # No agents: use delta hedge as fallback
            action = int(np.round((obs[3] - obs[5]) / 0.01 + 5))
            action = int(np.clip(action, 0, 10))
            coord_info = {'unified_cvar': 0.0, 'risk_veto': False}

        # Execute trade
        adj = self._action_to_hedge_change(action)
        new_hedge = float(np.clip(self.current_hedge + adj, -1, 1))
        trade = self._execute_order(snap, new_hedge, action, obs)

        # Record P&L
        self.portfolio.record_pnl(datetime.now().isoformat(), snap)

        # Build step record
        step_data = {
            'step':           self.step_count,
            'timestamp':      datetime.now().isoformat(),
            'price':          snap.price,
            'bid':            snap.bid,
            'ask':            snap.ask,
            'iv':             round(snap.iv * 100, 2),
            'vix':            snap.vix,
            'delta':          round(float(obs[3]), 4),
            'gamma':          round(float(obs[4]) / snap.price, 6),
            'hedge_ratio':    round(new_hedge, 4),
            'action':         action,
            'action_label':   f'{adj:+.0%} hedge',
            'cvar':           round(coord_info.get('unified_cvar', 0.0), 4),
            'risk_veto':      coord_info.get('risk_veto', False),
            'portfolio':      self.portfolio.get_summary(),
            'trade':          {
                'executed':   trade is not None,
                'qty':        round(trade.qty, 4) if trade else 0,
                'price':      trade.fill_price if trade else 0,
                'cost':       round(trade.cost, 4) if trade else 0,
                'hedge_err':  round(trade.hedging_error, 6) if trade else 0,
            },
        }

        self.step_log.append(step_data)
        return step_data

    def run_session(self, n_steps: int = 60, delay_seconds: float = 0) -> dict:
        """
        Run a full trading session.
        For live trading: set delay_seconds to your rebalancing interval.
        For backtesting: delay_seconds=0.
        """
        import time
        print(f"\nStarting trading session: {n_steps} steps")
        print(f"{'─'*55}")
        print(f"{'Step':>5} {'Price':>8} {'Delta':>7} {'Hedge':>7} "
              f"{'Action':>12} {'P&L':>10} {'CVaR':>8}")
        print(f"{'─'*55}")

        for i in range(n_steps):
            state = self.step()
            pf = state['portfolio']
            pnl = pf['unrealized_pnl']
            pnl_sym = '+' if pnl >= 0 else ''

            print(f"{state['step']:>5} "
                  f"${state['price']:>7.2f} "
                  f"{state['delta']:>7.4f} "
                  f"{state['hedge_ratio']:>7.4f} "
                  f"{state['action_label']:>12} "
                  f"{pnl_sym}${pnl:>8.2f} "
                  f"{state['cvar']:>8.4f}"
                  + (' [VETO]' if state['risk_veto'] else ''))

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        summary = self.get_session_summary()
        self._print_session_summary(summary)
        return summary

    def get_session_summary(self) -> dict:
        """Full session summary for dashboard."""
        pf = self.portfolio
        errors = [t.hedging_error for t in self.trade_log]
        costs  = [t.cost for t in self.trade_log]
        returns = [s['portfolio'].get('total_return', s['portfolio'].get('return_pct', 0.0)) for s in self.step_log]

        return {
            'session_id':     self.session_start.isoformat(),
            'symbol':         self.symbol,
            'n_steps':        self.step_count,
            'n_trades':       len(self.trade_log),
            'portfolio':      pf.get_summary(),
            'hedging_rmse':   round(float(np.sqrt(np.mean(np.array(errors)**2))) if errors else 0, 6),
            'total_cost':     round(sum(costs), 4),
            'final_pnl':      round(pf.total_unrealized_pnl, 2),
            'final_return':   round(pf.total_return_pct, 4),
            'sharpe':         round(pf.current_sharpe(), 3),
            'cvar_95':        round(pf.current_cvar(), 4),
            'pnl_history':    pf.pnl_history[-100:],  # last 100 for chart
            'trade_log':      [
                {'time': t.timestamp[:19], 'qty': t.qty, 'price': t.fill_price,
                 'side': t.side, 'cost': t.cost, 'err': t.hedging_error}
                for t in self.trade_log[-20:]  # last 20 trades
            ],
        }

    def _print_session_summary(self, s: dict):
        print(f"\n{'='*55}")
        print(f"  SESSION COMPLETE")
        print(f"{'='*55}")
        print(f"  Steps:          {s['n_steps']}")
        print(f"  Trades:         {s['n_trades']}")
        print(f"  Final P&L:      ${s['final_pnl']:+,.2f}")
        print(f"  Return:         {s['final_return']:+.2f}%")
        print(f"  Sharpe:         {s['sharpe']:.3f}")
        print(f"  CVaR@95%:       {s['cvar_95']:.4f}")
        print(f"  Hedging RMSE:   {s['hedging_rmse']:.6f}")
        print(f"  Total cost:     ${s['total_cost']:.4f}")
        print(f"{'='*55}")
