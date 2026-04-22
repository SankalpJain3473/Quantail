// src/types/index.ts
// All TypeScript types for Quantail frontend

export interface SessionStats {
  step: number;
  total_pnl: number;
  return_pct: number;
  cvar_95: number;
  sharpe: number;
  sortino: number;
  hedge_ratio: number;
  delta: number;
  gamma: number;
  iv: number;
  spot_price: number;
  bid: number;
  ask: number;
  n_trades: number;
  total_cost: number;
  hedging_rmse: number;
  veto_rate: number;
  agent_weights: Record<string, number>;
  data_source: string;
  timestamp: string;
}

export interface Trade {
  id: number;
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  fill_price: number;
  hedge_before: number;
  hedge_after: number;
  hedging_error: number;
  cost: number;
  cvar: number;
  pnl: number;
  source: string;
  reason: string;
}

export interface PnLPoint {
  step: number;
  total_value: number;
  pnl: number;
  timestamp: string;
}

export interface PricePoint {
  step: number;
  price: number;
  hedge: number;
}

export interface SessionConfig {
  symbol: string;
  mode: 'simulated' | 'paper' | 'live';
  n_steps: number;
  speed_ms: number;
  initial_capital: number;
}

export interface WebSocketMessage {
  type: 'stats_update' | 'connected' | 'session_complete' | 'heartbeat';
  data: SessionStats | null;
  trade?: Trade | null;
  pnl_point?: PnLPoint;
  price_point?: PricePoint;
  pnl_history?: PnLPoint[];
  price_history?: PricePoint[];
  timestamp?: string;
}

export interface AuthState {
  token: string | null;
  username: string | null;
  isAuthenticated: boolean;
}
