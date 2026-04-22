// src/store/tradingStore.ts
import { create } from 'zustand';
import { SessionStats, Trade, PnLPoint, PricePoint, SessionConfig, AuthState } from '../types';

interface TradingStore {
  // Auth
  auth: AuthState;
  setAuth: (auth: AuthState) => void;
  logout: () => void;

  // Session
  sessionActive: boolean;
  sessionConfig: SessionConfig | null;
  setSessionActive: (active: boolean) => void;
  setSessionConfig: (config: SessionConfig) => void;

  // Live data
  stats: SessionStats | null;
  trades: Trade[];
  pnlHistory: PnLPoint[];
  priceHistory: PricePoint[];
  connected: boolean;

  // Actions
  updateStats: (stats: SessionStats) => void;
  addTrade: (trade: Trade) => void;
  addPnLPoint: (point: PnLPoint) => void;
  addPricePoint: (point: PricePoint) => void;
  setConnected: (connected: boolean) => void;
  setPnLHistory: (history: PnLPoint[]) => void;
  setPriceHistory: (history: PricePoint[]) => void;
  resetSession: () => void;
}

export const useTradingStore = create<TradingStore>((set) => ({
  // Auth
  auth: {
    token: localStorage.getItem('quantail_token'),
    username: localStorage.getItem('quantail_username'),
    isAuthenticated: !!localStorage.getItem('quantail_token'),
  },
  setAuth: (auth) => {
    if (auth.token) {
      localStorage.setItem('quantail_token', auth.token);
      localStorage.setItem('quantail_username', auth.username || '');
    }
    set({ auth });
  },
  logout: () => {
    localStorage.removeItem('quantail_token');
    localStorage.removeItem('quantail_username');
    set({ auth: { token: null, username: null, isAuthenticated: false } });
  },

  // Session
  sessionActive: false,
  sessionConfig: null,
  setSessionActive: (active) => set({ sessionActive: active }),
  setSessionConfig: (config) => set({ sessionConfig: config }),

  // Live data
  stats: null,
  trades: [],
  pnlHistory: [],
  priceHistory: [],
  connected: false,

  // Actions
  updateStats: (stats) => set({ stats }),
  addTrade: (trade) => set((state) => ({
    trades: [trade, ...state.trades].slice(0, 500),
  })),
  addPnLPoint: (point) => set((state) => ({
    pnlHistory: [...state.pnlHistory, point].slice(-500),
  })),
  addPricePoint: (point) => set((state) => ({
    priceHistory: [...state.priceHistory, point].slice(-500),
  })),
  setConnected: (connected) => set({ connected }),
  setPnLHistory: (history) => set({ pnlHistory: history }),
  setPriceHistory: (history) => set({ priceHistory: history }),
  resetSession: () => set({
    stats: null, trades: [], pnlHistory: [], priceHistory: [], sessionActive: false,
  }),
}));
