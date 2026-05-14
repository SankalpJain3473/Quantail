// src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { WebSocketMessage } from '../types';

const WS_URL = import.meta.env.VITE_WS_URL || (() => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/trading`;
})();

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const {
    auth, updateStats, addTrade, addPnLPoint, addPricePoint,
    setConnected, setPnLHistory, setPriceHistory, setSessionActive,
  } = useTradingStore();

  const connect = useCallback(() => {
    if (!auth.token) return;
    if (ws.current?.readyState === WebSocket.OPEN) return;

    const url = `${WS_URL}?token=${encodeURIComponent(auth.token)}`;
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');
      // Keep-alive ping every 25s
      const pingInterval = setInterval(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send('ping');
        } else {
          clearInterval(pingInterval);
        }
      }, 25000);
    };

    ws.current.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'connected':
            if (message.data) updateStats(message.data);
            if (message.pnl_history) setPnLHistory(message.pnl_history);
            if (message.price_history) setPriceHistory(message.price_history);
            break;

          case 'stats_update':
            if (message.data) updateStats(message.data);
            if (message.trade) addTrade(message.trade);
            if (message.pnl_point) addPnLPoint(message.pnl_point);
            if (message.price_point) addPricePoint(message.price_point);
            break;

          case 'session_complete':
            setSessionActive(false);
            if (message.data) updateStats(message.data);
            break;

          case 'heartbeat':
            // Server heartbeat — connection healthy
            break;
        }
      } catch (e) {
        // Non-JSON message (e.g. "pong")
      }
    };

    ws.current.onclose = (event) => {
      setConnected(false);
      console.log('WebSocket disconnected', event.code);
      // Reconnect unless deliberately closed (4001 = unauthorized)
      if (event.code !== 4001 && event.code !== 1000) {
        reconnectTimer.current = setTimeout(connect, 3000);
      }
    };

    ws.current.onerror = () => {
      setConnected(false);
    };
  }, [auth.token]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    ws.current?.close(1000, 'User disconnected');
    setConnected(false);
  }, []);

  useEffect(() => {
    if (auth.isAuthenticated) connect();
    return () => disconnect();
  }, [auth.isAuthenticated, connect, disconnect]);

  return { connect, disconnect };
}
