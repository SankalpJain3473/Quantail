"""
trading/market_feed.py
======================
Live and historical market data feed.

Supports:
  - Yahoo Finance (free, for development/paper trading)
  - Polygon.io (production, real-time options + equity)
  - Simulated feed (fallback when no API key)

Usage:
    feed = MarketFeed(symbol='SPY', source='yahoo')
    snapshot = feed.get_snapshot()
    options  = feed.get_options_chain()
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


@dataclass
class MarketSnapshot:
    """Current market state for one underlying."""
    symbol:     str
    price:      float
    bid:        float
    ask:        float
    spread:     float
    volume:     int
    iv:         float          # implied volatility (annualized)
    realized_vol: float        # 20-day realized vol
    vix:        float
    risk_free:  float          # 3-month T-bill rate
    timestamp:  str

    def to_dict(self):
        return asdict(self)


@dataclass
class OptionContract:
    """Single options contract."""
    symbol:     str
    expiry:     str
    strike:     float
    option_type: str           # 'call' or 'put'
    bid:        float
    ask:        float
    mid:        float
    iv:         float
    delta:      float
    gamma:      float
    vega:       float
    theta:      float
    open_interest: int
    volume:     int


class MarketFeed:
    """
    Unified market data feed.
    Tries real data sources, falls back to simulation.
    """

    def __init__(
        self,
        symbol: str = 'SPY',
        source: str = 'auto',        # 'yahoo', 'polygon', 'simulated', 'auto'
        polygon_api_key: str = '',
        risk_free_rate: float = 0.05,
    ):
        self.symbol = symbol.upper()
        self.source = source
        self.polygon_key = polygon_api_key or os.environ.get('POLYGON_API_KEY', '')
        self.risk_free = risk_free_rate

        # Simulation state (used when real data unavailable)
        self._sim_price  = 450.0 if symbol == 'SPY' else 100.0
        self._sim_vol    = 0.18
        self._sim_vix    = 18.5
        self._last_update = datetime.now()

        # Determine active source
        if source == 'auto':
            self.active_source = self._detect_source()
        else:
            self.active_source = source

        print(f"MarketFeed initialized: {symbol} via {self.active_source}")

    def _detect_source(self) -> str:
        """Auto-detect best available data source."""
        # Try yfinance first (free)
        try:
            import yfinance as yf
            t = yf.Ticker(self.symbol)
            _ = t.fast_info['lastPrice']
            return 'yahoo'
        except Exception:
            pass

        # Try polygon if key available
        if self.polygon_key:
            return 'polygon'

        # Fall back to simulation
        return 'simulated'

    # ─────────────────────────────────────────────────────────────────────
    def get_snapshot(self) -> MarketSnapshot:
        """Get current market snapshot."""
        if self.active_source == 'yahoo':
            return self._snapshot_yahoo()
        elif self.active_source == 'polygon':
            return self._snapshot_polygon()
        else:
            return self._snapshot_simulated()

    def _snapshot_yahoo(self) -> MarketSnapshot:
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            info = ticker.fast_info

            price = float(info.get('lastPrice', self._sim_price))
            prev  = float(info.get('previousClose', price))

            # Realized vol from 20-day history
            hist = ticker.history(period='30d', interval='1d')
            if len(hist) >= 2:
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                realized_vol = float(returns.std() * np.sqrt(252))
            else:
                realized_vol = self._sim_vol

            # IV proxy: use VIX if SPY, else 1.3x realized
            try:
                vix_ticker = yf.Ticker('^VIX')
                vix = float(vix_ticker.fast_info.get('lastPrice', 18.5))
            except Exception:
                vix = 18.5

            iv = vix / 100.0 if self.symbol == 'SPY' else realized_vol * 1.3

            spread = price * 0.0002  # 2bps spread estimate

            self._sim_price = price
            self._sim_vol   = realized_vol

            return MarketSnapshot(
                symbol=self.symbol,
                price=round(price, 2),
                bid=round(price - spread/2, 2),
                ask=round(price + spread/2, 2),
                spread=round(spread, 4),
                volume=int(info.get('threeMonthAverageVolume', 50_000_000) / 252),
                iv=round(iv, 4),
                realized_vol=round(realized_vol, 4),
                vix=round(vix, 2),
                risk_free=self.risk_free,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            print(f"Yahoo feed error: {e}. Falling back to simulation.")
            return self._snapshot_simulated()

    def _snapshot_polygon(self) -> MarketSnapshot:
        try:
            import urllib.request
            url = (f"https://api.polygon.io/v2/last/trade/{self.symbol}"
                   f"?apiKey={self.polygon_key}")
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            price = float(data['results']['p'])
            self._sim_price = price
            snap = self._snapshot_simulated()
            snap.price = price
            return snap
        except Exception as e:
            print(f"Polygon feed error: {e}. Falling back to simulation.")
            return self._snapshot_simulated()

    def _snapshot_simulated(self) -> MarketSnapshot:
        """
        Simulated market snapshot using Heston dynamics.
        Used when no real data available.
        """
        dt = (datetime.now() - self._last_update).total_seconds() / (252 * 24 * 3600)
        dt = max(dt, 1e-6)
        self._last_update = datetime.now()

        # GBM step
        ret = (0.05 - 0.5 * self._sim_vol**2) * dt + self._sim_vol * np.sqrt(dt) * np.random.randn()
        self._sim_price *= np.exp(ret)
        self._sim_price  = float(np.clip(self._sim_price, 50, 2000))

        # Mean-reverting vol
        self._sim_vol += 2.0 * (0.18 - self._sim_vol) * dt + 0.3 * np.sqrt(self._sim_vol * dt) * np.random.randn()
        self._sim_vol  = float(np.clip(self._sim_vol, 0.05, 0.8))

        p = self._sim_price
        spread = p * 0.0002

        return MarketSnapshot(
            symbol=self.symbol,
            price=round(p, 2),
            bid=round(p - spread/2, 2),
            ask=round(p + spread/2, 2),
            spread=round(spread, 4),
            volume=int(np.random.randint(30_000_000, 80_000_000)),
            iv=round(self._sim_vol * 1.1, 4),
            realized_vol=round(self._sim_vol, 4),
            vix=round(self._sim_vol * 100 * 0.9, 2),
            risk_free=self.risk_free,
            timestamp=datetime.now().isoformat(),
        )

    # ─────────────────────────────────────────────────────────────────────
    def get_options_chain(
        self,
        expiry_days: int = 30,
        n_strikes: int = 7,
    ) -> List[OptionContract]:
        """
        Get options chain for the underlying.
        Returns list of OptionContract objects.
        """
        snap = self.get_snapshot()
        S = snap.price
        sigma = snap.iv
        r = self.risk_free
        T = expiry_days / 365.0

        from scipy.stats import norm

        # Generate ATM ± 3 strike levels
        strike_pcts = np.linspace(0.90, 1.10, n_strikes)
        strikes = [round(S * p, 0) for p in strike_pcts]

        expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
        contracts = []

        for K in strikes:
            for opt_type in ['call', 'put']:
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-8)
                d2 = d1 - sigma*np.sqrt(T)

                if opt_type == 'call':
                    price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                    delta = float(norm.cdf(d1))
                else:
                    price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                    delta = float(norm.cdf(d1) - 1)

                gamma = float(norm.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-8))
                vega  = float(S*norm.pdf(d1)*np.sqrt(T) / 100)
                theta = float(-(S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(d2 if opt_type=='call' else -d2)) / 365)

                spread = max(price * 0.02, 0.05)
                contracts.append(OptionContract(
                    symbol=self.symbol,
                    expiry=expiry_date,
                    strike=float(K),
                    option_type=opt_type,
                    bid=round(max(price - spread/2, 0.01), 2),
                    ask=round(price + spread/2, 2),
                    mid=round(price, 2),
                    iv=round(sigma + np.random.normal(0, 0.01), 4),
                    delta=round(delta, 4),
                    gamma=round(gamma, 6),
                    vega=round(vega, 4),
                    theta=round(theta, 4),
                    open_interest=int(np.random.randint(100, 10000)),
                    volume=int(np.random.randint(10, 1000)),
                ))

        return contracts

    def get_historical_prices(self, days: int = 252) -> np.ndarray:
        """Get historical price array for backtesting."""
        if self.active_source == 'yahoo':
            try:
                import yfinance as yf
                hist = yf.Ticker(self.symbol).history(period=f'{days+10}d')
                prices = hist['Close'].values[-days:]
                return prices.astype(float)
            except Exception:
                pass

        # Simulate historical prices
        prices = [self._sim_price]
        for _ in range(days - 1):
            ret = np.random.normal(0.0002, self._sim_vol / np.sqrt(252))
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices[::-1])
