"""
dashboard/app.py
================
Quantail Trading Dashboard — Flask web server.

Run with:
  python dashboard/app.py

Opens at: http://localhost:5000

Shows:
  - Live P&L chart
  - Current positions and hedge ratios
  - CVaR risk gauge
  - Agent signals and coordinator weights
  - Trade log
  - Greeks (delta, gamma, vega)
"""

import json
import threading
import time
import sys
import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trading.market_feed import MarketFeed
from trading.paper_trader import PaperTrader
from agents.agents import HedgingAgent, RiskAgent, PortfolioAgent, AlphaAgent
from coordinator.wasserstein_coordinator import WassersteinCoordinator

app = Flask(__name__)

# ── Global state ─────────────────────────────────────────────────────────────
trader: PaperTrader = None
session_running = False
session_thread  = None
latest_state    = {}
session_history = []

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Quantail — Trading Dashboard</title>
<meta http-equiv="refresh" content="3">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f13; color: #e0dfd8; font-size: 14px; }

.header { background: #18181f; border-bottom: 1px solid #2a2a35;
          padding: 12px 24px; display: flex; align-items: center;
          justify-content: space-between; }
.logo   { font-size: 20px; font-weight: 700; color: #7f77dd; letter-spacing: -0.5px; }
.logo span { color: #5dcaa5; }
.status-badge { padding: 4px 12px; border-radius: 12px; font-size: 12px;
                font-weight: 600; letter-spacing: 0.03em; }
.live   { background: #0f3d25; color: #5dcaa5; border: 1px solid #0f6e56; }
.paper  { background: #2a2015; color: #ef9f27; border: 1px solid #854f0b; }
.stopped{ background: #2a1515; color: #f09595; border: 1px solid #a32d2d; }

.grid   { display: grid; grid-template-columns: repeat(4, 1fr);
          gap: 12px; padding: 16px 24px; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: 2fr 1fr 1fr; }

.card   { background: #18181f; border: 1px solid #2a2a35;
          border-radius: 10px; padding: 16px; }
.card-title { font-size: 11px; font-weight: 600; color: #888780;
              text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }

.metric { font-size: 28px; font-weight: 700; line-height: 1.1; }
.metric.green  { color: #5dcaa5; }
.metric.red    { color: #f09595; }
.metric.purple { color: #7f77dd; }
.metric.amber  { color: #ef9f27; }
.metric.white  { color: #e0dfd8; }

.sub    { font-size: 12px; color: #888780; margin-top: 4px; }

.pnl-pos { color: #5dcaa5; }
.pnl-neg { color: #f09595; }

.table  { width: 100%; border-collapse: collapse; font-size: 12px; }
.table th { color: #888780; font-weight: 600; text-align: left;
            padding: 6px 8px; border-bottom: 1px solid #2a2a35;
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.table td { padding: 6px 8px; border-bottom: 1px solid #1e1e28; }
.table tr:last-child td { border-bottom: none; }
.table tr:hover td { background: #1e1e28; }

.badge  { display: inline-block; padding: 2px 8px; border-radius: 8px;
          font-size: 11px; font-weight: 600; }
.badge-buy  { background: #0f3d25; color: #5dcaa5; }
.badge-sell { background: #3d1515; color: #f09595; }
.badge-hold { background: #2a2015; color: #ef9f27; }
.badge-veto { background: #3d152a; color: #ed93b1; }

.bar-container { background: #0f0f13; border-radius: 4px;
                 height: 8px; margin-top: 6px; overflow: hidden; }
.bar { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
.bar-green  { background: #0f6e56; }
.bar-red    { background: #a32d2d; }
.bar-purple { background: #534ab7; }
.bar-amber  { background: #854f0b; }

.chart-container { height: 180px; position: relative; }
canvas { width: 100% !important; }

.agent-row { display: flex; align-items: center; gap: 10px;
             padding: 6px 0; border-bottom: 1px solid #1e1e28; }
.agent-row:last-child { border-bottom: none; }
.agent-name { width: 130px; font-size: 12px; color: #c2c0b6; }
.agent-weight { width: 48px; font-size: 12px; font-weight: 600;
                color: #7f77dd; text-align: right; }
.weight-bar-bg { flex: 1; background: #0f0f13; height: 6px;
                  border-radius: 3px; overflow: hidden; }
.weight-bar { height: 100%; background: #534ab7; border-radius: 3px; }

.footer { padding: 12px 24px; font-size: 11px; color: #5f5e5a;
          border-top: 1px solid #2a2a35; text-align: center; }

.section-label { font-size: 11px; font-weight: 700; color: #888780;
                 text-transform: uppercase; letter-spacing: 0.1em;
                 padding: 8px 24px 4px; }

.cvar-gauge { display: flex; align-items: center; gap: 12px; }
.cvar-value { font-size: 32px; font-weight: 700; }
.cvar-limit { font-size: 11px; color: #888780; }

.greek { display: flex; justify-content: space-between;
         padding: 5px 0; border-bottom: 1px solid #1e1e28; }
.greek:last-child { border-bottom: none; }
.greek-name { font-size: 12px; color: #888780; }
.greek-val  { font-size: 13px; font-weight: 600; color: #c2c0b6; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
</head>
<body>

<div class="header">
  <div class="logo">Quant<span>ail</span></div>
  <div style="font-size:13px;color:#888780;">
    Distributional Quantum RL — Dynamic Hedging Platform
  </div>
  <div class="status-badge paper" id="status-badge">PAPER TRADING</div>
</div>

<!-- KPI row -->
<div class="grid" style="padding-top:16px;">

  <div class="card">
    <div class="card-title">Total P&L</div>
    <div class="metric {{ 'green' if pnl >= 0 else 'red' }}" id="pnl">
      {{ '+' if pnl >= 0 else '' }}${{ "%.2f"|format(pnl) }}
    </div>
    <div class="sub">{{ '+' if ret >= 0 else '' }}{{ "%.3f"|format(ret) }}% total return</div>
  </div>

  <div class="card">
    <div class="card-title">CVaR @ 95%</div>
    <div class="metric {{ 'green' if cvar < 0.02 else ('amber' if cvar < 0.04 else 'red') }}" id="cvar">
      {{ "%.4f"|format(cvar) }}
    </div>
    <div class="sub">Limit: 0.03 | {{ 'Within budget' if cvar < 0.03 else 'LIMIT BREACHED' }}</div>
    <div class="bar-container">
      <div class="bar {{ 'bar-green' if cvar < 0.02 else ('bar-amber' if cvar < 0.04 else 'bar-red') }}"
           style="width:{{ [cvar/0.05*100, 100]|min|int }}%;"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Sharpe Ratio</div>
    <div class="metric {{ 'green' if sharpe > 1 else ('amber' if sharpe > 0 else 'red') }}" id="sharpe">
      {{ "%.3f"|format(sharpe) }}
    </div>
    <div class="sub">Annualized | {{ n_trades }} trades | {{ n_steps }} steps</div>
  </div>

  <div class="card">
    <div class="card-title">Hedge Ratio</div>
    <div class="metric purple" id="hedge">{{ "%.4f"|format(hedge) }}</div>
    <div class="sub">Delta: {{ "%.4f"|format(delta) }} | Action: {{ action_label }}</div>
    <div class="bar-container">
      <div class="bar bar-purple"
           style="width:{{ ((hedge+1)/2*100)|int }}%;"></div>
    </div>
  </div>

</div>

<!-- P&L chart + Greeks + Agents -->
<div class="grid grid-3" style="padding-top:0;">

  <div class="card">
    <div class="card-title">Portfolio Value</div>
    <div class="chart-container">
      <canvas id="pnlChart"></canvas>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Greeks</div>
    {% for g in greeks %}
    <div class="greek">
      <span class="greek-name">{{ g.name }}</span>
      <span class="greek-val">{{ g.value }}</span>
    </div>
    {% endfor %}
  </div>

  <div class="card">
    <div class="card-title">Agent Weights (Wasserstein)</div>
    {% for a in agents %}
    <div class="agent-row">
      <div class="agent-name">{{ a.name }}</div>
      <div class="weight-bar-bg">
        <div class="weight-bar" style="width:{{ (a.weight*100)|int }}%;"></div>
      </div>
      <div class="agent-weight">{{ "%.0f"|format(a.weight*100) }}%</div>
    </div>
    {% endfor %}
    <div style="margin-top:10px;font-size:11px;color:#888780;">
      CVaR veto rate: {{ "%.1f"|format(veto_rate*100) }}%
    </div>
  </div>

</div>

<!-- Trade log -->
<div class="section-label">Trade Log</div>
<div style="padding: 0 24px 8px;">
  <div class="card">
    <table class="table">
      <thead>
        <tr>
          <th>Time</th><th>Side</th><th>Qty</th><th>Fill Price</th>
          <th>Hedge Before</th><th>Hedge After</th>
          <th>Hedging Error</th><th>Cost</th>
        </tr>
      </thead>
      <tbody>
        {% for t in trades %}
        <tr>
          <td>{{ t.time }}</td>
          <td><span class="badge badge-{{ t.side }}">{{ t.side.upper() }}</span></td>
          <td>{{ "%.3f"|format(t.qty|abs) }}</td>
          <td>${{ "%.2f"|format(t.price) }}</td>
          <td>{{ "%.4f"|format(t.hedge_before) }}</td>
          <td>{{ "%.4f"|format(t.hedge_after) }}</td>
          <td class="{{ 'pnl-neg' if t.err|abs > 0.01 else '' }}">{{ "%.6f"|format(t.err) }}</td>
          <td>${{ "%.4f"|format(t.cost) }}</td>
        </tr>
        {% endfor %}
        {% if not trades %}
        <tr><td colspan="8" style="text-align:center;color:#5f5e5a;padding:20px;">
          No trades yet — waiting for first signal...
        </td></tr>
        {% endif %}
      </tbody>
    </table>
  </div>
</div>

<div class="footer">
  Quantail PoC — Sankalp Jain & Veronica Koval | Columbia University, New York |
  Step {{ n_steps }} | Last update: {{ timestamp }}
</div>

<script>
const pnlData = {{ pnl_history|tojson }};
const labels  = pnlData.map((_, i) => i);
const values  = pnlData.map(d => d.total_value || 100000);

const ctx = document.getElementById('pnlChart').getContext('2d');
new Chart(ctx, {
  type: 'line',
  data: {
    labels: labels,
    datasets: [{
      data: values,
      borderColor: values[values.length-1] >= values[0] ? '#5dcaa5' : '#f09595',
      borderWidth: 2,
      fill: true,
      backgroundColor: values[values.length-1] >= values[0]
        ? 'rgba(93,202,165,0.08)' : 'rgba(240,149,149,0.08)',
      pointRadius: 0,
      tension: 0.4,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        grid: { color: '#1e1e28' },
        ticks: { color: '#888780', font: { size: 10 },
                 callback: v => '$' + v.toLocaleString() }
      }
    }
  }
});
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def dashboard():
    global latest_state, session_history, trader
    s = latest_state
    pf = s.get('portfolio', {})
    trades_raw = s.get('trade_log', [])

    pnl    = pf.get('unrealized_pnl', 0.0)
    ret    = pf.get('total_return', 0.0)
    cvar_v = pf.get('cvar_95', 0.0)
    sharpe = pf.get('sharpe', 0.0)
    hedge  = s.get('hedge_ratio', 0.0)
    delta  = s.get('delta', 0.0)
    n_steps= s.get('step', 0)
    n_trades= pf.get('n_trades', 0)
    action_label = s.get('action_label', '—')
    veto_rate = s.get('veto_rate', 0.0)

    greeks = [
        {'name': 'Delta (Δ)',  'value': f"{delta:.4f}"},
        {'name': 'Gamma (Γ)',  'value': f"{s.get('gamma', 0):.6f}"},
        {'name': 'IV',         'value': f"{s.get('iv', 0):.1f}%"},
        {'name': 'VIX',        'value': f"{s.get('vix', 0):.1f}"},
        {'name': 'Spot',       'value': f"${s.get('price', 0):.2f}"},
        {'name': 'Bid/Ask',    'value': f"${s.get('bid',0):.2f} / ${s.get('ask',0):.2f}"},
    ]

    # Coordinator weights
    coord_weights = {'HedgingAgent': 0.40, 'RiskAgent': 0.30,
                     'PortfolioAgent': 0.20, 'AlphaAgent': 0.10}
    if trader and trader.coordinator:
        coord_weights = trader.coordinator.weights

    agents_display = [
        {'name': k.replace('Agent',''), 'weight': v}
        for k, v in coord_weights.items()
    ]

    pnl_history = []
    if trader:
        pnl_history = trader.portfolio.pnl_history[-100:]

    return render_template_string(
        DASHBOARD_HTML,
        pnl=pnl, ret=ret, cvar=cvar_v, sharpe=sharpe,
        hedge=hedge, delta=delta, n_steps=n_steps,
        n_trades=n_trades, action_label=action_label,
        greeks=greeks, agents=agents_display,
        veto_rate=veto_rate,
        trades=trades_raw[-15:],
        pnl_history=pnl_history,
        timestamp=datetime.now().strftime('%H:%M:%S'),
    )


@app.route('/api/state')
def api_state():
    """JSON endpoint for programmatic access."""
    return jsonify(latest_state)


@app.route('/api/summary')
def api_summary():
    """Full session summary."""
    if trader:
        return jsonify(trader.get_session_summary())
    return jsonify({'error': 'No active session'})


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start a trading session."""
    global trader, session_running, session_thread
    if session_running:
        return jsonify({'status': 'already running'})

    data = request.get_json() or {}
    symbol = data.get('symbol', 'SPY')
    n_steps = int(data.get('n_steps', 60))
    source  = data.get('source', 'simulated')

    def run():
        global session_running, latest_state
        session_running = True
        try:
            _run_session(symbol=symbol, n_steps=n_steps, source=source)
        finally:
            session_running = False

    session_thread = threading.Thread(target=run, daemon=True)
    session_thread.start()
    return jsonify({'status': 'started', 'symbol': symbol, 'n_steps': n_steps})


def _run_session(symbol='SPY', n_steps=60, source='simulated'):
    """Internal: run trading session and update global state."""
    global trader, latest_state

    np.random.seed(42)
    agents = {
        'HedgingAgent':   HedgingAgent(seed=42),
        'RiskAgent':      RiskAgent(seed=43),
        'PortfolioAgent': PortfolioAgent(seed=44),
        'AlphaAgent':     AlphaAgent(seed=45),
    }

    # Quick training (30 episodes)
    from envs.heston_env import HestonEnv
    from coordinator.wasserstein_coordinator import WassersteinCoordinator

    coord = WassersteinCoordinator(
        agent_names=list(agents.keys()),
        weights={'HedgingAgent':0.40,'RiskAgent':0.30,
                 'PortfolioAgent':0.20,'AlphaAgent':0.10}
    )

    train_env = HestonEnv(n_steps=30)
    for ep in range(30):
        eps = max(0.05, 0.5*(1-ep/30))
        obs, _ = train_env.reset(); done=False
        while not done:
            action, _ = coord.coordinate(obs, agents, eps)
            obs_n, _, term, trunc, info = train_env.step(action)
            done = term or trunc
            for agent in agents.values():
                r = agent.compute_reward(info)
                agent.store(obs, action, r, obs_n, done)
            obs = obs_n
        for agent in agents.values():
            agent.end_episode(); agent.update(16)

    # Create paper trader
    trader = PaperTrader(symbol=symbol, market_source=source)
    trader.set_agents(agents, coord)

    # Run steps
    for _ in range(n_steps):
        state = trader.step()
        state['trade_log'] = [
            {'time': t.timestamp[:19], 'qty': t.qty, 'price': t.fill_price,
             'side': t.side, 'cost': t.cost, 'err': t.hedging_error,
             'hedge_before': t.hedge_before, 'hedge_after': t.hedge_after}
            for t in trader.trade_log[-15:]
        ]
        state['veto_rate'] = coord.get_coordination_stats().get('veto_rate', 0.0)
        latest_state = state
        time.sleep(0.5)  # 500ms between steps for demo


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global session_running, session_thread

    print("=" * 55)
    print("  QUANTAIL TRADING DASHBOARD")
    print("  http://localhost:5000")
    print("=" * 55)

    # Auto-start a simulated session in background
    def auto_start():
        time.sleep(1)
        _run_session(symbol='SPY', n_steps=120, source='simulated')

    t = threading.Thread(target=auto_start, daemon=True)
    t.start()

    app.run(debug=False, port=5000, use_reloader=False)


if __name__ == '__main__':
    main()
