"""
backend/main.py
===============
Quantail FastAPI Backend — Complete Server

Features:
  - JWT auth (access + refresh tokens, bcrypt passwords)
  - Invite-only registration (admin generates codes)
  - Role-based access (admin / researcher / viewer)
  - SQLite (dev) or PostgreSQL (prod) via SQLAlchemy
  - WebSocket real-time trading updates
  - Full trade audit log persisted to database
  - CORS, rate limiting, input validation

Run:
  uvicorn backend.main:app --reload --port 8000

Default admin on first boot:
  Username: admin  (set ADMIN_USERNAME env var to change)
  Password: Quantail2026!  (set ADMIN_PASSWORD env var to change)
"""

import os
import json
import asyncio
import numpy as np
import httpx
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
from datetime import datetime
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, Depends, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
import uvicorn

from backend.db.database import init_db, get_db, SessionLocal
from backend.db.models import (
    User, UserRole, TradingSession, TradeRecord,
    SessionMode, TradeSide,
)
from backend.auth.service import (
    authenticate_user, create_user, get_user_by_id,
    create_access_token, create_refresh_token, decode_token,
    update_last_login, generate_invite_code,
    validate_invite_code, use_invite_code, create_admin_user,
    validate_password_strength, get_user_by_username, get_user_by_email,
)
from backend.agent_manager import agent_manager

# ── Config ─────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:5173"
).split(",")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")


# ── Pydantic schemas ────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username:    str   = Field(min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email:       str   = Field(min_length=5, max_length=255)
    password:    str   = Field(min_length=8, max_length=100)
    full_name:   str   = Field(default="", max_length=100)
    invite_code: str   = Field(min_length=10, max_length=200)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=1, max_length=100)


class RefreshRequest(BaseModel):
    refresh_token: str


class SessionConfig(BaseModel):
    symbol:          str   = Field(default="SPY", pattern="^[A-Z]{1,5}$")
    mode:            str   = Field(default="simulated")
    n_steps:         int   = Field(default=60, ge=10, le=500)
    speed_ms:        int   = Field(default=1000, ge=100, le=10000)
    initial_capital: float = Field(default=100000.0, ge=1000, le=10_000_000)

    @validator("mode")
    def mode_valid(cls, v):
        if v not in ("simulated", "paper", "live"):
            raise ValueError("mode must be simulated, paper, or live")
        return v


class InviteCreateRequest(BaseModel):
    note:         str = Field(default="", max_length=200)
    expires_days: int = Field(default=7, ge=1, le=30)


class SessionStats(BaseModel):
    step:          int
    total_pnl:     float
    return_pct:    float
    cvar_95:       float
    sharpe:        float
    sortino:       float
    hedge_ratio:   float
    delta:         float
    gamma:         float
    iv:            float
    spot_price:    float
    bid:           float
    ask:           float
    n_trades:      int
    total_cost:    float
    hedging_rmse:  float
    veto_rate:     float
    agent_weights: Dict[str, float]
    data_source:   str
    timestamp:     str


# ── Auth dependency ─────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def ws_get_user(token: str, db: Session) -> Optional[User]:
    if not token:
        return None
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None
    return get_user_by_id(db, payload["sub"])


# ── Trading Engine ──────────────────────────────────────────────────────────
class TradingEngine:
    """
    Per-user trading engine. Each user gets their own isolated session state.
    Runs Heston SDE simulation with multi-agent Quantail logic.
    """

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def get_session(self, user_id: str) -> dict:
        if user_id not in self.sessions:
            self.sessions[user_id] = self._blank_session()
        return self.sessions[user_id]

    def _blank_session(self) -> dict:
        return {
            "active": False, "config": None, "stats": None, "task": None,
            "ws_clients": [], "S": 450.0, "v": 0.04, "hedge": 0.0,
            "cash": 100000.0, "pos_qty": 0.0, "veto_count": 0,
            "total_cost": 0.0, "hedge_errors": [], "daily_returns": [],
            "opt_prev": None, "trade_id": 0, "step": 0,
            "pnl_history": [], "price_history": [],
            "weights": {
                "HedgingAgent": 0.40, "RiskAgent": 0.30,
                "PortfolioAgent": 0.20, "AlphaAgent": 0.10,
            },
            # Delta hedge baseline tracking (for comparison)
            "dh_cash": 0.0, "dh_pos_qty": 0.0, "dh_hedge": 0.0,
            "dh_pnl_history": [],
        }

    def reset_session(self, user_id: str, config: SessionConfig):
        s = self.get_session(user_id)
        s.update({
            "active": True, "config": config, "stats": None,
            "S": 450.0 if config.symbol in ("SPY", "QQQ") else 180.0,
            "K": 450.0 if config.symbol in ("SPY", "QQQ") else 180.0,
            "v": 0.04, "hedge": 0.0, "cash": config.initial_capital,
            "pos_qty": 0.0, "veto_count": 0, "total_cost": 0.0,
            "hedge_errors": [], "daily_returns": [], "opt_prev": None,
            "trade_id": 0, "step": 0, "pnl_history": [], "price_history": [],
            "weights": {
                "HedgingAgent": 0.40, "RiskAgent": 0.30,
                "PortfolioAgent": 0.20, "AlphaAgent": 0.10,
            },
            # Delta hedge baseline tracking (for comparison)
            "dh_cash": 0.0, "dh_pos_qty": 0.0, "dh_hedge": 0.0,
            "dh_pnl_history": [],
        })

    # ── Math helpers ─────────────────────────────────────────────────────
    def _randn(self):
        u, v = np.random.random(), np.random.random()
        return np.sqrt(-2 * np.log(max(u, 1e-10))) * np.cos(2 * np.pi * v)

    def _ncdf(self, x):
        t = 1 / (1 + 0.2316419 * abs(x))
        d = 0.3989423 * np.exp(-x * x / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
        return 1 - p if x > 0 else p

    def _npdf(self, x):
        return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)

    def _bs_call(self, S, K, sig, tau):
        if tau < 1e-6: return max(S - K, 0)
        d1 = (np.log(S / K) + (0.05 + .5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
        d2 = d1 - sig * np.sqrt(tau)
        return S * self._ncdf(d1) - K * np.exp(-0.05 * tau) * self._ncdf(d2)

    def _bs_delta(self, S, K, sig, tau):
        if tau < 1e-6: return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (0.05 + .5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
        return self._ncdf(d1)

    def _bs_gamma(self, S, K, sig, tau):
        if tau < 1e-6: return 0.0
        d1 = (np.log(S / K) + (0.05 + .5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
        return self._npdf(d1) / (S * sig * np.sqrt(tau))

    def _cvar(self, arr, alpha=0.05):
        if len(arr) < 3: return 0.0
        s = sorted(arr)
        n = max(1, int(alpha * len(s)))
        return -float(np.mean(s[:n]))

    def _sharpe(self, arr):
        if len(arr) < 3: return 0.0
        a = np.array(arr)
        return float(np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(252))

    def _sortino(self, arr):
        if len(arr) < 3: return 0.0
        a = np.array(arr)
        d = a[a < 0]
        return float(np.mean(a) / (np.std(d) + 1e-10) * np.sqrt(252)) if len(d) else 99.0

    # ── One trading step ──────────────────────────────────────────────────
    async def step(
        self, sess: dict, db: Session,
        user_id: str, session_db_id: str,
    ) -> SessionStats:
        sess["step"] += 1

        # Heston SDE step
        dt = 1 / 252 / 8
        vp = max(sess["v"], 0)
        z1, z2 = self._randn(), self._randn()
        sess["S"] *= np.exp((0.05 - .5 * vp) * dt + np.sqrt(vp * dt) * z1)
        sess["v"] = max(
            sess["v"] + 2.0 * (0.04 - vp) * dt
            + 0.3 * np.sqrt(vp * dt) * (-0.7 * z1 + np.sqrt(0.51) * z2),
            0,
        )
        sess["S"] = float(np.clip(sess["S"], 10, 5000))

        spot  = sess["S"]
        K     = spot * 0.998
        tau   = max((30 - sess["step"] % 30) / 365, 1e-6)
        sigma = np.sqrt(max(sess["v"], 1e-6))
        spread = spot * 0.0002

        delta   = self._bs_delta(spot, K, sigma, tau)
        gamma   = self._bs_gamma(spot, K, sigma, tau)
        opt_val = self._bs_call(spot, K, sigma, tau)

        cvar_est   = self._cvar(sess["daily_returns"][-20:])
        pnl_trend  = (
            sess["pnl_history"][-1]["total_value"] - sess["pnl_history"][-2]["total_value"]
            if len(sess["pnl_history"]) > 1 else 0
        )

        # ── REAL VQC DECISION ───────────────────────────────────────────────
        # Build 10-dim observation vector for quantum policy network
        # Matches HestonEnv obs space exactly

        # Realized volatility (rolling)
        if "price_hist_vol" not in sess:
            sess["price_hist_vol"] = []
            sess["peak_val"] = 1.0
        sess["price_hist_vol"].append(spot)
        if len(sess["price_hist_vol"]) > 21:
            sess["price_hist_vol"] = sess["price_hist_vol"][-21:]
        if len(sess["price_hist_vol"]) >= 2:
            lr = np.diff(np.log(np.maximum(sess["price_hist_vol"], 1e-8)))
            realized_vol = float(np.std(lr) * np.sqrt(252))
        else:
            realized_vol = np.sqrt(max(sess["v"], 1e-8))

        # Drawdown from peak
        total_now = sess["cash"] + sess["pos_qty"] * spot
        pv_norm = total_now / sess["config"].initial_capital
        if pv_norm > sess["peak_val"]:
            sess["peak_val"] = pv_norm
        drawdown = float(np.clip((sess["peak_val"] - pv_norm) / (sess["peak_val"] + 1e-8), 0, 1))

        # Regime vol and jump features (from hybrid model)
        regime_vol    = float(np.clip(np.sqrt(max(sess["v"],0)) / 0.5, 0, 1))
        if "jump_window" not in sess: sess["jump_window"] = []
        jump_intensity = float(np.mean(sess["jump_window"])) if sess["jump_window"] else 0.0
        if "time_in_regime" not in sess: sess["time_in_regime"] = 0
        time_in_regime = float(np.clip(sess["time_in_regime"] / 20.0, 0, 1))

        obs = np.array([
            spot / sess["K"],                                                       # [0] moneyness
            float(np.clip(sess["v"], 0, 1)),                                       # [1] variance
            float(np.clip(tau / (30/365), 0, 1)),                                   # [2] time
            float(np.clip(delta, 0, 1)),                                            # [3] delta
            float(np.clip(gamma * spot, 0, 5)),                                     # [4] gamma
            float(np.clip(sess["hedge"], -1, 1)),                                  # [5] hedge
            float(np.clip(pnl_trend / (sess["config"].initial_capital * 0.01), -5, 5)), # [6] pnl
            float(np.clip(sess["total_cost"] / (sess["config"].initial_capital * 0.05), 0, 1)), # [7] cost
            float(np.clip(realized_vol, 0, 2)),                                     # [8] realized_vol
            float(drawdown),                                                        # [9] drawdown
            float(np.clip(jump_intensity * 10, 0, 1)),                              # [10] jump intensity
            float(regime_vol),                                                      # [11] regime vol
            float(time_in_regime),                                                  # [12] time in regime
        ], dtype=np.float32)

        # Ask trained VQC agents via Wasserstein coordinator
        agent_sess = agent_manager.get_or_create(user_id)
        action, coord_info = agent_sess.decide(obs)

        # Map discrete action [0-10] -> hedge adjustment [-0.05, +0.05]
        clipped = float((action - 5) * 0.01)
        veto    = coord_info.get("risk_veto", False)
        if veto:
            sess["veto_count"] += 1

        # Update coordinator weights in session for dashboard display
        sess["weights"] = dict(agent_sess.coordinator.weights)

        new_hedge = float(np.clip(sess["hedge"] + clipped, -1, 1))
        trade_data  = None
        dH = new_hedge - sess["hedge"]

        if abs(dH) > 0.001 and not veto:
            qty        = dH * 100
            fill       = spot + np.sign(qty) * spread / 2
            cost       = abs(qty) * fill * 0.0002
            sess["cash"]       -= qty * fill + cost
            sess["pos_qty"]    += qty
            sess["total_cost"] += cost
            sess["trade_id"]   += 1

            hedge_err = 0.0
            if sess["opt_prev"] is not None and len(sess["price_history"]) > 1:
                dV = opt_val - sess["opt_prev"]
                dS = spot - sess["price_history"][-1]["price"]
                hedge_err = dV - sess["hedge"] * dS
            sess["hedge_errors"].append(hedge_err)

            # Online learning — agents improve from actual trade outcome
            reward_signal = -(hedge_err ** 2) - cost * 0.01
            agent_sess = agent_manager.get_or_create(user_id)
            agent_sess.learn_online(obs, reward_signal, obs, False)

            total_val = sess["cash"] + sess["pos_qty"] * spot
            pnl_now   = total_val - sess["config"].initial_capital

            # Persist trade to database
            tr = TradeRecord(
                session_id=session_db_id,
                user_id=user_id,
                trade_seq=sess["trade_id"],
                symbol=sess["config"].symbol,
                side=TradeSide.BUY if qty > 0 else TradeSide.SELL,
                qty=round(abs(qty), 4),
                fill_price=round(fill, 2),
                hedge_before=round(sess["hedge"], 4),
                hedge_after=round(new_hedge, 4),
                hedging_error=round(hedge_err, 6),
                transaction_cost=round(cost, 4),
                cvar_at_trade=round(cvar_est, 4),
                pnl_at_trade=round(pnl_now, 2),
                data_source=sess["config"].mode,
                reason="quantail-signal",
            )
            db.add(tr)
            db.commit()

            trade_data = {
                "id": tr.id, "timestamp": tr.executed_at.isoformat(),
                "symbol": tr.symbol, "side": tr.side.value,
                "qty": tr.qty, "fill_price": tr.fill_price,
                "hedge_before": tr.hedge_before, "hedge_after": tr.hedge_after,
                "hedging_error": tr.hedging_error, "cost": tr.transaction_cost,
                "cvar": tr.cvar_at_trade, "pnl": tr.pnl_at_trade,
                "source": tr.data_source, "reason": tr.reason,
            }
            sess["hedge"] = new_hedge

        sess["opt_prev"] = opt_val

        total_val = sess["cash"] + sess["pos_qty"] * spot
        pnl = total_val - sess["config"].initial_capital
        ret = pnl / sess["config"].initial_capital

        if sess["pnl_history"]:
            pv = sess["pnl_history"][-1]["total_value"]
            if pv > 0:
                sess["daily_returns"].append((total_val - pv) / pv)
        if len(sess["daily_returns"]) > 100:
            sess["daily_returns"] = sess["daily_returns"][-100:]

        sess["pnl_history"].append({
            "step": sess["step"], "total_value": round(total_val, 2),
            "pnl": round(pnl, 2), "timestamp": datetime.utcnow().isoformat(),
        })
        sess["price_history"].append({
            "step": sess["step"], "price": round(spot, 2),
            "hedge": round(sess["hedge"], 4),
        })
        if len(sess["pnl_history"]) > 500:
            sess["pnl_history"] = sess["pnl_history"][-500:]
        if len(sess["price_history"]) > 500:
            sess["price_history"] = sess["price_history"][-500:]

        # Slowly adapt agent weights
        if sess["step"] % 20 == 0:
            n = 0.015
            sess["weights"]["HedgingAgent"]  = float(np.clip(sess["weights"]["HedgingAgent"]  + (np.random.random() - .5) * n, .15, .60))
            sess["weights"]["RiskAgent"]     = float(np.clip(sess["weights"]["RiskAgent"]     + (np.random.random() - .5) * n, .15, .45))
            rest = 1 - sess["weights"]["HedgingAgent"] - sess["weights"]["RiskAgent"]
            sess["weights"]["PortfolioAgent"] = max(.05, rest * .6)
            sess["weights"]["AlphaAgent"]     = max(.05, rest * .4)

        rmse = float(np.sqrt(np.mean(np.array(sess["hedge_errors"]) ** 2))) if sess["hedge_errors"] else 0.0

        stats = SessionStats(
            step=sess["step"], total_pnl=round(pnl, 2),
            return_pct=round(ret * 100, 4),
            cvar_95=round(cvar_est, 4),
            sharpe=round(self._sharpe(sess["daily_returns"]), 3),
            sortino=round(self._sortino(sess["daily_returns"]), 3),
            hedge_ratio=round(sess["hedge"], 4), delta=round(delta, 4),
            gamma=round(gamma * spot, 6), iv=round(sigma * 100, 2),
            spot_price=round(spot, 2),
            bid=round(spot - spread / 2, 2), ask=round(spot + spread / 2, 2),
            n_trades=sess["trade_id"], total_cost=round(sess["total_cost"], 4),
            hedging_rmse=round(rmse, 6),
            veto_rate=round(sess["veto_count"] / max(sess["step"], 1) * 100, 2),
            agent_weights=dict(sess["weights"]),
            data_source=sess["config"].mode,
            timestamp=datetime.utcnow().isoformat(),
        )
        sess["stats"] = stats

        # ── Delta hedge baseline (runs in parallel for comparison) ──────
        dh_initial = sess["config"].initial_capital
        dh_new_hedge = float(np.clip(delta, -1, 1))  # delta hedge = set hedge = delta
        dh_dH = dh_new_hedge - sess["dh_hedge"]
        if abs(dh_dH) > 0.001:
            dh_qty  = dh_dH * 100
            dh_fill = spot + np.sign(dh_qty) * spread / 2
            sess["dh_cash"]    -= dh_qty * dh_fill + abs(dh_qty) * dh_fill * 0.0002
            sess["dh_pos_qty"] += dh_qty
            sess["dh_hedge"]    = dh_new_hedge
        dh_total = sess["dh_cash"] + sess["dh_pos_qty"] * spot
        dh_pnl   = round(dh_total - dh_initial, 2)
        sess["dh_pnl_history"].append({"step": sess["step"], "pnl": dh_pnl})
        if len(sess["dh_pnl_history"]) > 500:
            sess["dh_pnl_history"] = sess["dh_pnl_history"][-500:]

        # Broadcast to all WebSocket clients of this user
        message = json.dumps({
            "type": "stats_update",
            "data": stats.dict(),
            "trade": trade_data,
            "pnl_point":   sess["pnl_history"][-1],
            "price_point": sess["price_history"][-1],
            "dh_pnl_point": sess["dh_pnl_history"][-1] if sess["dh_pnl_history"] else None,
        })
        dead = []
        for ws in sess["ws_clients"]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in sess["ws_clients"]:
                sess["ws_clients"].remove(ws)

        return stats

    async def run_session(self, user_id: str, session_db_id: str, db: Session):
        sess = self.sessions[user_id]
        while sess["active"] and sess["step"] < sess["config"].n_steps:
            await self.step(sess, db, user_id, session_db_id)
            await asyncio.sleep(sess["config"].speed_ms / 1000)
        sess["active"] = False

        # Persist final session stats to database
        if sess.get("stats"):
            s = sess["stats"]
            try:
                db_sess = db.query(TradingSession).filter(
                    TradingSession.id == session_db_id
                ).first()
                if db_sess:
                    total_val = sess["cash"] + sess["pos_qty"] * s.spot_price
                    db_sess.final_pnl    = round(total_val - sess["config"].initial_capital, 2)
                    db_sess.final_return = round(s.return_pct, 4)
                    db_sess.cvar_95      = round(s.cvar_95, 4)
                    db_sess.sharpe       = round(s.sharpe, 3)
                    db_sess.sortino      = round(s.sortino, 3)
                    db_sess.hedging_rmse = round(s.hedging_rmse, 6)
                    db_sess.total_cost   = round(s.total_cost, 4)
                    db_sess.n_trades     = s.n_trades
                    db_sess.veto_rate    = round(s.veto_rate, 2)
                    db_sess.is_active    = False
                    db_sess.ended_at     = datetime.utcnow()
                    db.commit()
                    print(f"Session {session_db_id[:8]}... saved: P&L=${db_sess.final_pnl:+,.2f}")
            except Exception as e:
                print(f"Could not save session stats: {e}")

        for ws in sess["ws_clients"]:
            try:
                await ws.send_text(json.dumps({
                    "type": "session_complete",
                    "data": sess["stats"].dict() if sess["stats"] else {},
                }))
            except Exception:
                pass


engine = TradingEngine()


# ── App lifecycle ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database and create admin user on first boot
    init_db()
    db = SessionLocal()
    try:
        create_admin_user(db)
    finally:
        db.close()

    # Pre-train VQC agents in background (non-blocking)
    asyncio.create_task(agent_manager.startup_pretrain())
    yield


app = FastAPI(
    title="Quantail API",
    description="Distributional Quantum RL Trading Platform — Sankalp Jain & Veronica Koval",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if ENVIRONMENT == "development" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)

if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["quantail.ai", "*.quantail.ai", "localhost", "*.railway.app", "*.up.railway.app"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", status_code=201)
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user — requires a valid invite code."""
    valid, reason, invite = validate_invite_code(db, req.invite_code)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invite code: {reason}")

    if get_user_by_username(db, req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    ok, msg = validate_password_strength(req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    user = create_user(
        db, req.username, req.email, req.password, req.full_name,
        invited_by_id=invite.created_by,
    )
    use_invite_code(db, invite, user.id)

    return {
        "access_token":  create_access_token(user.id, user.username, user.role.value),
        "refresh_token": create_refresh_token(user.id),
        "token_type":    "bearer",
        "user": {
            "id": user.id, "username": user.username,
            "email": user.email, "role": user.role.value,
        },
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username + password."""
    user = authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    update_last_login(db, user)
    return {
        "access_token":  create_access_token(user.id, user.username, user.role.value),
        "refresh_token": create_refresh_token(user.id),
        "token_type":    "bearer",
        "user": {
            "id": user.id, "username": user.username,
            "email": user.email, "full_name": user.full_name,
            "role": user.role.value,
        },
    }


@app.post("/api/auth/refresh")
async def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    """Get new access token using refresh token."""
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "access_token": create_access_token(user.id, user.username, user.role.value),
        "token_type": "bearer",
    }


@app.get("/api/auth/me")
async def me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user profile."""
    return {
        "id":           current_user.id,
        "username":     current_user.username,
        "email":        current_user.email,
        "full_name":    current_user.full_name,
        "role":         current_user.role.value,
        "is_verified":  current_user.is_verified,
        "registered_at": current_user.registered_at.isoformat(),
        "last_login":   current_user.last_login.isoformat() if current_user.last_login else None,
    }


# ── Admin routes ────────────────────────────────────────────────────────────
@app.post("/api/admin/invites", status_code=201)
async def create_invite(
    req: InviteCreateRequest,
    admin: User    = Depends(require_admin),
    db: Session    = Depends(get_db),
):
    """Generate an invite code. Admin only."""
    invite = generate_invite_code(db, admin.id, req.note, req.expires_days)
    return {
        "code":       invite.code,
        "note":       invite.note,
        "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
        "created_by": admin.username,
        "register_url": f"/register?invite={invite.code}",
    }


@app.get("/api/admin/invites")
async def list_invites(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    from backend.db.models import Invitation
    invites = (db.query(Invitation)
               .filter(Invitation.created_by == admin.id)
               .order_by(Invitation.created_at.desc()).all())
    return {"invites": [
        {"code_preview": i.code[:8] + "...", "note": i.note,
         "is_used": i.is_used,
         "expires_at": i.expires_at.isoformat() if i.expires_at else None,
         "used_at": i.used_at.isoformat() if i.used_at else None}
        for i in invites
    ]}


@app.get("/api/admin/users")
async def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.registered_at.desc()).all()
    return {"users": [
        {"id": u.id, "username": u.username, "email": u.email,
         "role": u.role.value, "is_active": u.is_active,
         "registered_at": u.registered_at.isoformat(),
         "last_login": u.last_login.isoformat() if u.last_login else None}
        for u in users
    ]}


# ── Session routes ──────────────────────────────────────────────────────────
@app.post("/api/session/start")
async def start_session(
    config: SessionConfig,
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    sess = engine.get_session(current_user.id)
    if sess["active"]:
        raise HTTPException(status_code=409, detail="Session already active. Stop it first.")

    engine.reset_session(current_user.id, config)

    # Create DB record
    db_session = TradingSession(
        user_id=current_user.id,
        symbol=config.symbol,
        mode=SessionMode(config.mode),
        initial_capital=config.initial_capital,
        n_steps=config.n_steps,
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    # Run in background
    async def run():
        bg_db = SessionLocal()
        try:
            await engine.run_session(current_user.id, db_session.id, bg_db)
        finally:
            bg_db.close()

    sess["task"] = asyncio.create_task(run())
    return {"status": "started", "session_id": db_session.id, "config": config.dict()}


@app.post("/api/session/stop")
async def stop_session(current_user: User = Depends(get_current_user)):
    sess = engine.get_session(current_user.id)
    sess["active"] = False
    if sess.get("task"):
        sess["task"].cancel()
    return {"status": "stopped", "steps": sess["step"], "trades": sess["trade_id"]}


@app.get("/api/session/status")
async def session_status(current_user: User = Depends(get_current_user)):
    sess = engine.get_session(current_user.id)
    return {
        "active": sess["active"],
        "step":   sess["step"],
        "stats":  sess["stats"].dict() if sess.get("stats") else None,
    }


@app.get("/api/sessions/history")
async def session_history(
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    sessions = (db.query(TradingSession)
                 .filter(TradingSession.user_id == current_user.id)
                 .order_by(TradingSession.started_at.desc())
                 .limit(20).all())
    return {"sessions": [
        {"id": s.id, "symbol": s.symbol, "mode": s.mode.value,
         "started_at": s.started_at.isoformat(),
         "ended_at": s.ended_at.isoformat() if s.ended_at else None,
         "final_pnl": s.final_pnl, "sharpe": s.sharpe, "n_trades": s.n_trades}
        for s in sessions
    ]}


# ── Data routes ─────────────────────────────────────────────────────────────
@app.get("/api/trades")
async def get_trades(
    limit: int         = 50,
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    limit  = min(limit, 500)
    trades = (db.query(TradeRecord)
               .filter(TradeRecord.user_id == current_user.id)
               .order_by(TradeRecord.executed_at.desc())
               .limit(limit).all())
    total  = db.query(TradeRecord).filter(TradeRecord.user_id == current_user.id).count()
    return {"trades": [
        {"id": t.id, "timestamp": t.executed_at.isoformat(),
         "symbol": t.symbol, "side": t.side.value, "qty": t.qty,
         "fill_price": t.fill_price, "hedge_before": t.hedge_before,
         "hedge_after": t.hedge_after, "hedging_error": t.hedging_error,
         "cost": t.transaction_cost, "cvar": t.cvar_at_trade,
         "pnl": t.pnl_at_trade, "source": t.data_source, "reason": t.reason}
        for t in trades
    ], "total": total}


@app.get("/api/pnl-history")
async def get_pnl_history(
    limit: int         = 200,
    current_user: User = Depends(get_current_user),
):
    sess = engine.get_session(current_user.id)
    return {"history": sess["pnl_history"][-limit:]}


@app.get("/api/price-history")
async def get_price_history(
    limit: int         = 200,
    current_user: User = Depends(get_current_user),
):
    sess = engine.get_session(current_user.id)
    return {"history": sess["price_history"][-limit:]}


@app.get("/api/export/trades")
async def export_trades(
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    trades = (db.query(TradeRecord)
               .filter(TradeRecord.user_id == current_user.id)
               .order_by(TradeRecord.executed_at.asc()).all())
    if not trades:
        raise HTTPException(status_code=404, detail="No trades to export")
    header = "id,timestamp,symbol,side,qty,fill_price,hedge_before,hedge_after," \
             "hedging_error,cost,cvar,pnl,source,reason"
    rows   = "\n".join(
        f"{t.id},{t.executed_at.isoformat()},{t.symbol},{t.side.value},"
        f"{t.qty},{t.fill_price},{t.hedge_before},{t.hedge_after},"
        f"{t.hedging_error},{t.transaction_cost},{t.cvar_at_trade},"
        f"{t.pnl_at_trade},{t.data_source},{t.reason}"
        for t in trades
    )
    return {
        "csv":      header + "\n" + rows,
        "filename": f"quantail_{current_user.username}_{datetime.utcnow().date()}.csv",
        "total":    len(trades),
    }


# ── Baseline comparison ─────────────────────────────────────────────────────
@app.get("/api/baseline/pnl")
async def get_baseline_pnl(current_user: User = Depends(get_current_user)):
    """Delta hedge baseline P&L for comparison with Quantail."""
    sess = engine.get_session(current_user.id)
    return {
        "quantail": sess["pnl_history"][-100:],
        "delta_hedge": sess["dh_pnl_history"][-100:],
        "comparison": {
            "quantail_final":     sess["pnl_history"][-1]["pnl"] if sess["pnl_history"] else 0,
            "delta_hedge_final":  sess["dh_pnl_history"][-1]["pnl"] if sess["dh_pnl_history"] else 0,
        } if sess["pnl_history"] and sess["dh_pnl_history"] else {}
    }

# ── VQC agent info ──────────────────────────────────────────────────────────
@app.get("/api/agents/info")
async def get_agent_info(current_user: User = Depends(get_current_user)):
    """Return detailed VQC internals — what each agent is currently thinking."""
    sess = engine.get_session(current_user.id)
    agent_sess = agent_manager.get_or_create(current_user.id)

    if not sess.get("stats"):
        return {"ready": agent_manager.is_ready(), "message": "Start a session first"}

    # Build current observation
    s = sess["stats"]
    obs = np.array([
        s.spot_price / sess.get("K", s.spot_price),
        float(np.clip(sess["v"], 0, 1)),
        0.5, s.delta, float(np.clip(s.gamma, 0, 5)),
        s.hedge_ratio, 0.0, 0.0,
    ], dtype=np.float32)

    vqc_info = agent_sess.get_vqc_info(obs)
    return {
        "ready":       agent_manager.is_ready(),
        "trained":     agent_sess.trained,
        "step_count":  agent_sess.step_count,
        "agents":      vqc_info,
        "expressivity": vqc_info.get("expressivity", {}),
    }

@app.get("/api/agents/status")
async def get_agent_status():
    """Public endpoint — is the agent system ready?"""
    return {
        "ready":   agent_manager.is_ready(),
        "message": "VQC agents trained and ready" if agent_manager.is_ready() else "Pre-training in progress...",
    }


# ── AI Chat ─────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: str = ""
    csv_content: str = ""   # raw CSV text, optional

MAX_CSV_CHARS = 40_000  # ~10k rows of typical CSV

@app.post("/api/chat")
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured on server")

    system_prompt = req.system
    if req.csv_content:
        truncated = req.csv_content[:MAX_CSV_CHARS]
        was_truncated = len(req.csv_content) > MAX_CSV_CHARS
        system_prompt += f"\n\nUSER-UPLOADED CSV DATA{' (truncated to first 40k chars)' if was_truncated else ''}:\n```csv\n{truncated}\n```\nAnalyse this CSV when answering. Identify columns, row count, key statistics, and answer any specific questions about it."

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    # Groq uses OpenAI-compatible format — prepend system as first message
    groq_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": groq_messages,
                    "max_tokens": 1500,
                    "temperature": 0.4,
                },
            )
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid Groq API key")
        resp.raise_for_status()
        data = resp.json()
        return {"content": data["choices"][0]["message"]["content"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket ───────────────────────────────────────────────────────────────
@app.websocket("/ws/trading")
async def ws_trading(websocket: WebSocket, token: str = ""):
    """Real-time trading updates. Auth via ?token=JWT."""
    db = SessionLocal()
    try:
        user = ws_get_user(token, db)
    finally:
        db.close()

    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    sess = engine.get_session(user.id)
    sess["ws_clients"].append(websocket)

    # Send current state immediately on connect
    try:
        initial = json.dumps({
            "type": "connected",
            "data": sess["stats"].dict() if sess.get("stats") else None,
            "pnl_history":   sess["pnl_history"][-100:],
            "price_history": sess["price_history"][-100:],
        })
        await websocket.send_text(initial)

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "ts": datetime.utcnow().isoformat(),
                }))
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in sess["ws_clients"]:
            sess["ws_clients"].remove(websocket)


# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "environment": ENVIRONMENT}


# ── Serve React SPA (production) ────────────────────────────────────────────
_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_dist):
    _assets = os.path.join(_dist, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(os.path.join(_dist, "index.html"))


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
    )
