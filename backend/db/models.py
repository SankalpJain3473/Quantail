"""
backend/db/models.py
====================
SQLAlchemy database models for Quantail.

Tables:
  users           - registered users with hashed passwords
  sessions        - active trading sessions per user
  trades          - full trade audit log per user
  invitations     - invite codes (so not anyone can register)
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, DateTime, Float,
    Integer, ForeignKey, Text, Enum as SAEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum
import uuid

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────
class UserRole(str, enum.Enum):
    ADMIN      = "admin"       # full access + can invite users
    RESEARCHER = "researcher"  # can run sessions, view all data
    VIEWER     = "viewer"      # read-only access to dashboard


class TradeSide(str, enum.Enum):
    BUY  = "buy"
    SELL = "sell"


class SessionMode(str, enum.Enum):
    SIMULATED = "simulated"
    PAPER     = "paper"
    LIVE      = "live"


# ── Users ─────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(String(36), primary_key=True, default=gen_uuid)
    username      = Column(String(50), unique=True, nullable=False, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name     = Column(String(100), nullable=True)
    role          = Column(SAEnum(UserRole), default=UserRole.RESEARCHER, nullable=False)
    is_active     = Column(Boolean, default=True, nullable=False)
    is_verified   = Column(Boolean, default=False, nullable=False)

    # Registration metadata
    registered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login    = Column(DateTime, nullable=True)
    invited_by    = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Relationships
    trading_sessions = relationship("TradingSession", back_populates="user",
                                    cascade="all, delete-orphan")
    trades           = relationship("TradeRecord", back_populates="user",
                                    cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ── Invitations ────────────────────────────────────────────────────────────
class Invitation(Base):
    """
    Invite-only registration system.
    Only admins can generate invite codes.
    Each code can be used exactly once.
    """
    __tablename__ = "invitations"

    id          = Column(String(36), primary_key=True, default=gen_uuid)
    code        = Column(String(64), unique=True, nullable=False, index=True)
    created_by  = Column(String(36), ForeignKey("users.id"), nullable=False)
    used_by     = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    used_at     = Column(DateTime, nullable=True)
    expires_at  = Column(DateTime, nullable=True)
    is_used     = Column(Boolean, default=False, nullable=False)
    note        = Column(String(200), nullable=True)   # e.g. "For Veronica"


# ── Trading Sessions ───────────────────────────────────────────────────────
class TradingSession(Base):
    __tablename__ = "trading_sessions"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    user_id         = Column(String(36), ForeignKey("users.id"), nullable=False)
    symbol          = Column(String(10), nullable=False)
    mode            = Column(SAEnum(SessionMode), nullable=False)
    initial_capital = Column(Float, nullable=False)
    n_steps         = Column(Integer, nullable=False)

    # Results
    final_pnl       = Column(Float, nullable=True)
    final_return    = Column(Float, nullable=True)
    cvar_95         = Column(Float, nullable=True)
    sharpe          = Column(Float, nullable=True)
    sortino         = Column(Float, nullable=True)
    hedging_rmse    = Column(Float, nullable=True)
    total_cost      = Column(Float, nullable=True)
    n_trades        = Column(Integer, default=0)
    veto_rate       = Column(Float, nullable=True)

    started_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at        = Column(DateTime, nullable=True)
    is_active       = Column(Boolean, default=True)

    # Relationships
    user   = relationship("User", back_populates="trading_sessions")
    trades = relationship("TradeRecord", back_populates="session",
                          cascade="all, delete-orphan")


# ── Trade Records ──────────────────────────────────────────────────────────
class TradeRecord(Base):
    """Full audit trail of every trade executed."""
    __tablename__ = "trades"

    id             = Column(String(36), primary_key=True, default=gen_uuid)
    session_id     = Column(String(36), ForeignKey("trading_sessions.id"), nullable=False)
    user_id        = Column(String(36), ForeignKey("users.id"), nullable=False)
    trade_seq      = Column(Integer, nullable=False)    # sequence number within session

    symbol         = Column(String(10), nullable=False)
    side           = Column(SAEnum(TradeSide), nullable=False)
    qty            = Column(Float, nullable=False)
    fill_price     = Column(Float, nullable=False)
    hedge_before   = Column(Float, nullable=False)
    hedge_after    = Column(Float, nullable=False)
    hedging_error  = Column(Float, nullable=False)
    transaction_cost = Column(Float, nullable=False)
    cvar_at_trade  = Column(Float, nullable=False)
    pnl_at_trade   = Column(Float, nullable=False)
    data_source    = Column(String(20), nullable=False)
    reason         = Column(String(50), nullable=False)
    executed_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("TradingSession", back_populates="trades")
    user    = relationship("User", back_populates="trades")
