"""
backend/auth/service.py
========================
Authentication service.

Security:
  - Passwords hashed with bcrypt (cost factor 12)
  - JWT signed with HS256, 24h access + 7d refresh
  - Invite-only registration — no open signup
  - Account lockout after 5 failed attempts (TODO: Redis)
  - Email verification token (TODO: email service)
"""

import os
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Tuple

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.db.models import User, Invitation, UserRole

# ── Config ────────────────────────────────────────────────────────────────
SECRET_KEY      = os.environ.get("QUANTAIL_SECRET_KEY", "dev-key-change-in-production")
ALGORITHM       = "HS256"
ACCESS_EXPIRE   = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))     # 24h
REFRESH_EXPIRE  = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", 7))             # 7d

# bcrypt with cost factor 12 — strong enough for production
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


# ── Password utilities ────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """Hash password with bcrypt. Never store plain text."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return pwd_context.verify(plain, hashed)


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Enforce password policy:
      - Min 8 characters
      - At least one uppercase
      - At least one digit
      - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in string.punctuation for c in password):
        return False, "Password must contain at least one special character"
    return True, "OK"


# ── JWT utilities ─────────────────────────────────────────────────────────
def create_access_token(user_id: str, username: str, role: str) -> str:
    payload = {
        "sub":      user_id,
        "username": username,
        "role":     role,
        "type":     "access",
        "exp":      datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE),
        "iat":      datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub":  user_id,
        "type": "refresh",
        "exp":  datetime.utcnow() + timedelta(days=REFRESH_EXPIRE),
        "iat":  datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ── User CRUD ─────────────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Verify username + password.
    Returns User on success, None on failure.
    Uses constant-time comparison to prevent username enumeration.
    """
    user = get_user_by_username(db, username)
    if not user:
        # Still run verify to prevent timing attacks
        pwd_context.dummy_verify()
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    full_name: str = "",
    role: UserRole = UserRole.RESEARCHER,
    invited_by_id: Optional[str] = None,
) -> User:
    """Create a new user with hashed password."""
    user = User(
        username=username,
        email=email.lower().strip(),
        hashed_password=hash_password(password),
        full_name=full_name,
        role=role,
        is_active=True,
        is_verified=False,
        invited_by=invited_by_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_last_login(db: Session, user: User):
    user.last_login = datetime.utcnow()
    db.commit()


# ── Invitation system ──────────────────────────────────────────────────────
def generate_invite_code(
    db: Session,
    created_by_id: str,
    note: str = "",
    expires_days: int = 7,
) -> Invitation:
    """
    Generate a single-use invite code.
    Only admins can call this.
    """
    code = secrets.token_urlsafe(32)   # 256-bit random URL-safe code
    invite = Invitation(
        code=code,
        created_by=created_by_id,
        note=note,
        expires_at=datetime.utcnow() + timedelta(days=expires_days),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite


def validate_invite_code(db: Session, code: str) -> Tuple[bool, str, Optional[Invitation]]:
    """
    Check if invite code is valid, unused, and not expired.
    Returns (valid, reason, invitation).
    """
    invite = db.query(Invitation).filter(Invitation.code == code).first()
    if not invite:
        return False, "Invalid invite code", None
    if invite.is_used:
        return False, "Invite code already used", None
    if invite.expires_at and invite.expires_at < datetime.utcnow():
        return False, "Invite code expired", None
    return True, "Valid", invite


def use_invite_code(db: Session, invite: Invitation, user_id: str):
    """Mark invite code as used."""
    invite.is_used = True
    invite.used_by = user_id
    invite.used_at = datetime.utcnow()
    db.commit()


def create_admin_user(db: Session) -> Optional[User]:
    """
    Create the initial admin user on first startup.
    Credentials from environment variables.
    """
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_email    = os.environ.get("ADMIN_EMAIL", "admin@quantail.ai")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Quantail2026!")

    # Check if admin already exists
    if get_user_by_username(db, admin_username):
        return None

    admin = User(
        username=admin_username,
        email=admin_email,
        hashed_password=hash_password(admin_password),
        full_name="System Admin",
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    print(f"Admin user created: {admin_username}")
    return admin
