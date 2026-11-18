import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, create_engine, text
from sqlalchemy.exc import ArgumentError, IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.engine import URL, make_url
from sqlalchemy.dialects.postgresql import JSONB

from backend import config as _config


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

SESSION_COOKIE = "quantaai_session"
CSRF_COOKIE = "quantaai_csrf"
ACCESS_TOKEN_TTL = timedelta(minutes=15)
password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
_dummy_password_hash = password_hasher.hash(secrets.token_urlsafe(32))


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("uq_users_email_lower", text("lower(email)"), unique=True),
    )

    id: Mapped[int] = mapped_column("user_id", Integer, primary_key=True)
    name: Mapped[str] = mapped_column("full_name", String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    profile: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class RegistrationRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    profile: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        categories = (
            any(char.islower() for char in password),
            any(char.isupper() for char in password),
            any(char.isdigit() for char in password),
            any(not char.isalnum() for char in password),
        )
        if sum(categories) < 3:
            raise ValueError(
                "Password must include at least three of: lowercase, uppercase, "
                "number, and symbol."
            )
        return password


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class ProfileUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    profile: dict[str, Any] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_updates(self):
        if not self.model_fields_set:
            raise ValueError("Provide a name or profile update.")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("Name cannot be empty.")
        if "profile" in self.model_fields_set and self.profile is None:
            raise ValueError("Profile must be an object.")
        return self


DATABASE_CONNECT_TIMEOUT_SECONDS = 5
DATABASE_POOL_TIMEOUT_SECONDS = 5


def _database_url() -> URL:
    configured_url = os.getenv("DATABASE_URL", "").strip()
    if configured_url:
        if configured_url.startswith("postgres://"):
            configured_url = configured_url.replace("postgres://", "postgresql://", 1)
        try:
            url = make_url(configured_url)
        except ArgumentError:
            raise RuntimeError("DATABASE_URL is invalid.") from None

        if url.get_backend_name() == "sqlite":
            return url
        if url.get_backend_name() != "postgresql":
            raise RuntimeError("DATABASE_URL must use PostgreSQL or SQLite.")
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        elif url.drivername != "postgresql+psycopg":
            raise RuntimeError("DATABASE_URL must use the installed psycopg driver.")

        if url.host and url.host.lower().endswith(".neon.tech"):
            sslmode = url.query.get("sslmode")
            if isinstance(sslmode, tuple):
                sslmode = sslmode[-1]
            if sslmode not in {"verify-ca", "verify-full"}:
                url = url.set(query={**url.query, "sslmode": "require"})
        return url

    required = ("NEON_DB_HOST", "NEON_DB_NAME", "NEON_DB_PASSWORD")
    values = {key: os.getenv(key) for key in required}
    username = os.getenv("NEON_DB_USER") or os.getenv("NEON_DB_USERNAME")
    if any(not value for value in values.values()) or not username:
        raise RuntimeError("PostgreSQL connection settings are not configured.")

    return URL.create(
        "postgresql+psycopg",
        username=username,
        password=values["NEON_DB_PASSWORD"],
        host=values["NEON_DB_HOST"],
        database=values["NEON_DB_NAME"],
        query={"sslmode": "require"},
    )


@lru_cache(maxsize=1)
def _engine():
    url = _database_url()
    is_postgresql = url.get_backend_name() == "postgresql"
    engine_options = {
        "pool_pre_ping": True,
        "connect_args": {
            "connect_timeout" if is_postgresql else "timeout": (
                DATABASE_CONNECT_TIMEOUT_SECONDS
            )
        },
    }
    if is_postgresql:
        engine_options["pool_timeout"] = DATABASE_POOL_TIMEOUT_SECONDS
    return create_engine(url, **engine_options)


@lru_cache(maxsize=1)
def _session_factory():
    return sessionmaker(bind=_engine(), expire_on_commit=False)


def get_db():
    try:
        session = _session_factory()()
    except (RuntimeError, SQLAlchemyError) as error:
        logger.error("Authentication database configuration failed error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Authentication database is unavailable.") from None
    try:
        yield session
    finally:
        session.close()


def require_csrf(
    request: Request,
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> None:
    csrf_cookie = request.cookies.get(CSRF_COOKIE)
    if (
        not csrf_header
        or not csrf_cookie
        or not hmac.compare_digest(csrf_header, csrf_cookie)
    ):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")


def _session_cookie(request: Request) -> str:
    raw_token = request.cookies.get(SESSION_COOKIE)
    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return raw_token


def require_user(
    raw_token: str = Depends(_session_cookie),
    db: Session = Depends(get_db),
) -> User:
    try:
        auth_session = db.get(AuthSession, hashlib.sha256(raw_token.encode()).hexdigest())
        if not auth_session:
            raise HTTPException(status_code=401, detail="Invalid or expired session.")

        expires_at = auth_session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            db.delete(auth_session)
            db.commit()
            raise HTTPException(status_code=401, detail="Invalid or expired session.")

        user = db.get(User, auth_session.user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid or expired session.")
        return user
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        logger.error("Authentication session lookup failed error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Authentication is unavailable.") from None


def _set_session_cookie(response: Response, token: str) -> None:
    secure = _secure_cookie()
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(ACCESS_TOKEN_TTL.total_seconds()),
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _secure_cookie() -> bool:
    setting = os.getenv("AUTH_COOKIE_SECURE")
    if setting is not None:
        return setting.lower() in {"1", "true", "yes"}
    return os.getenv("APP_ENV", "").lower() == "production"


def _public_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "profile": user.profile or {},
        "created_at": user.created_at,
    }


def _issue_session(db: Session, user: User, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    db.add(
        AuthSession(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=datetime.now(timezone.utc) + ACCESS_TOKEN_TTL,
        )
    )
    db.commit()
    _set_session_cookie(response, token)


@router.get("/csrf")
def csrf_token(
    response: Response,
    csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE),
) -> dict[str, str]:
    token = csrf_cookie or secrets.token_urlsafe(32)
    response.set_cookie(
        CSRF_COOKIE,
        token,
        max_age=3600,
        httponly=False,
        secure=_secure_cookie(),
        samesite="lax",
        path="/",
    )
    return {"csrf_token": token}


@router.post("/register", status_code=201, dependencies=[Depends(require_csrf)])
def register(
    body: RegistrationRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    user = User(
        name=body.name,
        email=str(body.email),
        password_hash=password_hasher.hash(body.password),
        is_active=True,
        profile=body.profile,
    )
    try:
        db.add(user)
        db.flush()
        _issue_session(db, user, response)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        ) from None
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("Account registration failed error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Account registration is unavailable.") from None
    return {"user": _public_user(user)}


@router.post("/login", dependencies=[Depends(require_csrf)])
def login(
    body: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        user = db.query(User).filter_by(email=str(body.email)).first()
        password_hash = user.password_hash if user else _dummy_password_hash
        valid_password = False
        try:
            valid_password = password_hasher.verify(password_hash, body.password)
        except VerificationError:
            valid_password = False
        if not user or not user.is_active or not valid_password:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        if password_hasher.check_needs_rehash(password_hash):
            user.password_hash = password_hasher.hash(body.password)
            db.flush()

        _issue_session(db, user, response)
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        logger.error("Login failed error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Login is unavailable.") from None
    return {"user": _public_user(user)}


@router.get("/me")
def current_user(user: User = Depends(require_user)) -> dict[str, Any]:
    return _public_user(user)


@router.patch("/me", dependencies=[Depends(require_csrf)])
def update_current_user(
    body: ProfileUpdateRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if "name" in body.model_fields_set:
        if body.name is None:
            raise HTTPException(status_code=422, detail="Name cannot be empty.")
        user.name = body.name
    if "profile" in body.model_fields_set:
        if body.profile is None:
            raise HTTPException(status_code=422, detail="Profile must be an object.")
        user.profile = body.profile
    try:
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("Profile update failed error_type=%s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Profile update is unavailable.",
        ) from None
    return _public_user(user)


@router.post("/logout", status_code=204, dependencies=[Depends(require_csrf)])
def logout(
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    raw_token = request.cookies.get(SESSION_COOKIE)
    try:
        if raw_token:
            auth_session = db.get(AuthSession, hashlib.sha256(raw_token.encode()).hexdigest())
            if auth_session:
                db.delete(auth_session)
                db.commit()
    except SQLAlchemyError as error:
        logger.error("Logout failed error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Logout is unavailable.") from None
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=_secure_cookie(),
        httponly=True,
        samesite="lax",
    )
    response.status_code = 204
    return response
