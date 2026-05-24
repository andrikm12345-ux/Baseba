from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from loguru import logger

from src.config import settings


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _get_db_url() -> str:
    if settings.database_url:
        return _normalize_url(settings.database_url)
    return "sqlite+aiosqlite:///./baseball.db"


DATABASE_URL = _get_db_url()

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition: Mapped[str] = mapped_column(String(16), index=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    utc_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Probable starting pitchers
    home_pitcher_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_pitcher_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    home_pitcher_era: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_pitcher_whip: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_pitcher_k9: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_pitcher_bb9: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_pitcher_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_pitcher_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    away_pitcher_era: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_pitcher_whip: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_pitcher_k9: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_pitcher_bb9: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    market: Mapped[str] = mapped_column(String(32))   # ML, TOTAL, RL
    pick: Mapped[str] = mapped_column(String(16))      # HOME/AWAY, OVER/UNDER, COVER/LAY
    model_prob: Mapped[float] = mapped_column(Float)
    fair_odds: Mapped[float] = mapped_column(Float)
    book_odds: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    stake_units: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    settled: Mapped[bool] = mapped_column(Boolean, default=False)
    won: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    profit_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    commentary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_ai_ensemble: Mapped[bool] = mapped_column(Boolean, default=False)
    is_value: Mapped[bool] = mapped_column(Boolean, default=False)

    match = relationship("Match")

    __table_args__ = (
        UniqueConstraint("match_id", "market", "pick", name="uq_signal_match_market_pick"),
    )


class Subscriber(Base):
    __tablename__ = "subscribers"
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    subscribed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class Setting(Base):
    __tablename__ = "settings_kv"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PendingUser(Base):
    __tablename__ = "pending_users"
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    start_count: Mapped[int] = mapped_column(Integer, default=1)


class AiPrediction(Base):
    __tablename__ = "ai_predictions"
    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    payload: Mapped[str] = mapped_column(String)


class ModelBlob(Base):
    """Stores trained ML model files as binary blobs for persistence across restarts."""
    __tablename__ = "model_blobs"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "model_ml"
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data: Mapped[bytes] = mapped_column(LargeBinary)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # model_blobs: created by metadata.create_all for new installs;
        # for existing DBs that predate this table, create_all is idempotent.
        try:
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS commentary TEXT"))
        except Exception as e:
            logger.warning(f"commentary column migration skipped: {e}")
        try:
            await conn.execute(text(
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS is_ai_ensemble BOOLEAN DEFAULT FALSE"
            ))
        except Exception as e:
            logger.warning(f"is_ai_ensemble column migration skipped: {e}")
        try:
            await conn.execute(text(
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS is_value BOOLEAN DEFAULT FALSE"
            ))
        except Exception as e:
            logger.warning(f"is_value column migration skipped: {e}")
        try:
            await conn.execute(text(
                "ALTER TABLE subscribers "
                "ADD COLUMN IF NOT EXISTS notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE"
            ))
        except Exception as e:
            logger.warning(f"notifications_enabled column migration skipped: {e}")
        try:
            result = await conn.execute(text(
                "UPDATE subscribers SET active = TRUE, notifications_enabled = FALSE "
                "WHERE active = FALSE"
            ))
            if result.rowcount:
                logger.warning(f"Amnesty: restored access for {result.rowcount} users (notifications off)")
        except Exception as e:
            logger.warning(f"amnesty UPDATE skipped: {e}")
        # Pitcher columns migration
        pitcher_cols = [
            ("home_pitcher_id", "INTEGER"),
            ("home_pitcher_name", "VARCHAR(128)"),
            ("home_pitcher_era", "FLOAT"),
            ("home_pitcher_whip", "FLOAT"),
            ("home_pitcher_k9", "FLOAT"),
            ("home_pitcher_bb9", "FLOAT"),
            ("away_pitcher_id", "INTEGER"),
            ("away_pitcher_name", "VARCHAR(128)"),
            ("away_pitcher_era", "FLOAT"),
            ("away_pitcher_whip", "FLOAT"),
            ("away_pitcher_k9", "FLOAT"),
            ("away_pitcher_bb9", "FLOAT"),
        ]
        for col, col_type in pitcher_cols:
            try:
                await conn.execute(text(
                    f"ALTER TABLE matches ADD COLUMN IF NOT EXISTS {col} {col_type}"
                ))
            except Exception as e:
                logger.warning(f"pitcher column {col} migration skipped: {e}")
