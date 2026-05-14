"""
core/database.py

SQLAlchemy async database setup — SQLite by default, swappable to
PostgreSQL by changing database.url in config.yaml.

Tables are created automatically on startup via Base.metadata.create_all().

YOLOX update: Added bedding_condition, bedding_confidence, bedding_area_pct
              columns to PipelineResultRow.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from dotmap import DotMap
from loguru import logger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PipelineResultRow(Base):
    """One row per processed frame."""

    __tablename__ = "pipeline_results"

    id                  = Column(String,  primary_key=True)
    cage_id             = Column(String,  nullable=True,  index=True)
    image_path          = Column(Text,    nullable=True)
    water_pct           = Column(Float,   nullable=True)
    food_pct            = Column(Float,   nullable=True)
    mouse_present       = Column(Boolean, nullable=True)
    water_confidence    = Column(Float,   nullable=True)
    food_confidence     = Column(Float,   nullable=True)
    mouse_confidence    = Column(Float,   nullable=True)
    # Bedding fields — populated when YOLOX detector is active
    bedding_condition   = Column(String,  nullable=True)  # PERFECT | OK | BAD | WORST | NOT_DETECTED
    bedding_confidence  = Column(Float,   nullable=True)
    bedding_area_pct    = Column(Float,   nullable=True)
    uncertain_targets   = Column(JSON,    nullable=True)
    raw_detections      = Column(JSON,    nullable=True)
    processed_at        = Column(DateTime(timezone=True), nullable=False)


class AlertLogRow(Base):
    """One row per LOW / EMPTY alert fired."""

    __tablename__ = "alert_log"

    id         = Column(String,  primary_key=True)
    cage_id    = Column(String,  nullable=True, index=True)
    target     = Column(String,  nullable=False)
    alert_type = Column(String,  nullable=False)
    value      = Column(Float,   nullable=True)
    message    = Column(Text,    nullable=True)
    fired_at   = Column(DateTime(timezone=True), nullable=False)


_engine = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_DEFAULT_URL = "sqlite+aiosqlite:///./vivarium.db"


async def init_db(config: DotMap) -> None:
    global _engine, _session_factory

    db_cfg = getattr(config, "database", DotMap())
    url: str = str(getattr(db_cfg, "url", _DEFAULT_URL))

    logger.info(f"Initialising database | url={url}")

    _engine = create_async_engine(url, echo=False, future=True)
    _session_factory = async_sessionmaker(
        bind=_engine, class_=AsyncSession, expire_on_commit=False,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database ready — tables verified")


async def close_db() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialised — was init_db() called?")
    async with _session_factory() as session:
        yield session


def is_db_ready() -> bool:
    return _engine is not None and _session_factory is not None
