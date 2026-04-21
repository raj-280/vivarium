"""
pipeline/storage/postgres.py

PostgreSQL async storage implementation using SQLAlchemy async + asyncpg.
All connection parameters read from config.storage.postgres.

FIXES APPLIED:
  FIX 3: Implemented record_threshold_breach() and count_recent_threshold_breaches()
          — both were called by cooldown.py voting system but never existed here,
          causing AttributeError at runtime. Added threshold_breaches table to support them.
          Also add the CREATE TABLE for threshold_breaches in the SQL migration note below.

  FIX 7: Replaced all datetime.utcnow() with datetime.now(tz=timezone.utc).
          utcnow() is deprecated in Python 3.12+ and returns a naive datetime
          that causes subtle timezone comparison bugs in the cooldown logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from dotmap import DotMap
from loguru import logger
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.result import AlertRecord, PipelineResult
from pipeline.storage.base import BaseStorage
from pipeline.storage.models import AlertLogModel, PipelineResultModel


class PostgresStorage(BaseStorage):
    """Async PostgreSQL storage backed by asyncpg + SQLAlchemy."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._engine = None
        self._session_factory: async_sessionmaker | None = None

    def _build_dsn(self) -> str:
        pg = self.config.storage.postgres
        return (
            f"postgresql+asyncpg://{pg.user}:{pg.password}"
            f"@{pg.host}:{pg.port}/{pg.db}"
        )

    async def connect(self) -> None:
        """Create async engine and session factory."""
        pg = self.config.storage.postgres
        dsn = self._build_dsn()
        logger.info(f"Connecting to PostgreSQL at {pg.host}:{pg.port}/{pg.db}")

        self._engine = create_async_engine(
            dsn,
            pool_size=int(pg.pool_size),
            pool_timeout=float(pg.pool_timeout_seconds),
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("PostgreSQL connection pool established")

    async def disconnect(self) -> None:
        """Dispose the engine connection pool."""
        if self._engine:
            await self._engine.dispose()
            logger.info("PostgreSQL connection pool closed")

    def _session(self) -> AsyncSession:
        if self._session_factory is None:
            raise RuntimeError("PostgresStorage not connected — call .connect() first")
        return self._session_factory()

    async def save_result(self, result: PipelineResult) -> str:
        """Insert a new pipeline_results row and return its UUID."""
        new_id = uuid.uuid4()
        row = PipelineResultModel(
            id=new_id,
            image_path=result.image_path,
            water_pct=result.water_pct,
            food_pct=result.food_pct,
            mouse_present=result.mouse_present,
            water_confidence=result.water_confidence,
            food_confidence=result.food_confidence,
            mouse_confidence=result.mouse_confidence,
            uncertain_targets=result.uncertain_targets or [],
            raw_detections=result.raw_detections or {},
            # FIX 7: timezone-aware datetime
            processed_at=datetime.now(tz=timezone.utc),
        )
        async with self._session() as session:
            session.add(row)
            await session.commit()

        result_id = str(new_id)
        logger.debug(f"Saved pipeline result | id={result_id}")
        return result_id

    async def update_image_path(self, result_id: str, image_path: str) -> None:
        """Update image_path for an existing pipeline_results row."""
        async with self._session() as session:
            await session.execute(
                update(PipelineResultModel)
                .where(PipelineResultModel.id == uuid.UUID(result_id))
                .values(image_path=image_path)
            )
            await session.commit()
        logger.debug(f"Updated image_path for result_id={result_id}")

    async def get_results(
        self,
        limit: int = 20,
        offset: int = 0,
        target: Optional[str] = None,
    ) -> List[dict]:
        """Fetch pipeline_results rows as dicts, ordered by processed_at DESC."""
        async with self._session() as session:
            if target:
                col_map = {
                    "water": "water_pct",
                    "food": "food_pct",
                    "mouse": "mouse_present",
                }
                col = col_map.get(target, "water_pct")
                q = text(
                    f"SELECT * FROM pipeline_results "
                    f"WHERE {col} IS NOT NULL "
                    f"ORDER BY processed_at DESC "
                    f"LIMIT :limit OFFSET :offset"
                )
            else:
                q = text(
                    "SELECT * FROM pipeline_results "
                    "ORDER BY processed_at DESC "
                    "LIMIT :limit OFFSET :offset"
                )

            rows = await session.execute(q, {"limit": limit, "offset": offset})
            results = [dict(row._mapping) for row in rows]

        for r in results:
            r["id"] = str(r["id"])
            if r.get("processed_at"):
                r["processed_at"] = r["processed_at"].isoformat()
        return results

    async def save_alert(self, alert: AlertRecord) -> None:
        """Insert a row into alert_log."""
        row = AlertLogModel(
            id=uuid.uuid4(),
            target=alert.target,
            alert_type=alert.alert_type,
            value=alert.value,
            message=alert.message,
            notifiers_fired=alert.notifiers_fired or [],
            # FIX 7: timezone-aware datetime
            fired_at=datetime.now(tz=timezone.utc),
        )
        async with self._session() as session:
            session.add(row)
            await session.commit()
        logger.debug(f"Saved alert | target={alert.target} type={alert.alert_type}")

    async def get_last_alert_at(self, target: str) -> Optional[str]:
        """Return ISO timestamp of last alert for `target`, or None."""
        async with self._session() as session:
            q = text(
                "SELECT last_alert_at FROM cooldown_state WHERE target = :target"
            )
            row = await session.execute(q, {"target": target})
            result = row.fetchone()
            if result and result[0]:
                return result[0].isoformat()
        return None

    async def upsert_cooldown(self, target: str) -> None:
        """Insert or update cooldown_state for `target` to now()."""
        async with self._session() as session:
            await session.execute(
                text(
                    "INSERT INTO cooldown_state (target, last_alert_at) "
                    "VALUES (:target, now()) "
                    "ON CONFLICT (target) DO UPDATE SET last_alert_at = now()"
                ),
                {"target": target},
            )
            await session.commit()
        logger.debug(f"Cooldown upserted for target={target}")

    # ------------------------------------------------------------------
    # FIX 3: Implement voting system methods — missing from original code
    # These are called by CooldownManager.should_fire_with_voting()
    # and CooldownManager.record_breach() in cooldown.py.
    # Requires threshold_breaches table — see migration note below.
    #
    # Add this to your migrations/001_initial.sql (or create 002_voting.sql):
    #
    # CREATE TABLE IF NOT EXISTS threshold_breaches (
    #     id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    #     target     TEXT NOT NULL,
    #     breached_at TIMESTAMPTZ DEFAULT now()
    # );
    # CREATE INDEX IF NOT EXISTS idx_threshold_breaches_target_time
    #     ON threshold_breaches (target, breached_at DESC);
    # ------------------------------------------------------------------

    async def record_threshold_breach(self, target: str) -> None:
        """
        Insert a threshold breach event for `target`.
        Called by the voting system on every image where a target is below threshold.
        """
        async with self._session() as session:
            await session.execute(
                text(
                    "INSERT INTO threshold_breaches (id, target, breached_at) "
                    "VALUES (gen_random_uuid(), :target, now())"
                ),
                {"target": target},
            )
            await session.commit()
        logger.debug(f"Threshold breach recorded for target={target}")

    async def count_recent_threshold_breaches(
        self, target: str, minutes: int
    ) -> int:
        """
        Return count of breach events for `target` in the last `minutes` minutes.
        Used by the voting system to require N consecutive breaches before alerting.
        """
        async with self._session() as session:
            row = await session.execute(
                text(
                    "SELECT COUNT(*) FROM threshold_breaches "
                    "WHERE target = :target "
                    "AND breached_at >= now() - INTERVAL ':minutes minutes'"
                ),
                {"target": target, "minutes": minutes},
            )
            result = row.scalar()
            return int(result or 0)