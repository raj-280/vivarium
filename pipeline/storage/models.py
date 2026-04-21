"""
pipeline/storage/models.py

SQLAlchemy ORM models for PostgreSQL.
All table/column names follow the schema in migrations/001_initial.sql.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PipelineResultModel(Base):
    __tablename__ = "pipeline_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_path = Column(Text, nullable=True)
    water_pct = Column(Float, nullable=True)
    food_pct = Column(Float, nullable=True)
    mouse_present = Column(Boolean, nullable=True)
    water_confidence = Column(Float, nullable=True)
    food_confidence = Column(Float, nullable=True)
    mouse_confidence = Column(Float, nullable=True)
    uncertain_targets = Column(ARRAY(Text), nullable=True, default=list)
    raw_detections = Column(JSONB, nullable=True)
    processed_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class AlertLogModel(Base):
    __tablename__ = "alert_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=True)
    message = Column(Text, nullable=False)
    notifiers_fired = Column(ARRAY(Text), nullable=True, default=list)
    fired_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class CooldownStateModel(Base):
    __tablename__ = "cooldown_state"

    target = Column(String(50), primary_key=True)
    last_alert_at = Column(DateTime(timezone=True), nullable=False)
