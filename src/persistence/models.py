"""
SQLAlchemy ORM models for predictions and jobs.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, JSON, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=True, index=True)
    building_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=False)
    perfil_modelo = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False, default="p001")
    schema_version = Column(String, nullable=False, default="1.0.0")
    taxonomy_version = Column(String, nullable=False, default="1.0.0")

    image_uris = Column(JSON, nullable=False, default=list)

    classification = Column(JSON, nullable=False)

    primary_label = Column(String, nullable=True, index=True)
    max_confidence = Column(Float, nullable=True)
    requires_review = Column(Boolean, nullable=False, default=False)
    latency_ms = Column(Float, nullable=True)

    review_status = Column(
        String,
        nullable=False,
        default="pending",
    )
    reviewed_by = Column(String, nullable=True)
    review_timestamp = Column(DateTime(timezone=True), nullable=True)
    correction_reason = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "building_id": self.building_id,
            "model_id": self.model_id,
            "perfil_modelo": self.perfil_modelo,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "taxonomy_version": self.taxonomy_version,
            "image_uris": self.image_uris,
            "classification": self.classification,
            "primary_label": self.primary_label,
            "max_confidence": self.max_confidence,
            "requires_review": self.requires_review,
            "latency_ms": self.latency_ms,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "review_timestamp": self.review_timestamp.isoformat() if self.review_timestamp else None,
            "correction_reason": self.correction_reason,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(
        String,
        nullable=False,
        default="pending",
    )
    total_buildings = Column(Integer, nullable=False, default=0)
    completed_buildings = Column(Integer, nullable=False, default=0)
    failed_buildings = Column(Integer, nullable=False, default=0)
    review_count = Column(Integer, nullable=False, default=0)
    aml_job_name = Column(String, nullable=True)
    data_asset_path = Column(String, nullable=True)
    manifest_path = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
