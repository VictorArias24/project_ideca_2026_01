"""
Repository layer for prediction CRUD operations.
"""

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.persistence.models import Prediction, Job

logger = logging.getLogger(__name__)


class PredictionRepository:
    """CRUD operations for Prediction records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, prediction: Prediction) -> Prediction:
        self.session.add(prediction)
        await self.session.commit()
        await self.session.refresh(prediction)
        return prediction

    async def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        result = await self.session.execute(
            select(Prediction).where(Prediction.id == prediction_id)
        )
        return result.scalar_one_or_none()

    async def get_by_building(self, building_id: str) -> list[Prediction]:
        result = await self.session.execute(
            select(Prediction).where(Prediction.building_id == building_id)
        )
        return list(result.scalars().all())

    async def get_review_queue(self, limit: int = 50) -> list[Prediction]:
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.review_status == "pending")
            .where(Prediction.requires_review == True)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_job(self, job_id: str) -> list[Prediction]:
        result = await self.session.execute(
            select(Prediction).where(Prediction.job_id == job_id)
        )
        return list(result.scalars().all())

    async def count_by_job(self, job_id: str) -> int:
        result = await self.session.execute(
            select(func.count()).where(Prediction.job_id == job_id)
        )
        return result.scalar() or 0


class JobRepository:
    """CRUD operations for Job records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, job: Job) -> Job:
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get_by_id(self, job_id: str) -> Optional[Job]:
        result = await self.session.execute(
            select(Job).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_jobs(self, limit: int = 50) -> list[Job]:
        result = await self.session.execute(
            select(Job).order_by(Job.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
