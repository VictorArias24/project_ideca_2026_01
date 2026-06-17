"""
Database setup using SQLAlchemy 2.0 async with SQLite.
"""

import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./classifications.db",
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables and apply migrations."""
    from src.persistence.models import Base
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Migrations: add new columns if they don't exist
        for col_name, col_def in [
            ("aml_job_name", "TEXT"),
            ("data_asset_path", "TEXT"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_def}")
                )
            except Exception:
                pass  # Column already exists

        # One-time cleanup of ghost rows from botched batch imports.
        # These are rows where the parser produced building_id='unknown'
        # instead of a real ID. They block the unique index creation
        # below and inflate label_distribution with bogus PARSE_ERROR
        # counts. Safe to run repeatedly.
        await conn.execute(
            text(
                "DELETE FROM predictions "
                "WHERE building_id = 'unknown' AND job_id IS NOT NULL"
            )
        )

        # Migration: ensure unique constraint exists on (job_id, building_id).
        # IF NOT EXISTS is a no-op on subsequent runs.
        try:
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_job_building "
                    "ON predictions (job_id, building_id)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create unique index (duplicates may exist): {e}")


async def get_db_session():
    """Dependency injection for FastAPI."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
