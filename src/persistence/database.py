"""
Database setup using SQLAlchemy 2.0 async with SQLite.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

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


async def get_db_session():
    """Dependency injection for FastAPI."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
