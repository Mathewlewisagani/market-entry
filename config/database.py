"""CONTEXT: SQLAlchemy session management for the analytics platform.

REQUIREMENTS:
- Create a pooled SQLAlchemy engine with safe defaults.
- Handle connection failures gracefully.
- Provide a ``get_session`` context manager for transactions.
- Centralize session factory logic for reuse.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: SQLAlchemy ORM
- Design pattern: Repository-style session gateway
- Error handling: Catch SQLAlchemyError and rollback sessions

INPUT/OUTPUT:
- Input: Database URL from configuration.
- Output: SQLAlchemy sessions yielded via context manager.

EXAMPLE USAGE:
```python
from config.database import get_session

with get_session() as session:
    session.execute(\"SELECT 1\")
```

TESTING:
- Verify pool initializes with valid credentials.
- Ensure failed queries rollback transactions.

CODE STYLE:
- Use type hints throughout.
- Include docstrings (Google style).
- Follow PEP 8 with 88 character lines.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from config.settings import BaseConfig, get_config

LOGGER = logging.getLogger(__name__)


def _create_engine(config: BaseConfig) -> Engine:
    """Create application engine with pooling."""
    pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    return create_engine(
        config.database.url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        echo=False,
        future=True,
    )


APP_CONFIG = get_config()
ENGINE = _create_engine(APP_CONFIG)
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)


def _test_connection() -> None:
    """Attempt a lightweight connection to validate credentials."""
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        LOGGER.exception("Database connection failed: %s", exc)
        raise


_test_connection()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide transactional scope around a series of operations."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        LOGGER.exception("Session rollback due to error: %s", exc)
        raise
    finally:
        session.close()
