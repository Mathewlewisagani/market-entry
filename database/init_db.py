"""CONTEXT: Bootstrap PostgreSQL schema for analytical workloads.

REQUIREMENTS:
- Read and execute ``database/schema.sql``.
- Create database if absent and support reset for tests.
- Log each step for observability.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: SQLAlchemy Core
- Design pattern: Scriptable initializer utility
- Error handling: Catch SQLAlchemyError and raise informative logs

INPUT/OUTPUT:
- Input: DATABASE_URL environment variable.
- Output: Initialized tables defined in schema.sql.

EXAMPLE USAGE:
```python
from database.init_db import initialize_database

initialize_database()
```

TESTING:
- Run against a disposable database to ensure idempotency.
- Exercise ``reset_database`` to confirm drop/create works.

CODE STYLE:
- Use type hints and Google-style docstrings.
- Keep lines under 88 characters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_config

LOGGER = logging.getLogger(__name__)


def _schema_sql(path: Path) -> str:
    """Read schema SQL from disk."""
    return path.read_text(encoding="utf-8")


def _create_database_if_missing(db_url: URL) -> None:
    """Ensure the target database exists."""
    admin_url = db_url.set(database="postgres")
    engine = create_engine(admin_url)
    db_name = db_url.database
    if not db_name:
        raise ValueError("Database name missing in URL.")
    query = text("SELECT 1 FROM pg_database WHERE datname=:name")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(query, {"name": db_name}).scalar()
        if not exists:
            LOGGER.info("Creating database %s", db_name)
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))


def initialize_database(schema_path: Optional[Path] = None) -> None:
    """Create database and execute schema DDL."""
    config = get_config()
    db_url = make_url(config.database.url)
    schema_file = (
        schema_path or Path(__file__).resolve().parent / "schema.sql"
    )
    _create_database_if_missing(db_url)
    engine = create_engine(db_url)
    ddl = _schema_sql(schema_file)
    try:
        with engine.begin() as conn:
            LOGGER.info("Applying schema from %s", schema_file)
            conn.exec_driver_sql(ddl)
    except SQLAlchemyError as exc:
        LOGGER.exception("Schema application failed: %s", exc)
        raise


def reset_database() -> None:
    """Drop and recreate the configured database."""
    config = get_config()
    db_url = make_url(config.database.url)
    admin_url = db_url.set(database="postgres")
    engine = create_engine(admin_url)
    drop_stmt = text(f'DROP DATABASE IF EXISTS "{db_url.database}" WITH (FORCE)')
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        LOGGER.warning("Dropping database %s", db_url.database)
        conn.execute(drop_stmt)
    initialize_database()


if __name__ == "__main__":
    initialize_database()
