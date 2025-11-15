"""CONTEXT: Centralized configuration for the market timing platform.

REQUIREMENTS:
- Load environment variables from a .env file.
- Provide Development, Production, and Testing config classes.
- Expose DB, API, Flask, logging, and training settings.
- Return the correct config via a factory based on ``FLASK_ENV``.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Flask application support
- Design pattern: Factory for configuration objects
- Error handling: Provide safe defaults when env vars are missing

INPUT/OUTPUT:
- Input: Environment variables (.env, OS env)
- Output: Config dataclasses ready for dependency injection

EXAMPLE USAGE:
```python
from config.settings import get_config

config = get_config()  # honors FLASK_ENV
engine_url = config.database.url
```

TESTING:
- Validate each environment returns expected defaults.
- Ensure misconfigured DATABASE_URL still yields structured data.

CODE STYLE:
- Use type hints for all function signatures.
- Include docstrings (Google style).
- Follow PEP 8 and 88 char max line length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class DatabaseSettings:
    """Container for parsed database settings."""

    url: str
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(frozen=True)
class APISettings:
    """API keys and integration secrets."""

    alpha_vantage_key: str


@dataclass(frozen=True)
class FlaskSettings:
    """Flask-centric settings."""

    secret_key: str
    debug: bool


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration."""

    level: str
    log_dir: Path
    app_log: Path
    pipeline_log: Path


@dataclass(frozen=True)
class TrainingSettings:
    """Model training hyper-parameters."""

    window_size: int
    prediction_horizon: int


@dataclass(frozen=True)
class BaseConfig:
    """Base configuration shared across environments."""

    database: DatabaseSettings
    apis: APISettings
    flask: FlaskSettings
    logging: LoggingSettings
    training: TrainingSettings


def _parse_database_url(url: str) -> DatabaseSettings:
    """Parse database URL into structured components."""
    parsed = urlparse(url)
    name = parsed.path.lstrip("/") or "postgres"
    port = parsed.port or 5432
    password = parsed.password or ""
    return DatabaseSettings(
        url=url,
        host=parsed.hostname or "localhost",
        port=port,
        name=name,
        user=parsed.username or "postgres",
        password=password,
    )


def _shared_logging_settings() -> LoggingSettings:
    """Build shared logging configuration."""
    log_dir = Path("logs")
    return LoggingSettings(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=log_dir,
        app_log=log_dir / "app.log",
        pipeline_log=log_dir / "pipeline.log",
    )


def _shared_training_settings() -> TrainingSettings:
    """Provide training defaults across environments."""
    window_size = int(os.getenv("TRAIN_WINDOW_SIZE", "30"))
    horizon = int(os.getenv("TRAIN_PREDICTION_HORIZON", "5"))
    return TrainingSettings(window_size=window_size, prediction_horizon=horizon)


def _base_config(debug: bool) -> BaseConfig:
    """Create the base configuration for specific environments."""
    db_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/market_timing"
    )
    database = _parse_database_url(db_url)
    apis = APISettings(alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    flask_settings = FlaskSettings(
        secret_key=os.getenv("SECRET_KEY", "changeme"),
        debug=debug,
    )
    logging_settings = _shared_logging_settings()
    training_settings = _shared_training_settings()
    return BaseConfig(
        database=database,
        apis=apis,
        flask=flask_settings,
        logging=logging_settings,
        training=training_settings,
    )


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""

    def __init__(self) -> None:
        super().__init__(**_base_config(debug=True).__dict__)


class ProductionConfig(BaseConfig):
    """Production environment configuration."""

    def __init__(self) -> None:
        super().__init__(**_base_config(debug=False).__dict__)


class TestingConfig(BaseConfig):
    """Testing-focused configuration."""

    def __init__(self) -> None:
        base = _base_config(debug=False)
        object.__setattr__(
            base,
            "database",
            _parse_database_url(
                os.getenv(
                    "TEST_DATABASE_URL",
                    "postgresql://postgres:postgres@localhost:5432/market_timing_test",
                )
            ),
        )
        super().__init__(**base.__dict__)


CONFIG_MAP: Dict[str, Type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(env: str | None = None) -> BaseConfig:
    """Return the config instance for the requested environment.

    Args:
        env: Optional explicit environment override.

    Returns:
        BaseConfig: Concrete configuration for ``env``.
    """
    env_name = (env or os.getenv("FLASK_ENV", "development")).lower()
    config_cls = CONFIG_MAP.get(env_name, DevelopmentConfig)
    return config_cls()
