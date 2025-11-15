"""CONTEXT: Central repository for managing external API keys.

REQUIREMENTS:
- Load secrets from environment variables with optional fallbacks.
- Provide typed accessor helpers for integrations (Alpha Vantage, etc.).

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Standard library (os).
- Design pattern: Simple accessor functions.
- Error handling: Raise ValueError when required keys missing.

INPUT/OUTPUT:
- Input: Environment variables populated via .env or hosting env.
- Output: Strings containing API credentials.

EXAMPLE USAGE:
```python
from config.api_keys import get_alpha_vantage_key

key = get_alpha_vantage_key()
```

TESTING:
- Validate raises when key absent (in strict mode).

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _fetch_key(env_var: str, required: bool = True) -> str:
    """Retrieve key from environment with optional enforcement."""
    value = os.getenv(env_var, "")
    if required and not value:
        raise ValueError(f"{env_var} is not configured.")
    return value


def get_alpha_vantage_key(required: bool = True) -> str:
    """Return Alpha Vantage API key."""
    return _fetch_key("ALPHA_VANTAGE_API_KEY", required=required)
