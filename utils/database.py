"""CONTEXT: Utility helpers for database operations and retries.

REQUIREMENTS:
- Provide convenience wrappers for executing text queries.
- Offer helper to convert query results to pandas DataFrames.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: SQLAlchemy + pandas
- Design pattern: Utility module.
- Error handling: Propagate SQLAlchemyError to callers.

INPUT/OUTPUT:
- Input: SQLAlchemy text queries + params.
- Output: Execution results or DataFrames.

EXAMPLE USAGE:
```python
from utils.database import fetch_dataframe

df = fetch_dataframe(\"SELECT * FROM stock_prices LIMIT 10\")
```

TESTING:
- Mock session + ensure commits/rollbacks as expected.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Result

from config.database import ENGINE, get_session


def execute(query: str, params: Dict[str, Any] | None = None) -> Result:
    """Execute a write query inside transactional session."""
    with get_session() as session:
        return session.execute(text(query), params or {})


def fetch_dataframe(query: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch query results as DataFrame."""
    return pd.read_sql_query(text(query), ENGINE, params=params)
