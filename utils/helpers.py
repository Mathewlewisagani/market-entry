"""CONTEXT: Generic helper utilities shared across modules.

REQUIREMENTS:
- Provide chunking helper used for batch operations.
- Offer safe float conversion utility for external data.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Standard library only.
- Design pattern: Utility functions.
- Error handling: Gracefully handle conversion errors.

INPUT/OUTPUT:
- Input: Iterables and numeric strings.
- Output: Chunked iterables or floats.

EXAMPLE USAGE:
```python
from utils.helpers import chunked

for chunk in chunked(data, 500):
    ...
```

TESTING:
- Validate chunk boundaries and conversion error paths.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from typing import Generator, Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def chunked(iterable: Sequence[T], size: int) -> Generator[List[T], None, None]:
    """Yield successive chunks from a sequence."""
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for idx in range(0, len(iterable), size):
        yield list(iterable[idx : idx + size])


def safe_float(value: str | float | None, default: float = 0.0) -> float:
    """Convert to float, returning default on failure."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
