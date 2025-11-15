"""CONTEXT: Central logging configuration helper.

REQUIREMENTS:
- Configure logging formatters/handlers consistently.
- Allow optional file logging path.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: logging module.
- Design pattern: Utility function.
- Error handling: Safe to call multiple times.

INPUT/OUTPUT:
- Input: log level + optional log file path.
- Output: Configured logging module.

EXAMPLE USAGE:
```python
from utils.logging_config import configure_logging

configure_logging(\"app.log\")
```

TESTING:
- Ensure repeated calls don't duplicate handlers.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
