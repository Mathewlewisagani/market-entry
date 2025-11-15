"""CONTEXT: Domain-specific exception hierarchy for the platform.

REQUIREMENTS:
- Provide base exception plus specialized subclasses.
- Offer consistent error semantics for data + prediction issues.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Standard library only.
- Design pattern: Exception hierarchy.
- Error handling: Human-readable messages.

INPUT/OUTPUT:
- Input: Error conditions across modules.
- Output: Raised exceptions.

EXAMPLE USAGE:
```python
from utils.exceptions import DataValidationError

raise DataValidationError(\"missing close price\")
```

TESTING:
- Ensure inheritance works for except blocks.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""


class PlatformError(Exception):
    """Base exception for the market timing platform."""


class DataValidationError(PlatformError):
    """Raised when extracted data fails validation checks."""


class PredictionError(PlatformError):
    """Raised when the prediction workflow encounters an issue."""
