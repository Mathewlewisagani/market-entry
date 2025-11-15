"""CONTEXT: Prediction-related HTTP endpoints for the Flask API.

REQUIREMENTS:
- Provide single and batch prediction endpoints.
- Return historical predictions and performance stats.
- Utilize ThreadPoolExecutor for batch requests.
- Produce JSON error responses with proper status codes.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Flask
- Design pattern: Blueprint registration.
- Error handling: Catch service exceptions and respond gracefully.

INPUT/OUTPUT:
- Input: Path params (symbol) or JSON payloads.
- Output: JSON responses matching contract.

EXAMPLE USAGE:
```bash
curl /api/predict/AAPL
```

TESTING:
- Validate bad input handling.
- Ensure batch route parallelism.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

predictions_bp = Blueprint("predictions", __name__)


def _service():
    return current_app.config["prediction_service"]


@predictions_bp.route("/api/predict/<symbol>", methods=["GET"])
def predict_symbol(symbol: str):
    """Return real-time prediction for a symbol."""
    try:
        payload = _service().predict_symbol(symbol.upper())
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@predictions_bp.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Return predictions for multiple symbols."""
    data = request.get_json(force=True) or {}
    symbols = data.get("symbols")
    if not symbols:
        return jsonify({"error": "symbols list required"}), 400
    try:
        results = _service().batch_predict([symbol.upper() for symbol in symbols])
        return jsonify({"predictions": results})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@predictions_bp.route("/api/history/<symbol>", methods=["GET"])
def prediction_history(symbol: str):
    """Fetch prediction history."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = int(request.args.get("limit", "100"))
    try:
        history = _service().prediction_history(
            symbol.upper(), start_date=start_date, end_date=end_date, limit=limit
        )
        return jsonify({"history": history})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@predictions_bp.route("/api/performance/<symbol>", methods=["GET"])
def performance(symbol: str):
    """Return backtesting results for a symbol."""
    try:
        stats = _service().performance(symbol.upper())
        return jsonify(stats)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
