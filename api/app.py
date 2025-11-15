"""CONTEXT: Flask entrypoint exposing predictive analytics endpoints.

REQUIREMENTS:
- Initialize Flask with configuration + logging.
- Register prediction routes and enable CORS.
- Load ML model + DB connections at startup.
- Provide JSON error handlers for 400/404/500 statuses.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Flask
- Design pattern: Application factory.
- Error handling: Custom responses for common HTTP errors.

INPUT/OUTPUT:
- Input: HTTP requests from frontend clients.
- Output: JSON responses defined in blueprint routes.

EXAMPLE USAGE:
```bash
export FLASK_APP=api.app
flask run
```

TESTING:
- Validate error handler payloads.
- Ensure model loads on app init.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify
from flask_cors import CORS

from api.routes.predictions import predictions_bp
from api.services.prediction_service import PredictionService
from config.settings import get_config


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "api.log"),
        ],
    )


def create_app() -> Flask:
    """Application factory."""
    config = get_config()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.flask.secret_key
    app.config["ENV"] = config.flask.debug and "development" or "production"
    _configure_logging(config.logging.log_dir)
    CORS(app)
    app.register_blueprint(predictions_bp)
    app.config["prediction_service"] = PredictionService()

    @app.errorhandler(400)
    def bad_request(error: Exception):
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error: Exception):
        return jsonify({"error": "Not Found", "message": str(error)}), 404

    @app.errorhandler(500)
    def server_error(error: Exception):
        logging.exception("Unhandled error: %s", error)
        return jsonify({"error": "Internal Server Error"}), 500

    return app


app = create_app()
