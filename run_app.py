#!/usr/bin/env python3
"""Simple script to run the Flask app with helpful output."""

import os
import sys
from pathlib import Path

# Set Flask app
os.environ["FLASK_APP"] = "api.app"
os.environ["FLASK_ENV"] = "development"

# Default database URL if not set
# Note: Update this to match your PostgreSQL credentials
if "DATABASE_URL" not in os.environ:
    # Try common configurations
    # Docker container password: postgres123, database: market_timing_db
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres123@localhost:5432/market_timing_db"

print("=" * 60)
print("🚀 Starting Market Timing Platform API")
print("=" * 60)
print(f"📊 Database: {os.environ.get('DATABASE_URL', 'Not set')}")
print(f"🔑 Flask App: {os.environ.get('FLASK_APP')}")
print(f"🌍 Environment: {os.environ.get('FLASK_ENV')}")
print("\n📡 Available Endpoints:")
print("  • GET  /api/predict/<symbol>        - Get prediction for a stock")
print("  • POST /api/predict/batch           - Batch predictions")
print("  • GET  /api/history/<symbol>        - Prediction history")
print("  • GET  /api/performance/<symbol>     - Backtesting results")
# Get port from environment or use 5001 (5000 is often taken by AirPlay on macOS)
PORT = int(os.environ.get("PORT", "5001"))

print("\n💡 Example usage:")
print(f"  curl http://localhost:{PORT}/api/predict/AAPL")
print("\n" + "=" * 60)
print(f"Starting Flask server on port {PORT}...\n")

# Import and run Flask
from flask import Flask
from api.app import app

if __name__ == "__main__":
    import socket
    
    # Try to bind to the port, if it fails try the next one
    def find_free_port(start_port):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback
    
    free_port = find_free_port(PORT)
    if free_port != PORT:
        print(f"⚠️  Port {PORT} is in use, using port {free_port} instead")
    
    app.run(host="0.0.0.0", port=free_port, debug=True)

