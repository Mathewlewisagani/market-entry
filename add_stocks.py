#!/usr/bin/env python3
"""Add more stocks to the database."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from etl.pipeline import run_etl_pipeline

def add_stocks(symbols, days_back=730):
    """Add stock data for the given symbols."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📈 Adding {len(symbols)} stocks...")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")
    
    results = run_etl_pipeline(
        symbols=symbols,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for symbol in symbols:
        result = results.get(symbol, {})
        success = result.get("success", False)
        rows = result.get("rows_inserted", 0)
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{symbol}: {status} - {rows} rows")
        if not success and "error" in result:
            print(f"    Error: {result['error']}")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 add_stocks.py SYMBOL1 SYMBOL2 ...")
        print("\nExample:")
        print("  python3 add_stocks.py NVDA META NFLX")
        print("  python3 add_stocks.py JPM BAC WFC  # Banks")
        print("  python3 add_stocks.py SPY QQQ      # ETFs")
        sys.exit(1)
    
    symbols = [s.upper() for s in sys.argv[1:]]
    add_stocks(symbols)

