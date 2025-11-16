#!/usr/bin/env python3
"""Test script to demonstrate API endpoints."""

import os
import requests
import json

# Use PORT environment variable or default to 5001 (5000 is often taken by AirPlay on macOS)
PORT = os.environ.get("PORT", "5001")
BASE_URL = f"http://localhost:{PORT}"

def test_endpoint(method, endpoint, data=None):
    """Test an API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unknown method: {method}")
            return
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ Success!")
            try:
                result = response.json()
                print(f"Response:\n{json.dumps(result, indent=2)}")
            except:
                print(f"Response: {response.text}")
        else:
            print(f"❌ Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the Flask app running?")
        print("   Start it with: python3 run_app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Market Timing Platform API")
    print(f"Base URL: {BASE_URL}\n")
    
    # Test 1: Single prediction
    test_endpoint("GET", "/api/predict/AAPL")
    
    # Test 2: Batch predictions
    test_endpoint("POST", "/api/predict/batch", {"symbols": ["AAPL", "MSFT"]})
    
    # Test 3: Prediction history
    test_endpoint("GET", "/api/history/AAPL?limit=5")
    
    # Test 4: Performance (might fail if no data)
    test_endpoint("GET", "/api/performance/AAPL")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)

