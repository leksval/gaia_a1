#!/usr/bin/env python3
"""
Simple test script to validate GAIA API functionality
"""

import requests
import json

def test_gaia_api():
    """Test the GAIA API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print(f"Health Status: {health_response.status_code}")
    print(f"Health Response: {health_response.json()}")
    print()
    
    # Test query endpoint
    print("Testing query endpoint...")
    query_data = {
        "question": "Calculate 5+3"
    }
    
    try:
        query_response = requests.post(
            f"{base_url}/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Query Status: {query_response.status_code}")
        if query_response.status_code == 200:
            response_data = query_response.json()
            print(f"Final Answer: {response_data.get('final_answer', 'No answer')}")
            print(f"Steps Count: {len(response_data.get('intermediate_steps', []))}")
        else:
            print(f"Query Error: {query_response.text}")
    except Exception as e:
        print(f"Query Exception: {e}")
    
    print()
    
    # Test GAIA answer endpoint
    print("Testing GAIA answer endpoint...")
    try:
        gaia_response = requests.post(
            f"{base_url}/gaia-answer",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"GAIA Answer Status: {gaia_response.status_code}")
        if gaia_response.status_code == 200:
            gaia_data = gaia_response.json()
            print(f"GAIA Answer: {gaia_data.get('answer', 'No answer')}")
            print(f"GAIA Reasoning: {gaia_data.get('reasoning', 'No reasoning')[:100]}...")
        else:
            print(f"GAIA Answer Error: {gaia_response.text}")
    except Exception as e:
        print(f"GAIA Answer Exception: {e}")

if __name__ == "__main__":
    test_gaia_api()