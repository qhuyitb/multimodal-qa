"""
Test production QA API endpoints
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json


BASE_URL = "http://localhost:9010"


def test_health():
    """Test health check"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/v1/qa/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_stats():
    """Test stats endpoint"""
    print("\n=== Testing Stats ===")
    response = requests.get(f"{BASE_URL}/api/v1/qa/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_vietnamese_question():
    """Test Vietnamese question"""
    print("\n=== Testing Vietnamese Question ===")
    payload = {
        "question": "XLM-RoBERTa là gì?",
        "top_k": 3
    }
    response = requests.post(f"{BASE_URL}/api/v1/qa/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Language: {data.get('query_language', 'N/A')}")
        print(f"Translated: {data.get('translated', False)}")
        print(f"Sources: {len(data.get('sources', []))}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_english_question():
    """Test English question with auto-translation"""
    print("\n=== Testing English Question ===")
    payload = {
        "question": "What is XLM-RoBERTa?",
        "top_k": 3,
        "target_language": "en"
    }
    response = requests.post(f"{BASE_URL}/api/v1/qa/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Language: {data.get('query_language', 'N/A')}")
        print(f"Translated: {data.get('translated', False)}")
        print(f"Sources: {len(data.get('sources', []))}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_cross_language():
    """Test cross-language QA: English question, Vietnamese answer"""
    print("\n=== Testing Cross-Language QA ===")
    payload = {
        "question": "How many parameters does the model have?",
        "top_k": 3,
        "target_language": "vi"
    }
    response = requests.post(f"{BASE_URL}/api/v1/qa/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Source Lang: {data.get('source_language', 'N/A')}")
        print(f"Target Lang: {data.get('target_language', 'N/A')}")
        print(f"Translated: {data.get('translated', False)}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_error_handling():
    """Test error handling with invalid request"""
    print("\n=== Testing Error Handling ===")
    
    # Empty question
    payload = {"question": ""}
    response = requests.post(f"{BASE_URL}/api/v1/qa/ask", json=payload)
    print(f"Empty question - Status: {response.status_code} (expected 422)")
    
    # Missing required field
    payload = {"top_k": 5}
    response = requests.post(f"{BASE_URL}/api/v1/qa/ask", json=payload)
    print(f"Missing question - Status: {response.status_code} (expected 422)")
    
    return True


def main():
    print("="*80)
    print("  Production QA API Test Suite")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  .venv/bin/python -m uvicorn src.api.main:app --port 9010")
    
    tests = [
        ("Health Check", test_health),
        ("Stats", test_stats),
        ("Vietnamese Question", test_vietnamese_question),
        ("English Question", test_english_question),
        ("Cross-Language QA", test_cross_language),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            results.append((name, "ERROR"))
    
    print("\n" + "="*80)
    print("  Test Results")
    print("="*80)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")


if __name__ == "__main__":
    main()
