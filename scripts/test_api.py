#!/usr/bin/env python3
"""
Test script for Multimodal QA API
Usage: python scripts/test_api.py
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("\nTesting Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_qa_simple():
    """Test simple QA endpoint"""
    print("\nTesting Simple QA...")
    
    payload = {
        "question": "What is the capital of France?",
        "top_k": 3,
        "translate_answer": False
    }
    
    response = requests.post(f"{BASE_URL}/qa/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_document_upload():
    """Test document upload"""
    print("\nTesting Document Upload...")
    
    test_doc_path = Path("data/input/documents/txt/test_sample.txt")
    test_doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not test_doc_path.exists():
        test_doc_path.write_text("""
This is a test document for the Multimodal QA system.
The capital of France is Paris.
Python is a popular programming language.
Machine learning is a subset of artificial intelligence.
        """.strip())
    
    with open(test_doc_path, 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = requests.post(f"{BASE_URL}/document/upload", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Task ID: {result['task_id']}")
        print(f"Filename: {result['filename']}")
        print(f"Size: {result['size_bytes']} bytes")
        return result['task_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_document_processing(task_id: str):
    """Test document processing"""
    print(f"\nTesting Document Processing (Task: {task_id})...")
    
    response = requests.post(
        f"{BASE_URL}/document/process/{task_id}",
        params={
            "index_content": True,
            "generate_translation": False
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\nPolling processing status...")
    max_attempts = 30
    for i in range(max_attempts):
        time.sleep(2)
        status_response = requests.get(f"{BASE_URL}/document/status/{task_id}")
        
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"Attempt {i+1}/{max_attempts}: {status['status']} (progress: {status.get('progress', 0)}%)")
            
            if status['status'] in ['completed', 'failed']:
                print(f"\nFinal Status: {status['status']}")
                if status['status'] == 'completed':
                    print(f"Result: {json.dumps(status.get('result', {}), indent=2)}")
                return status['status'] == 'completed'
        else:
            print(f"Error checking status: {status_response.text}")
            return False
    
    print("Processing timeout")
    return False

def test_qa_after_indexing():
    """Test QA after document indexing"""
    print("\nTesting QA after Document Indexing...")
    
    payload = {
        "question": "What is the capital of France?",
        "top_k": 3,
        "translate_answer": False
    }
    
    response = requests.post(f"{BASE_URL}/qa/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources found: {len(result['sources'])}")
        
        if result['sources']:
            print("\nTop source:")
            print(f"  Text: {result['sources'][0]['text'][:100]}...")
            print(f"  Score: {result['sources'][0]['score']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_batch_qa():
    """Test batch QA endpoint"""
    print("\nTesting Batch QA...")
    
    payload = {
        "questions": [
            "What is the capital of France?",
            "What is Python?",
            "What is machine learning?"
        ],
        "top_k": 2,
        "translate_answers": False
    }
    
    response = requests.post(f"{BASE_URL}/qa/batch", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total questions: {result['total_questions']}")
        print(f"\nResults:")
        for i, qa in enumerate(result['results'], 1):
            print(f"  {i}. {payload['questions'][i-1]}")
            print(f"     Answer: {qa['answer'][:80]}...")
            print(f"     Confidence: {qa['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("Multimodal QA API Test Suite")
    print("=" * 60)
    
    results = {}
    
    results['health'] = test_health_check()
    results['qa_simple'] = test_qa_simple()
    
    task_id = test_document_upload()
    results['document_upload'] = task_id is not None
    
    if task_id:
        results['document_processing'] = test_document_processing(task_id)
        
        if results['document_processing']:
            results['qa_after_indexing'] = test_qa_after_indexing()
            results['batch_qa'] = test_batch_qa()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:25s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API. Is the server running?")
        print("Start server with: uvicorn src.api.main:app --reload")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
