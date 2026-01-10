#!/usr/bin/env python3
"""
Run all tests for Multimodal QA project
"""

import sys
import pytest


def main():
    """Run all tests with proper configuration"""

    print("MULTIMODAL-QA TEST SUITE")
   
    
    # Test arguments
    args = [
        "tests/",           # Test directory
        "-v",               # Verbose
        "-s",               # Show print statements
        "--tb=short",       # Short traceback
        "-x",               # Stop on first failure (optional, comment out to continue)
    ]
    
    # Run pytest
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("[PASS] ALL TESTS PASSED")
       
    else:
        print("[FAIL] SOME TESTS FAILED")
       
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
