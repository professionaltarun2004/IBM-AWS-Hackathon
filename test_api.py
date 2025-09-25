"""
Comprehensive testing script for the stress detection API.
Tests all endpoints and validates model performance.
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List
import asyncio
import aiohttp

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def create_sample_request():
    """Create a sample stress assessment request."""
    return {
        "physiological": {
            "skin_conductance": 2.5,
            "accelerometer": 1.2,
            "heart_rate": 75
        },
        "behavioral": {
            "call_duration": 15.5,
            "num_calls": 5,
            "num_sms": 12,
            "screen_on_time": 6.5,
            "mobility_radius": 2.1,
            "mobility_distance": 5.3
        },
        "sleep": {
            "sleep_duration": 7.5,
            "PSQI_score": 3
        },
        "personality": {
            "openness": 3.2,
            "conscientiousness": 4.1,
            "extraversion": 2.8,
            "agreeableness": 3.9,
            "neuroticism": 2.3
        },
        "participant_id": "test_participant",
        "timestamp": "2024-01-15T10:30:00"
    }

def test_stress_level_prediction():
    """Test stress level classification endpoint."""
    print("\nTesting stress level prediction...")
    
    try:
        sample_request = create_sample_request()
        response = requests.post(
            f"{BASE_URL}/predict/stress-level",
            json=sample_request
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Stress Level: {result['stress_level']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"Probabilities: {result['probabilities']}")
            
            # Validate response time (should be < 2000ms)
            if result['processing_time_ms'] < 2000:
                print("✓ Response time requirement met (<2s)")
            else:
                print("✗ Response time requirement not met")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Stress level prediction test failed: {e}")
        return False

def test_stress_intensity_prediction():
    """Test stress intensity regression endpoint."""
    print("\nTesting stress intensity prediction...")
    
    try:
        sample_request = create_sample_request()
        response = requests.post(
            f"{BASE_URL}/predict/stress-intensity",
            json=sample_request
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"PSS Score: {result['pss_score']:.2f}")
            print(f"Confidence Interval: {result['confidence_interval']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            
            # Validate PSS score range (0-40)
            if 0 <= result['pss_score'] <= 40:
                print("✓ PSS score in valid range (0-40)")
            else:
                print("✗ PSS score out of valid range")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Stress intensity prediction test failed: {e}")
        return False

def test_panic_probability_prediction():
    """Test panic attack probability endpoint."""
    print("\nTesting panic probability prediction...")
    
    try:
        sample_request = create_sample_request()
        response = requests.post(
            f"{BASE_URL}/predict/panic-probability",
            json=sample_request
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Panic Probability: {result['panic_probability']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Alert Triggered: {result['alert_triggered']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            
            # Validate probability range (0-1)
            if 0 <= result['panic_probability'] <= 1:
                print("✓ Panic probability in valid range (0-1)")
            else:
                print("✗ Panic probability out of valid range")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Panic probability prediction test failed: {e}")
        return False

def test_feature_importance():
    """Test feature importance endpoint."""
    print("\nTesting feature importance endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/models/feature-importance")
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if 'feature_importance' in result:
                print("Top 5 Important Features:")
                for i, feature in enumerate(result['feature_importance'][:5]):
                    print(f"  {i+1}. {feature['feature']}: {feature['importance']:.3f}")
                print(f"Total Features: {result['total_features']}")
            else:
                print(f"Feature importance not available: {result}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Feature importance test failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction...")
    
    try:
        # Create multiple sample requests
        batch_requests = []
        for i in range(3):
            request = create_sample_request()
            request['participant_id'] = f"batch_participant_{i}"
            # Vary some parameters
            request['physiological']['skin_conductance'] = 2.0 + i * 0.5
            request['sleep']['sleep_duration'] = 7.0 + i * 0.5
            batch_requests.append(request)
        
        response = requests.post(
            f"{BASE_URL}/batch/predict",
            json=batch_requests
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Batch processed: {result['total_processed']} samples")
            
            # Show results for first participant
            if result['batch_results']:
                first_result = result['batch_results'][0]
                print(f"Sample result for {first_result['participant_id']}:")
                print(f"  Stress Level: {first_result['stress_classification']['stress_level']}")
                print(f"  PSS Score: {first_result['stress_intensity']['pss_score']:.2f}")
                print(f"  Panic Probability: {first_result['panic_probability']['panic_probability']:.3f}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Batch prediction test failed: {e}")
        return False

def test_performance_load():
    """Test API performance under load."""
    print("\nTesting API performance under load...")
    
    try:
        sample_request = create_sample_request()
        num_requests = 10
        response_times = []
        
        print(f"Sending {num_requests} concurrent requests...")
        
        start_time = time.time()
        for i in range(num_requests):
            response = requests.post(
                f"{BASE_URL}/predict/stress-level",
                json=sample_request
            )
            if response.status_code == 200:
                result = response.json()
                response_times.append(result['processing_time_ms'])
        
        total_time = time.time() - start_time
        
        if response_times:
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            min_response_time = np.min(response_times)
            
            print(f"Total time: {total_time:.2f}s")
            print(f"Average response time: {avg_response_time:.2f}ms")
            print(f"Min response time: {min_response_time:.2f}ms")
            print(f"Max response time: {max_response_time:.2f}ms")
            print(f"Requests per second: {num_requests/total_time:.2f}")
            
            # Check if average response time meets requirement (<2s)
            if avg_response_time < 2000:
                print("✓ Average response time requirement met")
                return True
            else:
                print("✗ Average response time requirement not met")
                return False
        else:
            print("No successful responses received")
            return False
            
    except Exception as e:
        print(f"Performance load test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    test_cases = [
        {
            "name": "Missing required field",
            "request": {
                "physiological": {
                    "skin_conductance": 2.5,
                    "accelerometer": 1.2
                },
                # Missing other required fields
            },
            "expected_status": 422
        },
        {
            "name": "Invalid data types",
            "request": {
                "physiological": {
                    "skin_conductance": "invalid",
                    "accelerometer": 1.2
                },
                "behavioral": {
                    "call_duration": 15.5,
                    "num_calls": 5,
                    "num_sms": 12,
                    "screen_on_time": 6.5,
                    "mobility_radius": 2.1,
                    "mobility_distance": 5.3
                },
                "sleep": {
                    "sleep_duration": 7.5,
                    "PSQI_score": 3
                },
                "personality": {
                    "openness": 3.2,
                    "conscientiousness": 4.1,
                    "extraversion": 2.8,
                    "agreeableness": 3.9,
                    "neuroticism": 2.3
                }
            },
            "expected_status": 422
        }
    ]
    
    passed_tests = 0
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/predict/stress-level",
                json=test_case["request"]
            )
            
            if response.status_code == test_case["expected_status"]:
                print(f"✓ {test_case['name']}: Expected status {test_case['expected_status']}")
                passed_tests += 1
            else:
                print(f"✗ {test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"✗ {test_case['name']}: Exception occurred - {e}")
    
    return passed_tests == len(test_cases)

def run_comprehensive_tests():
    """Run all API tests."""
    print("="*60)
    print("COMPREHENSIVE API TESTING")
    print("="*60)
    
    test_results = []
    
    # Individual tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Stress Level Prediction", test_stress_level_prediction),
        ("Stress Intensity Prediction", test_stress_intensity_prediction),
        ("Panic Probability Prediction", test_panic_probability_prediction),
        ("Feature Importance", test_feature_importance),
        ("Batch Prediction", test_batch_prediction),
        ("Performance Load Test", test_performance_load),
        ("Edge Cases", test_edge_cases)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            print(f"\n{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"\n{test_name}: FAILED - {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the API and models.")
    
    return test_results

if __name__ == "__main__":
    # Wait a moment for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    results = run_comprehensive_tests()
    
    # Save test results
    test_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [{"test": name, "passed": result} for name, result in results],
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for _, result in results if result),
            "failed": sum(1 for _, result in results if not result)
        }
    }
    
    with open("api_test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\nTest report saved to api_test_report.json")