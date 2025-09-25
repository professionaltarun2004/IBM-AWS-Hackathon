"""
Simple script to start the API server for testing.
"""

import subprocess
import time
import sys
import os

def start_api_server():
    """Start the FastAPI server."""
    print("Starting FastAPI server...")
    
    try:
        # Change to src directory and start the API
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("API server started on http://localhost:8000")
        print("Process ID:", process.pid)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        return process
        
    except Exception as e:
        print(f"Error starting API server: {e}")
        return None

def test_api():
    """Test the API endpoints."""
    print("\nTesting API endpoints...")
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, "test_api.py"], 
                              capture_output=True, text=True)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Test errors:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("STRESS DETECTION API - STARTUP AND TEST")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists("models"):
        print("Models directory not found. Please run training first.")
        sys.exit(1)
    
    required_models = [
        "models/preprocessor.joblib",
        "models/stress_classifier.joblib", 
        "models/stress_regressor.joblib",
        "models/panic_predictor.joblib"
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    if missing_models:
        print(f"Missing required models: {missing_models}")
        print("Please run the training pipeline first.")
        sys.exit(1)
    
    print("All required models found ✓")
    
    # Start API server
    server_process = start_api_server()
    
    if server_process:
        try:
            # Test the API
            test_success = test_api()
            
            if test_success:
                print("\n🎉 API tests completed successfully!")
            else:
                print("\n⚠️  Some API tests failed.")
            
            print("\nAPI server is running. Press Ctrl+C to stop.")
            
            # Keep server running
            server_process.wait()
            
        except KeyboardInterrupt:
            print("\nShutting down API server...")
            server_process.terminate()
            server_process.wait()
            print("API server stopped.")
            
    else:
        print("Failed to start API server.")
        sys.exit(1)