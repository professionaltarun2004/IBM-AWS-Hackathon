"""
Test script to verify Streamlit app functionality
"""

import sys
import os
import subprocess
import time
import requests

def test_streamlit_import():
    """Test if all imports work correctly."""
    print("Testing Streamlit app imports...")
    
    try:
        # Test individual imports
        import streamlit as st
        print("✅ Streamlit imported")
        
        import pandas as pd
        print("✅ Pandas imported")
        
        import numpy as np
        print("✅ NumPy imported")
        
        import plotly.graph_objects as go
        print("✅ Plotly imported")
        
        from enhanced_panic_model import EnhancedPanicPredictor
        print("✅ Enhanced panic model imported")
        
        # Test model loading
        predictor = EnhancedPanicPredictor()
        predictor.load_model('panic_model.pkl')
        print("✅ Model loaded successfully")
        
        # Test prediction
        test_prob = predictor.predict_panic_probability(85, 2.0, 3.0, 6.0)
        print(f"✅ Test prediction: {test_prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def start_streamlit_test():
    """Start Streamlit app for testing."""
    print("\nStarting Streamlit app...")
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for app to start
        print("Waiting for app to start...")
        time.sleep(10)
        
        # Test if app is responding
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit app is running successfully!")
                print("🌐 Access at: http://localhost:8501")
                return process
            else:
                print(f"❌ App returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to app: {e}")
        
        # If we get here, something went wrong
        stdout, stderr = process.communicate(timeout=5)
        print(f"Stdout: {stdout.decode()}")
        print(f"Stderr: {stderr.decode()}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None

def main():
    """Main test function."""
    print("="*50)
    print("STREAMLIT APP TESTING")
    print("="*50)
    
    # Test imports
    if not test_streamlit_import():
        print("❌ Import tests failed. Cannot proceed.")
        return False
    
    print("\n" + "="*50)
    print("All imports successful! Starting Streamlit app...")
    print("="*50)
    
    # Start app
    process = start_streamlit_test()
    
    if process:
        try:
            print("\n🎉 App is running! Press Ctrl+C to stop.")
            print("📱 Open http://localhost:8501 in your browser")
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping app...")
            process.terminate()
            process.wait()
            print("✅ App stopped")
    else:
        print("❌ Failed to start Streamlit app")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)