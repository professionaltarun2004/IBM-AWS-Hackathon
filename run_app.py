"""
Simple script to run the Good Doctor Streamlit app
"""

import subprocess
import sys
import os
import time

def check_model():
    """Check if the panic model exists."""
    if not os.path.exists('panic_model.pkl'):
        print("❌ Model not found. Training model first...")
        try:
            subprocess.run([sys.executable, 'enhanced_panic_model.py'], check=True)
            print("✅ Model trained successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to train model")
            return False
    else:
        print("✅ Model found")
    return True

def run_streamlit():
    """Run the Streamlit app."""
    print("🚀 Starting Good Doctor Streamlit App...")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main function."""
    print("="*50)
    print("🏥 GOOD DOCTOR - PANIC ATTACK PREDICTION")
    print("="*50)
    
    # Check if model exists
    if not check_model():
        return
    
    print("\n📱 Starting web application...")
    print("🌐 Access at: http://localhost:8501")
    print("🌐 Alternative: http://127.0.0.1:8501")
    print("📞 Configure Twilio in the sidebar for real emergency calls")
    print("⚠️  Note: Use 'localhost' or '127.0.0.1' in your browser, NOT '0.0.0.0'")
    print("\n" + "="*50)
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main()