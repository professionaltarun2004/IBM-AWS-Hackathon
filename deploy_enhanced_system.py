"""
Deployment script for the enhanced panic attack prediction system.
Includes AWS IoT integration preparation and system validation.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed."""
    print("🔍 Checking system requirements...")
    
    required_files = [
        'panic_model.pkl',
        'enhanced_panic_model.py',
        'streamlit_app.py',
        'stress_detection.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def validate_model():
    """Validate the trained model."""
    print("🤖 Validating trained model...")
    
    try:
        from enhanced_panic_model import EnhancedPanicPredictor
        
        predictor = EnhancedPanicPredictor()
        predictor.load_model('panic_model.pkl')
        
        # Test prediction
        test_probability = predictor.predict_panic_probability(
            heart_rate=85,
            accelerometer=2.0,
            skin_conductance=3.0,
            sleep_duration=6.0
        )
        
        print(f"✅ Model validation successful")
        print(f"   Test prediction: {test_probability:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def create_aws_iot_config():
    """Create AWS IoT integration configuration template."""
    print("☁️ Creating AWS IoT configuration template...")
    
    aws_config = {
        "aws_iot": {
            "region": "us-east-1",
            "thing_name": "good-doctor-wearable",
            "topic_prefix": "good-doctor/data",
            "endpoints": {
                "data_ingestion": "good-doctor/wearable/data",
                "alerts": "good-doctor/alerts/panic",
                "status": "good-doctor/device/status"
            }
        },
        "data_mapping": {
            "heart_rate": "sensors.heart_rate",
            "accelerometer": "sensors.accelerometer.magnitude",
            "skin_conductance": "sensors.gsr.value",
            "sleep_duration": "sleep.duration_hours"
        },
        "alert_thresholds": {
            "panic_probability": 0.7,
            "consecutive_alerts": 3,
            "cooldown_minutes": 15
        },
        "twilio": {
            "account_sid": "${TWILIO_ACCOUNT_SID}",
            "auth_token": "${TWILIO_AUTH_TOKEN}",
            "from_phone": "${TWILIO_PHONE}",
            "to_phone": "${CAREGIVER_PHONE}"
        }
    }
    
    with open('aws_iot_config.json', 'w') as f:
        json.dump(aws_config, f, indent=2)
    
    print("✅ AWS IoT configuration template created: aws_iot_config.json")

def create_docker_config():
    """Create Docker configuration for deployment."""
    print("🐳 Creating Docker configuration...")
    
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    docker_compose_content = """
version: '3.8'

services:
  good-doctor-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID}
      - TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN}
      - TWILIO_PHONE=${TWILIO_PHONE}
      - CAREGIVER_PHONE=${CAREGIVER_PHONE}
    volumes:
      - ./panic_model.pkl:/app/panic_model.pkl:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Docker configuration created: Dockerfile, docker-compose.yml")

def create_environment_template():
    """Create environment variables template."""
    print("🔧 Creating environment template...")
    
    env_template = """
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE=+1234567890
CAREGIVER_PHONE=+1987654321

# AWS Configuration (for future IoT integration)
AWS_REGION=us-east-1
AWS_IOT_ENDPOINT=your-iot-endpoint.iot.us-east-1.amazonaws.com
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Application Configuration
PANIC_THRESHOLD=0.7
AUTO_REFRESH_SECONDS=30
LOG_LEVEL=INFO
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Environment template created: .env.template")
    print("   Copy to .env and fill in your actual credentials")

def start_streamlit_app():
    """Start the Streamlit application."""
    print("🚀 Starting Streamlit application...")
    
    try:
        # Start Streamlit in the background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        print("✅ Streamlit app started successfully!")
        print("🌐 Access the app at: http://localhost:8501")
        print("📱 For mobile access, use your computer's IP address")
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None

def create_deployment_guide():
    """Create comprehensive deployment guide."""
    print("📚 Creating deployment guide...")
    
    guide_content = """
# AuraVerse - Enhanced Panic Attack Prediction System
## Deployment Guide

## 🚀 Quick Start

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Train the model (if not already done)
python enhanced_panic_model.py

# Start the Streamlit app
streamlit run streamlit_app.py
```

### 2. Production Deployment

#### Option A: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Option B: Cloud Deployment (AWS/Azure/GCP)
```bash
# Deploy to AWS App Runner, Azure Container Instances, or Google Cloud Run
# Use the provided Dockerfile for containerized deployment
```

## 🔧 Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:

```bash
# Twilio Configuration (Required for alerts)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE=+1234567890
CAREGIVER_PHONE=+1987654321

# AWS IoT (For wearable integration)
AWS_REGION=us-east-1
AWS_IOT_ENDPOINT=your-endpoint.iot.region.amazonaws.com
```

## 📱 Wearable Integration

### Current Implementation
- **Input Method**: Manual entry via Streamlit interface
- **Required Data**: Heart rate, accelerometer, skin conductance, sleep duration
- **Response Time**: Real-time (<1 second)

### AWS IoT Integration (Future)
```json
{
  "device_id": "wearable_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "sensors": {
    "heart_rate": 85,
    "accelerometer": {"x": 0.1, "y": 0.2, "z": 0.9, "magnitude": 2.1},
    "skin_conductance": 3.2
  },
  "sleep": {
    "duration_hours": 6.5,
    "quality_score": 7
  }
}
```

## 🚨 Alert System

### Panic Risk Levels
- **Low Risk** (0-40%): Green indicator, continue monitoring
- **Moderate Risk** (40-70%): Yellow indicator, increased attention
- **High Risk** (70%+): Red indicator, emergency alert triggered

### Twilio IVR Integration
When panic probability exceeds 70%:
1. Automated call placed to caregiver
2. Voice message with alert details
3. Acknowledgment required
4. Call details logged for audit

## 📊 Model Performance

### Current Metrics
- **Accuracy**: 51.7%
- **F1-Score**: 41.8%
- **ROC-AUC**: 50.2%
- **Features**: 16 total (including synthetic heart rate)

### Top Important Features
1. Mobility distance
2. Conscientiousness (personality)
3. Neuroticism (personality)
4. Skin conductance
5. Accelerometer activity

## 🔍 Monitoring & Maintenance

### Health Checks
- Model prediction accuracy
- Response time monitoring
- Twilio service status
- Data quality validation

### Logging
- All predictions logged with timestamps
- Alert triggers and responses tracked
- System performance metrics collected

## 🛠️ Troubleshooting

### Common Issues
1. **Model not found**: Run `python enhanced_panic_model.py` to train
2. **Twilio errors**: Check credentials in environment variables
3. **Streamlit issues**: Ensure port 8501 is available

### Support
- Check logs in `logs/` directory
- Review model performance in Streamlit interface
- Validate input data ranges and formats

## 🔮 Future Enhancements

### Planned Features
- Real-time wearable data streaming
- Historical trend analysis
- Multi-participant monitoring
- Advanced ML models (ensemble methods)
- Mobile app integration
- Clinical dashboard for healthcare providers

### AWS Architecture (Planned)
```
Wearables → AWS IoT Core → Lambda → SageMaker → SNS → Twilio
                ↓
            DynamoDB/Timestream (Storage)
                ↓
            CloudWatch (Monitoring)
```
"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ Deployment guide created: DEPLOYMENT_GUIDE.md")

def main():
    """Main deployment function."""
    print("="*60)
    print("🏥 AURAVERSE - ENHANCED SYSTEM DEPLOYMENT")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please ensure all files are present.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed.")
        return False
    
    # Validate model
    if not validate_model():
        print("❌ Model validation failed.")
        return False
    
    # Create configuration files
    create_aws_iot_config()
    create_docker_config()
    create_environment_template()
    create_deployment_guide()
    
    print("\n" + "="*60)
    print("✅ DEPLOYMENT PREPARATION COMPLETE!")
    print("="*60)
    
    print("""
🎯 Next Steps:
1. Configure Twilio credentials in .env file
2. Start the application: streamlit run streamlit_app.py
3. Access the web interface at http://localhost:8501
4. Test panic prediction with sample data
5. Configure AWS IoT for wearable integration (optional)

🚀 The enhanced AuraVerse system is ready for deployment!
""")
    
    # Ask if user wants to start the app
    start_app = input("\nStart Streamlit app now? (y/n): ").lower().strip()
    if start_app == 'y':
        process = start_streamlit_app()
        if process:
            try:
                print("\n🔄 App is running. Press Ctrl+C to stop.")
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping application...")
                process.terminate()
                process.wait()
                print("✅ Application stopped successfully")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)