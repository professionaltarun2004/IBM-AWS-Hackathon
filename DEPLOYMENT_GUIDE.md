
# Good Doctor - Enhanced Panic Attack Prediction System
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
