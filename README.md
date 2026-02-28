# AuraVerse ML - Stress Detection System

A comprehensive machine learning system for stress detection and panic attack prediction in autistic children using wearable devices and behavioral data.

##Live demo streamlit
https://ibmaws.streamlit.app/

## 🎯 Project Overview

This system implements a complete ML pipeline for:
- **Stress Level Classification**: Categorize stress as Low/Moderate/High
- **Stress Intensity Prediction**: Predict continuous PSS scores (0-40)
- **Next-Day Forecasting**: Predict tomorrow's stress levels
- **Panic Attack Prediction**: Assess panic attack probability

## 📊 Dataset

The system uses a comprehensive dataset with 3,000 samples containing:
- **Physiological Data**: Skin conductance, accelerometer readings
- **Behavioral Data**: Phone usage, mobility patterns, social interactions
- **Sleep Data**: Duration, quality scores (PSQI)
- **Personality Traits**: Big Five personality dimensions
- **Target Variable**: Perceived Stress Scale (PSS) scores

## 🏗️ Architecture

### Core Components

1. **Data Preprocessing Pipeline** (`src/data_preprocessing.py`)
   - Data cleaning and validation
   - Feature engineering and scaling
   - Temporal feature creation
   - Missing value imputation

2. **ML Models** (`src/stress_models.py`)
   - Random Forest Classifier/Regressor
   - Neural Network models
   - LSTM for time-series forecasting
   - Panic prediction models

3. **Training Pipeline** (`src/train_evaluate.py`)
   - Complete model training workflow
   - Performance evaluation
   - Model persistence

4. **Real-time API** (`src/api.py`)
   - FastAPI-based REST endpoints
   - Real-time inference (<2s response time)
   - Batch prediction support

5. **Model Optimization** (`src/model_optimization.py`)
   - Hyperparameter tuning
   - SHAP-based feature importance
   - Advanced feature engineering

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Run complete training pipeline
python src/train_evaluate.py
```

### 3. Start API Server

```bash
# Start the API server
python start_api.py
```

### 4. Test the System

```bash
# Run comprehensive tests
python test_api.py
```

## 📈 Current Performance

Based on the latest evaluation:

### Stress Classification
- **Accuracy**: 61.2%
- **F1-Score**: 58.0%
- **ROC-AUC**: 76.0%
- **Status**: Below target (85% required)

### Stress Regression
- **RMSE**: 6.227
- **MAE**: 5.024
- **R²**: 0.478
- **Status**: Below target (RMSE ≤3.0, R² ≥0.75 required)

### Forecasting
- **MAE**: 9.363
- **RMSE**: 11.373
- **Status**: Below target (MAE ≤2.5 required)

### Panic Prediction
- **PR-AUC**: 0.391
- **ROC-AUC**: 0.772
- **Status**: Below target (PR-AUC ≥0.80 required)

## 🔧 API Endpoints

### Health Check
```
GET /
```

### Stress Level Prediction
```
POST /predict/stress-level
```

### Stress Intensity Prediction
```
POST /predict/stress-intensity
```

### Panic Probability Prediction
```
POST /predict/panic-probability
```

### Batch Prediction
```
POST /batch/predict
```

### Feature Importance
```
GET /models/feature-importance
```

## 📝 Example API Usage

```python
import requests

# Sample request data
data = {
    "physiological": {
        "skin_conductance": 2.5,
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
}

# Make prediction
response = requests.post("http://localhost:8000/predict/stress-level", json=data)
result = response.json()

print(f"Stress Level: {result['stress_level']}")
print(f"Confidence: {result['confidence']}")
```

## 🎯 Key Features Implemented

### ✅ Completed Features

1. **Data Preprocessing**
   - ✅ Missing value imputation
   - ✅ Feature scaling and normalization
   - ✅ Outlier detection and handling
   - ✅ Temporal feature engineering

2. **Feature Engineering**
   - ✅ Sleep efficiency calculation
   - ✅ Social load index
   - ✅ Mobility index
   - ✅ Rolling averages and lag features

3. **Model Training**
   - ✅ Multiple model comparison
   - ✅ Cross-validation
   - ✅ Performance evaluation
   - ✅ Model persistence

4. **Real-time Inference**
   - ✅ FastAPI REST endpoints
   - ✅ Sub-2-second response times
   - ✅ Batch prediction support
   - ✅ Error handling

5. **Evaluation and Monitoring**
   - ✅ Comprehensive metrics
   - ✅ Feature importance analysis
   - ✅ Performance benchmarking

### 🔄 Areas for Improvement

1. **Model Performance**
   - Current accuracy below requirements
   - Need hyperparameter optimization
   - Consider ensemble methods
   - Add more sophisticated features

2. **Data Quality**
   - Investigate feature correlations
   - Add domain-specific features
   - Consider data augmentation
   - Improve temporal modeling

3. **Advanced Features**
   - SHAP explainability
   - Personalized models
   - Alert system
   - Model monitoring

## 📊 Feature Importance

Top 5 most important features:
1. **PSS_score_3d_avg** (41.9%) - 3-day rolling average of stress
2. **PSS_score_lag1** (15.2%) - Previous day's stress level
3. **Openness** (2.1%) - Personality trait
4. **Neuroticism** (2.0%) - Personality trait
5. **call_duration** (2.0%) - Phone call duration

## 🔬 Technical Implementation

### Data Pipeline
- **Input**: CSV with 20 features, 3000 samples
- **Processing**: 7-step preprocessing pipeline
- **Output**: 32 engineered features
- **Scaling**: Z-score normalization

### Model Architecture
- **Classification**: Random Forest (best performer)
- **Regression**: Random Forest (best performer)
- **Forecasting**: LSTM neural network
- **Panic Prediction**: Random Forest binary classifier

### Performance Optimization
- **Feature Selection**: Top features identified via importance
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: 5-fold stratified CV
- **Model Selection**: Best performer selection

## 📁 Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── stress_models.py         # ML model implementations
│   ├── train_evaluate.py        # Training pipeline
│   ├── model_optimization.py    # Advanced optimization
│   └── api.py                   # FastAPI application
├── models/                      # Trained model files
├── stress_detection.csv         # Dataset
├── requirements.txt             # Dependencies
├── start_api.py                # API startup script
├── test_api.py                 # API testing script
└── README.md                   # This file
```

## 🧪 Testing

The system includes comprehensive testing:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **API Tests**: REST endpoint validation
- **Performance Tests**: Response time and load testing
- **Edge Case Tests**: Error handling validation

## 🚀 Deployment

### Local Development
```bash
python start_api.py
```

### Production Deployment
```bash
# Using Docker (create Dockerfile)
docker build -t stress-detection-api .
docker run -p 8000:8000 stress-detection-api

# Using cloud services
# Deploy to AWS Lambda, Google Cloud Run, or Azure Functions
```

## 📋 Requirements Met

### Functional Requirements
- ✅ Stress level classification
- ✅ Continuous stress prediction
- ✅ Time-series forecasting
- ✅ Panic attack prediction
- ✅ Real-time inference API
- ✅ Feature importance analysis

### Performance Requirements
- ✅ Sub-2-second response time
- ⚠️ Accuracy targets (need improvement)
- ✅ Scalable architecture
- ✅ Error handling

### Technical Requirements
- ✅ Data preprocessing pipeline
- ✅ Multiple ML algorithms
- ✅ Model evaluation framework
- ✅ REST API implementation
- ✅ Comprehensive testing

## 🔮 Future Enhancements

1. **Model Improvements**
   - Ensemble methods (voting, stacking)
   - Deep learning architectures
   - Personalized models per participant
   - Online learning capabilities

2. **Feature Engineering**
   - Advanced temporal patterns
   - Interaction features
   - Domain-specific features
   - Automated feature selection

3. **System Enhancements**
   - Real-time streaming data
   - Model monitoring and drift detection
   - A/B testing framework
   - Advanced alerting system

4. **Integration**
   - Wearable device integration
   - Unity VR system connection
   - Healthcare provider dashboard
   - Mobile application

## 📞 Support

For questions or issues:
- Check the API documentation at `http://localhost:8000/docs`
- Review test results in `api_test_report.json`
- Examine model performance in `model_evaluation_report.txt`

## 📄 License

This project is developed for healthcare applications and research purposes.

---

**Note**: This system is designed for research and development purposes. For clinical use, additional validation, regulatory approval, and safety measures would be required.
