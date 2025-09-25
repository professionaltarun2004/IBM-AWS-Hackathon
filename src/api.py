"""
FastAPI application for real-time stress detection and prediction.
Provides REST endpoints for all ML models and real-time inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict, List, Optional, Any
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Good Doctor ML - Stress Detection API",
    description="Real-time stress detection and panic attack prediction system",
    version="1.0.0"
)

# Global model storage
models = {}
preprocessor = None

# Pydantic models for API requests
class PhysiologicalData(BaseModel):
    skin_conductance: float
    accelerometer: float
    heart_rate: Optional[float] = None

class BehavioralData(BaseModel):
    call_duration: float
    num_calls: int
    num_sms: int
    screen_on_time: float
    mobility_radius: float
    mobility_distance: float

class SleepData(BaseModel):
    sleep_duration: float
    PSQI_score: int

class PersonalityData(BaseModel):
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

class StressAssessmentRequest(BaseModel):
    physiological: PhysiologicalData
    behavioral: BehavioralData
    sleep: SleepData
    personality: PersonalityData
    participant_id: Optional[str] = "unknown"
    timestamp: Optional[str] = None

class StressClassificationResponse(BaseModel):
    stress_level: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    processing_time_ms: float

class StressIntensityResponse(BaseModel):
    pss_score: float
    confidence_interval: List[float]
    timestamp: str
    processing_time_ms: float

class PanicProbabilityResponse(BaseModel):
    panic_probability: float
    risk_level: str
    alert_triggered: bool
    timestamp: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime: str

@app.on_event("startup")
async def load_models():
    """Load all trained models on startup."""
    global models, preprocessor
    
    try:
        logger.info("Loading trained models...")
        
        # Load preprocessor
        preprocessor = joblib.load("models/preprocessor.joblib")
        logger.info("Preprocessor loaded successfully")
        
        # Load classification model
        models['classifier'] = joblib.load("models/stress_classifier.joblib")
        logger.info("Stress classifier loaded successfully")
        
        # Load regression model
        models['regressor'] = joblib.load("models/stress_regressor.joblib")
        logger.info("Stress regressor loaded successfully")
        
        # Load panic predictor
        models['panic_predictor'] = joblib.load("models/panic_predictor.joblib")
        logger.info("Panic predictor loaded successfully")
        
        # Try to load forecasting model
        try:
            models['forecaster'] = tf.keras.models.load_model("models/forecasting_model.h5")
            logger.info("Forecasting model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load forecasting model: {e}")
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def prepare_input_data(request: StressAssessmentRequest) -> pd.DataFrame:
    """Convert API request to DataFrame for model input."""
    data = {
        'Openness': request.personality.openness,
        'Conscientiousness': request.personality.conscientiousness,
        'Extraversion': request.personality.extraversion,
        'Agreeableness': request.personality.agreeableness,
        'Neuroticism': request.personality.neuroticism,
        'sleep_duration': request.sleep.sleep_duration,
        'PSQI_score': request.sleep.PSQI_score,
        'call_duration': request.behavioral.call_duration,
        'num_calls': request.behavioral.num_calls,
        'num_sms': request.behavioral.num_sms,
        'screen_on_time': request.behavioral.screen_on_time,
        'skin_conductance': request.physiological.skin_conductance,
        'accelerometer': request.physiological.accelerometer,
        'mobility_radius': request.behavioral.mobility_radius,
        'mobility_distance': request.behavioral.mobility_distance,
        'PSS_score': 25  # Placeholder for preprocessing
    }
    
    df = pd.DataFrame([data])
    return df

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        uptime=str(datetime.now())
    )

@app.post("/predict/stress-level", response_model=StressClassificationResponse)
async def predict_stress_level(request: StressAssessmentRequest):
    """Predict stress level classification (Low/Moderate/High)."""
    start_time = datetime.now()
    
    try:
        # Prepare input data
        df = prepare_input_data(request)
        
        # Preprocess data
        df_processed = preprocessor.transform(df)
        X = df_processed.drop('PSS_score', axis=1).values
        
        # Make prediction
        classifier = models['classifier']
        stress_levels, probabilities = classifier.predict(X)
        
        # Format response
        prob_dict = {}
        if hasattr(classifier, 'label_encoder'):
            classes = classifier.label_encoder.classes_
            for i, class_name in enumerate(classes):
                prob_dict[class_name] = float(probabilities[0][i])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return StressClassificationResponse(
            stress_level=stress_levels[0],
            confidence=float(max(probabilities[0])),
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in stress level prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/stress-intensity", response_model=StressIntensityResponse)
async def predict_stress_intensity(request: StressAssessmentRequest):
    """Predict continuous stress intensity (PSS score)."""
    start_time = datetime.now()
    
    try:
        # Prepare input data
        df = prepare_input_data(request)
        
        # Preprocess data
        df_processed = preprocessor.transform(df)
        X = df_processed.drop('PSS_score', axis=1).values
        
        # Make prediction
        regressor = models['regressor']
        pss_score, uncertainty = regressor.predict_with_confidence(X)
        
        # Calculate confidence interval
        confidence_interval = [
            float(pss_score[0] - uncertainty[0]),
            float(pss_score[0] + uncertainty[0])
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return StressIntensityResponse(
            pss_score=float(pss_score[0]),
            confidence_interval=confidence_interval,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in stress intensity prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/panic-probability", response_model=PanicProbabilityResponse)
async def predict_panic_probability(request: StressAssessmentRequest):
    """Predict panic attack probability."""
    start_time = datetime.now()
    
    try:
        # Prepare input data
        df = prepare_input_data(request)
        
        # Preprocess data
        df_processed = preprocessor.transform(df)
        X = df_processed.drop('PSS_score', axis=1).values
        
        # Make prediction
        panic_predictor = models['panic_predictor']
        panic_prob = panic_predictor.predict_panic_probability(X)
        
        # Determine risk level and alert
        prob_value = float(panic_prob[0])
        if prob_value > 0.7:
            risk_level = "HIGH"
            alert_triggered = True
        elif prob_value > 0.4:
            risk_level = "MODERATE"
            alert_triggered = False
        else:
            risk_level = "LOW"
            alert_triggered = False
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PanicProbabilityResponse(
            panic_probability=prob_value,
            risk_level=risk_level,
            alert_triggered=alert_triggered,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in panic probability prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/feature-importance")
async def get_feature_importance():
    """Get feature importance from the regression model."""
    try:
        regressor = models['regressor']
        if hasattr(regressor, 'best_model') and hasattr(regressor.best_model, 'feature_importances_'):
            importance = regressor.best_model.feature_importances_
            
            # Get feature names (this would need to be stored during training)
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            importance_data = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importance)
            ]
            
            # Sort by importance
            importance_data.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                "feature_importance": importance_data[:10],  # Top 10
                "total_features": len(importance_data)
            }
        else:
            return {"error": "Feature importance not available"}
            
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/predict")
async def batch_predict(requests: List[StressAssessmentRequest]):
    """Batch prediction endpoint for multiple samples."""
    try:
        results = []
        
        for request in requests:
            # Get all predictions for this request
            stress_level_result = await predict_stress_level(request)
            stress_intensity_result = await predict_stress_intensity(request)
            panic_prob_result = await predict_panic_probability(request)
            
            results.append({
                "participant_id": request.participant_id,
                "stress_classification": stress_level_result,
                "stress_intensity": stress_intensity_result,
                "panic_probability": panic_prob_result
            })
        
        return {
            "batch_results": results,
            "total_processed": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage endpoint
@app.get("/example/request")
async def get_example_request():
    """Get an example request format."""
    return {
        "example_request": {
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
            "participant_id": "participant_001",
            "timestamp": "2024-01-15T10:30:00"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )