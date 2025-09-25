# Requirements Document

## Introduction

The Good Doctor ML system is designed to provide comprehensive stress monitoring and panic attack prediction for autistic children using wearable devices and Unity VR data. The system will analyze physiological, behavioral, and contextual data to detect current stress levels, predict future panic attacks, and provide personalized risk assessments to help caregivers and healthcare providers intervene proactively.

## Requirements

### Requirement 1: Stress Level Classification

**User Story:** As a healthcare provider, I want to classify current stress levels from real-time physiological and behavioral data, so that I can provide immediate interventions when stress levels are elevated.

#### Acceptance Criteria

1. WHEN physiological data (skin conductance, accelerometer) and behavioral data (phone usage, mobility) are provided THEN the system SHALL classify stress as Low (0-13), Moderate (14-26), or High (27-40) based on PSS score ranges
2. WHEN classification is performed THEN the system SHALL achieve minimum 85% accuracy on test data
3. WHEN real-time data is received THEN the system SHALL provide classification results within 2 seconds
4. IF input data is missing or incomplete THEN the system SHALL handle missing values through median/mean imputation and still provide a classification

### Requirement 2: Stress Intensity Prediction

**User Story:** As a researcher, I want to predict continuous stress intensity scores, so that I can understand nuanced stress patterns and track gradual changes over time.

#### Acceptance Criteria

1. WHEN physiological and behavioral features are provided THEN the system SHALL predict a continuous PSS score between 0-40
2. WHEN regression prediction is performed THEN the system SHALL achieve RMSE ≤ 3.0 and R² ≥ 0.75 on test data
3. WHEN predictions are made THEN the system SHALL provide confidence intervals for the predicted scores
4. IF feature importance analysis is requested THEN the system SHALL identify and rank the top 5 most predictive features

### Requirement 3: Next-Day Stress Forecasting

**User Story:** As a caregiver, I want to predict tomorrow's stress levels based on today's data, so that I can prepare appropriate support and interventions in advance.

#### Acceptance Criteria

1. WHEN current day features are provided THEN the system SHALL predict next-day PSS score
2. WHEN forecasting is performed THEN the system SHALL achieve MAE ≤ 2.5 on next-day predictions
3. WHEN time-series data spans multiple days THEN the system SHALL use sliding window approach to incorporate temporal patterns
4. IF insufficient historical data exists THEN the system SHALL provide predictions with appropriate uncertainty indicators

### Requirement 4: Panic Attack Probability Prediction

**User Story:** As a healthcare provider, I want to predict the probability of a panic attack occurring within a specified time window, so that I can implement preventive measures and alert appropriate personnel.

#### Acceptance Criteria

1. WHEN sliding time-series windows of physiological signals are analyzed THEN the system SHALL predict probability of panic attack within next 30 minutes, 2 hours, and 24 hours
2. WHEN panic prediction is performed THEN the system SHALL achieve Precision-Recall AUC ≥ 0.80 for 30-minute predictions
3. WHEN high-risk probability (>0.7) is detected THEN the system SHALL trigger immediate alerts through configured notification channels
4. IF real-time monitoring is active THEN the system SHALL update panic probabilities every 5 minutes

### Requirement 5: Data Preprocessing and Feature Engineering

**User Story:** As a data scientist, I want automated data preprocessing and feature engineering, so that raw sensor data can be transformed into meaningful predictive features.

#### Acceptance Criteria

1. WHEN raw CSV data is ingested THEN the system SHALL automatically drop non-predictive identifiers (participant_id, day, raw timestamps)
2. WHEN missing values are detected THEN the system SHALL apply appropriate imputation strategies (median for numerical, mode for categorical)
3. WHEN feature engineering is performed THEN the system SHALL create derived features including sleep efficiency, social load index, and mobility index
4. WHEN data normalization is required THEN the system SHALL apply z-score scaling to all numerical features

### Requirement 6: Model Training and Evaluation

**User Story:** As a machine learning engineer, I want to train and evaluate multiple ML models, so that I can select the best performing algorithm for each prediction task.

#### Acceptance Criteria

1. WHEN model training is initiated THEN the system SHALL train Random Forest, XGBoost, LSTM, and Transformer models
2. WHEN data splitting is performed THEN the system SHALL use 80% for training and 20% for testing with stratified sampling
3. WHEN model evaluation is completed THEN the system SHALL report accuracy, F1-score, and ROC-AUC for classification tasks
4. WHEN regression evaluation is completed THEN the system SHALL report RMSE, MAE, and R² metrics

### Requirement 7: Feature Importance and Explainability

**User Story:** As a healthcare researcher, I want to understand which features are most predictive of stress and panic attacks, so that I can validate the clinical relevance of the model and guide future data collection.

#### Acceptance Criteria

1. WHEN model training is complete THEN the system SHALL generate SHAP values for feature importance analysis
2. WHEN feature ranking is requested THEN the system SHALL identify and rank all features by their predictive importance
3. WHEN explainability reports are generated THEN the system SHALL highlight the top 5 most important features for each prediction type
4. IF clinical validation is needed THEN the system SHALL provide interpretable explanations for individual predictions

### Requirement 8: Real-time Data Integration

**User Story:** As a system administrator, I want to integrate real-time data from wearable devices and Unity VR systems, so that the ML models can provide continuous monitoring and predictions.

#### Acceptance Criteria

1. WHEN wearable device data is received THEN the system SHALL ingest physiological signals (skin conductance, accelerometer) in real-time
2. WHEN Unity VR session data is available THEN the system SHALL incorporate behavioral and contextual features
3. WHEN data ingestion occurs THEN the system SHALL validate data quality and handle streaming data with <1 second latency
4. IF data stream interruption occurs THEN the system SHALL maintain prediction capability using last known values and provide appropriate warnings

### Requirement 9: Personalized Risk Scoring

**User Story:** As a caregiver of an autistic child, I want personalized risk scores that account for individual differences, so that interventions can be tailored to each child's specific needs and patterns.

#### Acceptance Criteria

1. WHEN personality features (Big Five traits) are available THEN the system SHALL incorporate them into personalized models
2. WHEN individual historical data exists THEN the system SHALL adapt predictions based on personal baseline patterns
3. WHEN personalized scores are generated THEN the system SHALL provide individual risk thresholds rather than population-based thresholds
4. IF insufficient personal data exists THEN the system SHALL use population models with confidence adjustments

### Requirement 10: Alert and Notification System

**User Story:** As a healthcare provider, I want automated alerts when high stress or panic risk is detected, so that I can respond quickly to provide appropriate care and intervention.

#### Acceptance Criteria

1. WHEN stress level is classified as High (27-40 PSS) THEN the system SHALL send immediate notifications to configured recipients
2. WHEN panic probability exceeds 0.7 THEN the system SHALL trigger emergency alert protocols
3. WHEN alerts are sent THEN the system SHALL include current readings, trend information, and recommended actions
4. IF alert fatigue is detected (>10 alerts per day) THEN the system SHALL adjust sensitivity thresholds and provide summary reports instead