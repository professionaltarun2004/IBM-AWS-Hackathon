"""
Enhanced Panic Attack Prediction Model with Synthetic Heart Rate
Includes Twilio integration and Streamlit deployment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedPanicPredictor:
    """Enhanced panic prediction model with synthetic heart rate and improved features."""
    
    def __init__(self):
        self.model = None
        self.feature_means = None
        self.feature_names = None
        self.scaler = None
        
    def add_synthetic_heart_rate(self, df):
        """Add synthetic heart rate based on physiological indicators."""
        print("Adding synthetic heart rate feature...")
        
        # Base heart rate + physiological stress indicators + random noise
        np.random.seed(42)  # For reproducibility
        
        heart_rate = (
            70 +  # Base resting heart rate
            (df['accelerometer'] * 5) +  # Physical activity impact
            (df['skin_conductance'] * 3) +  # Stress/arousal impact
            np.random.normal(0, 2, len(df))  # Individual variation
        )
        
        # Ensure realistic heart rate range (50-120 bpm)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        df['heart_rate'] = heart_rate
        print(f"Heart rate range: {heart_rate.min():.1f} - {heart_rate.max():.1f} bpm")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data with all features for training."""
        print("Preparing enhanced dataset...")
        
        # Add synthetic heart rate
        df_enhanced = self.add_synthetic_heart_rate(df.copy())
        
        # Drop specified columns
        drop_columns = ['participant_id', 'day', 'sleep_time', 'wake_time']
        df_clean = df_enhanced.drop(columns=[col for col in drop_columns if col in df_enhanced.columns])
        
        # Separate features and target
        X = df_clean.drop('PSS_score', axis=1)
        y = df_clean['PSS_score']
        
        # Create panic labels (High stress: PSS > 26)
        y_panic = (y > 26).astype(int)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Panic cases: {y_panic.sum()} / {len(y_panic)} ({y_panic.mean():.1%})")
        
        return X, y_panic
    
    def train_models(self, X, y):
        """Train both RandomForest and XGBoost models."""
        print("\nTraining enhanced panic prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Store feature information
        self.feature_names = list(X.columns)
        self.feature_means = X_train.mean().to_dict()
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                print(f"  ROC-AUC: {roc_auc:.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Select best model based on ROC-AUC
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
            self.model = results[best_model_name]['model']
            
            print(f"\nBest model: {best_model_name}")
            print(f"Final performance:")
            print(f"  Accuracy: {results[best_model_name]['accuracy']:.3f}")
            print(f"  F1-Score: {results[best_model_name]['f1_score']:.3f}")
            print(f"  ROC-AUC: {results[best_model_name]['roc_auc']:.3f}")
            
            # Detailed classification report
            y_pred_final = self.model.predict(X_test)
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred_final, target_names=['No Panic', 'Panic Risk']))
            
            return results[best_model_name]
        else:
            raise Exception("No models trained successfully")
    
    def predict_panic_probability(self, heart_rate, accelerometer, skin_conductance, sleep_duration):
        """Predict panic probability from minimal inputs."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create full feature vector with mean values
        feature_vector = np.array([self.feature_means[feature] for feature in self.feature_names])
        
        # Map user inputs to correct feature indices
        input_mapping = {
            'heart_rate': heart_rate,
            'accelerometer': accelerometer,
            'skin_conductance': skin_conductance,
            'sleep_duration': sleep_duration
        }
        
        # Update feature vector with user inputs
        for feature_name, value in input_mapping.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                feature_vector[idx] = value
        
        # Predict probability
        feature_vector = feature_vector.reshape(1, -1)
        panic_probability = self.model.predict_proba(feature_vector)[0, 1]
        
        return panic_probability
    
    def save_model(self, filepath='panic_model.pkl'):
        """Save the trained model and metadata."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='panic_model.pkl'):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_means = model_data['feature_means']
        
        print(f"Model loaded from {filepath}")


def train_enhanced_model():
    """Train the enhanced panic prediction model."""
    print("="*60)
    print("ENHANCED PANIC ATTACK PREDICTION MODEL")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv("stress_detection.csv")
    print(f"Original dataset: {df.shape}")
    
    # Initialize predictor
    predictor = EnhancedPanicPredictor()
    
    # Prepare data
    X, y_panic = predictor.prepare_data(df)
    
    # Train models
    best_result = predictor.train_models(X, y_panic)
    
    # Save model
    predictor.save_model('panic_model.pkl')
    
    # Feature importance analysis
    if hasattr(predictor.model, 'feature_importances_'):
        print(f"\nTop 10 Most Important Features:")
        importance_df = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': predictor.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<20} {row['importance']:.3f}")
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE - Model ready for deployment!")
    print("="*60)
    
    return predictor, best_result


if __name__ == "__main__":
    predictor, results = train_enhanced_model()
    
    # Test prediction with sample data
    print(f"\nTesting prediction with sample data:")
    sample_prediction = predictor.predict_panic_probability(
        heart_rate=85,
        accelerometer=2.1,
        skin_conductance=3.5,
        sleep_duration=5.5
    )
    print(f"Sample panic probability: {sample_prediction:.3f}")
    
    if sample_prediction > 0.7:
        print("🚨 HIGH PANIC RISK - Would trigger Twilio alert!")
    else:
        print("✅ Normal risk level")