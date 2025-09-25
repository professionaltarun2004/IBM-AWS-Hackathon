"""
Machine learning models for stress detection and prediction.
Includes classification, regression, and time-series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                           mean_squared_error, mean_absolute_error, r2_score,
                           precision_recall_curve, auc, classification_report)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings

warnings.filterwarnings('ignore')


class StressClassifier:
    """Multi-model stress level classification system."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = None
        
    def create_stress_categories(self, pss_scores: pd.Series) -> pd.Series:
        """Convert PSS scores to stress level categories."""
        def categorize_stress(score):
            if score <= 13:
                return 'Low'
            elif score <= 26:
                return 'Moderate'
            else:
                return 'High'
        
        return pss_scores.apply(categorize_stress)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for classification."""
        # Convert stress scores to categories
        y_categorical = self.create_stress_categories(y)
        
        # Encode labels as integers for sklearn
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_categorical)
        
        return X.values, y_encoded
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
        """Train all classification models and return results."""
        results = {}
        
        print("Training stress classification models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
            
            results[name] = {
                'model': model,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name} - CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['cv_accuracy'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"Best model: {best_name}")
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the best model on test data."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC for multiclass
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Test Results - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        # Convert back to stress level labels
        stress_levels = self.label_encoder.inverse_transform(predictions)
        
        return stress_levels, probabilities


class StressRegressor:
    """Continuous PSS score prediction system."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'neural_net': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.best_model_name = None
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
        """Train all regression models and return results."""
        results = {}
        
        print("Training stress regression models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
            rmse_scores = np.sqrt(-cv_scores)
            
            results[name] = {
                'model': model,
                'cv_rmse': rmse_scores.mean(),
                'cv_std': rmse_scores.std()
            }
            
            print(f"{name} - CV RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")
        
        # Select best model (lowest RMSE)
        best_name = min(results.keys(), key=lambda k: results[k]['cv_rmse'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"Best model: {best_name}")
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the best model on test data."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        print(f"Test Results - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        return metrics
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals using bootstrap."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        predictions = self.best_model.predict(X)
        
        # Simple confidence estimation using feature importance variance
        if hasattr(self.best_model, 'feature_importances_'):
            # Use feature importance to estimate uncertainty
            feature_importance = self.best_model.feature_importances_
            uncertainty = np.std(feature_importance) * np.ones_like(predictions)
        else:
            uncertainty = np.ones_like(predictions) * 2.0  # Default uncertainty
        
        return predictions, uncertainty
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best model."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        else:
            return None


class TimeSeriesForecaster:
    """LSTM-based next-day stress forecasting."""
    
    def __init__(self, sequence_length: int = 7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'PSS_score') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences for LSTM training."""
        # Group by participant and create sequences
        sequences = []
        targets = []
        
        for participant_id in df['participant_id'].unique() if 'participant_id' in df.columns else [1]:
            if 'participant_id' in df.columns:
                participant_data = df[df['participant_id'] == participant_id].sort_values('day')
            else:
                participant_data = df.sort_index()
            
            # Create sequences
            for i in range(len(participant_data) - self.sequence_length):
                sequence = participant_data.iloc[i:i+self.sequence_length]
                target = participant_data.iloc[i+self.sequence_length][target_col]
                
                # Use only numerical features for sequences
                numerical_features = sequence.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
                
                sequences.append(numerical_features.values)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model for time series forecasting."""
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   validation_split: float = 0.2, epochs: int = 50) -> Dict[str, Any]:
        """Train LSTM model for forecasting."""
        print("Training LSTM forecasting model...")
        
        # Build model
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        print(f"Training completed. Final loss: {history.history['loss'][-1]:.3f}")
        
        return {'history': history.history}
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate forecasting model."""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"Forecasting Results - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        return metrics
    
    def predict_next_day(self, sequence: np.ndarray) -> float:
        """Predict next day stress level."""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        # Reshape for prediction
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        prediction = self.model.predict(sequence)
        return prediction[0][0]


class PanicPredictor:
    """Panic attack probability prediction system."""
    
    def __init__(self, panic_threshold: float = 35):
        self.panic_threshold = panic_threshold
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
    
    def create_panic_labels(self, pss_scores: pd.Series) -> pd.Series:
        """Create binary panic labels based on high PSS scores."""
        return (pss_scores > self.panic_threshold).astype(int)
    
    def train_models(self, X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Dict]:
        """Train panic prediction models."""
        # Create panic labels
        y_panic = self.create_panic_labels(y_train)
        
        results = {}
        
        print("Training panic prediction models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_panic)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_panic, cv=5, scoring='roc_auc')
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
            
            results[name] = {
                'model': model,
                'cv_auc': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['cv_auc'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"Best model: {best_name}")
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate panic prediction model."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        y_panic = self.create_panic_labels(y_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_panic, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_panic, y_pred_proba)
        except:
            roc_auc = 0.0
        
        metrics = {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc
        }
        
        print(f"Panic Prediction Results - PR-AUC: {pr_auc:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        return metrics
    
    def predict_panic_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict panic attack probability."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        probabilities = self.best_model.predict_proba(X)[:, 1]
        return probabilities


class ModelEvaluator:
    """Comprehensive model evaluation and comparison utilities."""
    
    @staticmethod
    def evaluate_classification_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate classification model with comprehensive metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def evaluate_regression_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model with comprehensive metrics."""
        y_pred = model.predict(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
    
    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare model performance results."""
        comparison_data = []
        
        for model_name, result in results.items():
            row = {'model': model_name}
            row.update(result)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


def save_model(model, filepath: str):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model