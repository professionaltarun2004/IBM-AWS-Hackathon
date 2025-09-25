"""
Main training and evaluation script for the stress detection ML system.
Orchestrates the complete pipeline from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

from data_preprocessing import DataPreprocessor, load_and_explore_data
from stress_models import (StressClassifier, StressRegressor, TimeSeriesForecaster, 
                          PanicPredictor, ModelEvaluator, save_model)


class StressDetectionPipeline:
    """Complete ML pipeline for stress detection and prediction."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
        self.stress_classifier = StressClassifier()
        self.stress_regressor = StressRegressor()
        self.forecaster = TimeSeriesForecaster()
        self.panic_predictor = PanicPredictor()
        
        self.results = {}
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        print("="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Load data
        df = load_and_explore_data(self.data_path)
        
        # Preprocess data
        df_processed = self.preprocessor.fit_transform(df, include_temporal=True)
        
        print(f"\nFinal processed dataset shape: {df_processed.shape}")
        print(f"Features: {[col for col in df_processed.columns if col != 'PSS_score']}")
        
        return df_processed
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare train-test split with stratification."""
        print("\n" + "="*60)
        print("PREPARING TRAIN-TEST SPLIT")
        print("="*60)
        
        # Separate features and target
        X = df.drop('PSS_score', axis=1)
        y = df['PSS_score']
        
        # Create stress categories for stratification
        stress_categories = self.stress_classifier.create_stress_categories(y)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stress_categories
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Print class distribution
        train_categories = self.stress_classifier.create_stress_categories(y_train)
        test_categories = self.stress_classifier.create_stress_categories(y_test)
        
        print(f"\nTraining set stress distribution:")
        print(train_categories.value_counts())
        print(f"\nTest set stress distribution:")
        print(test_categories.value_counts())
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'feature_names': list(X.columns)
        }
    
    def train_stress_classification(self, data_split: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate stress classification models."""
        print("\n" + "="*60)
        print("TRAINING STRESS CLASSIFICATION MODELS")
        print("="*60)
        
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        
        # Prepare data for classification
        X_train_np, y_train_encoded = self.stress_classifier.prepare_data(X_train, y_train)
        X_test_np, y_test_encoded = self.stress_classifier.prepare_data(X_test, y_test)
        
        # Train models
        training_results = self.stress_classifier.train_models(X_train_np, y_train_encoded)
        
        # Evaluate best model
        evaluation_results = self.stress_classifier.evaluate_model(X_test_np, y_test_encoded)
        
        # Check if accuracy requirement is met (≥85%)
        accuracy_met = evaluation_results['accuracy'] >= 0.85
        print(f"\nAccuracy requirement (≥85%): {'✓ MET' if accuracy_met else '✗ NOT MET'}")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'accuracy_requirement_met': accuracy_met,
            'best_model': self.stress_classifier.best_model,
            'best_model_name': self.stress_classifier.best_model_name
        }
    
    def train_stress_regression(self, data_split: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate stress regression models."""
        print("\n" + "="*60)
        print("TRAINING STRESS REGRESSION MODELS")
        print("="*60)
        
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        
        # Train models
        training_results = self.stress_regressor.train_models(X_train.values, y_train.values)
        
        # Evaluate best model
        evaluation_results = self.stress_regressor.evaluate_model(X_test.values, y_test.values)
        
        # Check if requirements are met (RMSE ≤3.0, R² ≥0.75)
        rmse_met = evaluation_results['rmse'] <= 3.0
        r2_met = evaluation_results['r2_score'] >= 0.75
        
        print(f"\nRMSE requirement (≤3.0): {'✓ MET' if rmse_met else '✗ NOT MET'}")
        print(f"R² requirement (≥0.75): {'✓ MET' if r2_met else '✗ NOT MET'}")
        
        # Get feature importance
        feature_importance = self.stress_regressor.get_feature_importance()
        if feature_importance is not None:
            # Create feature importance ranking
            feature_names = data_split['feature_names']
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Most Important Features:")
            print(importance_df.head().to_string(index=False))
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'rmse_requirement_met': rmse_met,
            'r2_requirement_met': r2_met,
            'feature_importance': importance_df if feature_importance is not None else None,
            'best_model': self.stress_regressor.best_model,
            'best_model_name': self.stress_regressor.best_model_name
        }
    
    def train_forecasting_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train and evaluate time-series forecasting model."""
        print("\n" + "="*60)
        print("TRAINING TIME-SERIES FORECASTING MODEL")
        print("="*60)
        
        # Prepare sequences
        try:
            X_sequences, y_sequences = self.forecaster.prepare_sequences(df)
            
            if len(X_sequences) == 0:
                print("Not enough data for time-series forecasting")
                return {'error': 'Insufficient data for time-series modeling'}
            
            # Split sequences
            split_idx = int(0.8 * len(X_sequences))
            X_train_seq = X_sequences[:split_idx]
            X_test_seq = X_sequences[split_idx:]
            y_train_seq = y_sequences[:split_idx]
            y_test_seq = y_sequences[split_idx:]
            
            print(f"Training sequences: {X_train_seq.shape}")
            print(f"Test sequences: {X_test_seq.shape}")
            
            # Train model
            training_results = self.forecaster.train_model(X_train_seq, y_train_seq)
            
            # Evaluate model
            evaluation_results = self.forecaster.evaluate_model(X_test_seq, y_test_seq)
            
            # Check if MAE requirement is met (≤2.5)
            mae_met = evaluation_results['mae'] <= 2.5
            print(f"\nMAE requirement (≤2.5): {'✓ MET' if mae_met else '✗ NOT MET'}")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'mae_requirement_met': mae_met,
                'model': self.forecaster.model
            }
            
        except Exception as e:
            print(f"Error in forecasting model training: {str(e)}")
            return {'error': str(e)}
    
    def train_panic_prediction(self, data_split: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate panic attack prediction models."""
        print("\n" + "="*60)
        print("TRAINING PANIC ATTACK PREDICTION MODELS")
        print("="*60)
        
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        
        # Check if we have any panic cases (PSS > 35)
        panic_cases_train = sum(y_train > 35)
        panic_cases_test = sum(y_test > 35)
        
        print(f"Panic cases in training set: {panic_cases_train}")
        print(f"Panic cases in test set: {panic_cases_test}")
        
        if panic_cases_train == 0:
            print("No panic cases found in training data. Using lower threshold (PSS > 30)")
            self.panic_predictor.panic_threshold = 30
        
        # Train models
        training_results = self.panic_predictor.train_models(X_train.values, y_train)
        
        # Evaluate best model
        evaluation_results = self.panic_predictor.evaluate_model(X_test.values, y_test)
        
        # Check if PR-AUC requirement is met (≥0.80)
        pr_auc_met = evaluation_results['pr_auc'] >= 0.80
        print(f"\nPR-AUC requirement (≥0.80): {'✓ MET' if pr_auc_met else '✗ NOT MET'}")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'pr_auc_requirement_met': pr_auc_met,
            'best_model': self.panic_predictor.best_model,
            'best_model_name': self.panic_predictor.best_model_name
        }
    
    def create_summary_report(self) -> str:
        """Create a comprehensive summary report of all results."""
        report = []
        report.append("="*80)
        report.append("STRESS DETECTION ML SYSTEM - FINAL RESULTS SUMMARY")
        report.append("="*80)
        
        # Classification results
        if 'classification' in self.results:
            cls_results = self.results['classification']
            report.append(f"\n1. STRESS LEVEL CLASSIFICATION")
            report.append(f"   Best Model: {cls_results['best_model_name']}")
            report.append(f"   Accuracy: {cls_results['evaluation_results']['accuracy']:.3f}")
            report.append(f"   F1-Score: {cls_results['evaluation_results']['f1_score']:.3f}")
            report.append(f"   ROC-AUC: {cls_results['evaluation_results']['roc_auc']:.3f}")
            report.append(f"   Requirement Met (>=85%): {'YES' if cls_results['accuracy_requirement_met'] else 'NO'}")
        
        # Regression results
        if 'regression' in self.results:
            reg_results = self.results['regression']
            report.append(f"\n2. STRESS INTENSITY REGRESSION")
            report.append(f"   Best Model: {reg_results['best_model_name']}")
            report.append(f"   RMSE: {reg_results['evaluation_results']['rmse']:.3f}")
            report.append(f"   MAE: {reg_results['evaluation_results']['mae']:.3f}")
            report.append(f"   R²: {reg_results['evaluation_results']['r2_score']:.3f}")
            report.append(f"   RMSE Requirement Met (<=3.0): {'YES' if reg_results['rmse_requirement_met'] else 'NO'}")
            report.append(f"   R2 Requirement Met (>=0.75): {'YES' if reg_results['r2_requirement_met'] else 'NO'}")
            
            if reg_results['feature_importance'] is not None:
                report.append(f"\n   Top 5 Important Features:")
                for i, row in reg_results['feature_importance'].head().iterrows():
                    report.append(f"   - {row['feature']}: {row['importance']:.3f}")
        
        # Forecasting results
        if 'forecasting' in self.results and 'error' not in self.results['forecasting']:
            fore_results = self.results['forecasting']
            report.append(f"\n3. NEXT-DAY STRESS FORECASTING")
            report.append(f"   MAE: {fore_results['evaluation_results']['mae']:.3f}")
            report.append(f"   RMSE: {fore_results['evaluation_results']['rmse']:.3f}")
            report.append(f"   MAE Requirement Met (<=2.5): {'YES' if fore_results['mae_requirement_met'] else 'NO'}")
        
        # Panic prediction results
        if 'panic' in self.results:
            panic_results = self.results['panic']
            report.append(f"\n4. PANIC ATTACK PREDICTION")
            report.append(f"   Best Model: {panic_results['best_model_name']}")
            report.append(f"   PR-AUC: {panic_results['evaluation_results']['pr_auc']:.3f}")
            report.append(f"   ROC-AUC: {panic_results['evaluation_results']['roc_auc']:.3f}")
            report.append(f"   PR-AUC Requirement Met (>=0.80): {'YES' if panic_results['pr_auc_requirement_met'] else 'NO'}")
        
        report.append(f"\n" + "="*80)
        
        return "\n".join(report)
    
    def save_models(self, model_dir: str = "models"):
        """Save all trained models."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print(f"\nSaving models to {model_dir}/...")
        
        # Save preprocessor
        save_model(self.preprocessor, f"{model_dir}/preprocessor.joblib")
        
        # Save classification model
        if hasattr(self.stress_classifier, 'best_model') and self.stress_classifier.best_model:
            save_model(self.stress_classifier, f"{model_dir}/stress_classifier.joblib")
        
        # Save regression model
        if hasattr(self.stress_regressor, 'best_model') and self.stress_regressor.best_model:
            save_model(self.stress_regressor, f"{model_dir}/stress_regressor.joblib")
        
        # Save forecasting model
        if hasattr(self.forecaster, 'model') and self.forecaster.model:
            self.forecaster.model.save(f"{model_dir}/forecasting_model.h5")
        
        # Save panic prediction model
        if hasattr(self.panic_predictor, 'best_model') and self.panic_predictor.best_model:
            save_model(self.panic_predictor, f"{model_dir}/panic_predictor.joblib")
        
        print("All models saved successfully!")
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline."""
        print("Starting Complete Stress Detection ML Pipeline...")
        
        # 1. Load and preprocess data
        df_processed = self.load_and_preprocess_data()
        
        # 2. Prepare train-test split
        data_split = self.prepare_train_test_split(df_processed)
        
        # 3. Train classification models
        self.results['classification'] = self.train_stress_classification(data_split)
        
        # 4. Train regression models
        self.results['regression'] = self.train_stress_regression(data_split)
        
        # 5. Train forecasting model
        self.results['forecasting'] = self.train_forecasting_model(df_processed)
        
        # 6. Train panic prediction models
        self.results['panic'] = self.train_panic_prediction(data_split)
        
        # 7. Create and display summary report
        summary_report = self.create_summary_report()
        print(summary_report)
        
        # 8. Save models
        self.save_models()
        
        # 9. Save summary report
        with open("model_evaluation_report.txt", "w", encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"\nPipeline completed! Summary report saved to 'model_evaluation_report.txt'")
        
        return self.results


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = StressDetectionPipeline("stress_detection.csv")
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()