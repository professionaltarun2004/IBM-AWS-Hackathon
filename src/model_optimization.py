"""
Model optimization and feature importance analysis for stress detection system.
Includes SHAP analysis, hyperparameter tuning, and performance improvements.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import joblib

from data_preprocessing import DataPreprocessor
from stress_models import StressClassifier, StressRegressor


class ModelOptimizer:
    """Advanced model optimization and analysis."""
    
    def __init__(self):
        self.best_params = {}
        self.shap_explainer = None
        self.feature_importance_df = None
    
    def optimize_random_forest_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest classifier with hyperparameter tuning."""
        print("Optimizing Random Forest Classifier...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params['classifier'] = grid_search.best_params_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def optimize_random_forest_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest regressor with hyperparameter tuning."""
        print("Optimizing Random Forest Regressor...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        rf = RandomForestRegressor(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params['regressor'] = grid_search.best_params_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.3f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def analyze_feature_importance_shap(self, model, X_train: np.ndarray, 
                                       feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature importance using SHAP values."""
        print("Analyzing feature importance with SHAP...")
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (use subset for performance)
        sample_size = min(100, X_train.shape[0])
        X_sample = X_train[:sample_size]
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # For multiclass classification, use the first class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap_values,
            'model_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else [0] * len(feature_names)
        }).sort_values('shap_importance', ascending=False)
        
        print("Top 10 Most Important Features (SHAP):")
        print(self.feature_importance_df.head(10).to_string(index=False))
        
        return self.feature_importance_df
    
    def create_feature_importance_plot(self, save_path: str = None):
        """Create feature importance visualization."""
        if self.feature_importance_df is None:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot top 15 features
        top_features = self.feature_importance_df.head(15)
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(top_features)), top_features['shap_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('SHAP Importance')
        plt.title('Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        plt.barh(range(len(top_features)), top_features['model_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Model Importance')
        plt.title('Feature Importance (Model)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def feature_selection_analysis(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 feature_names: List[str], model_type: str = 'classifier') -> Dict[str, Any]:
        """Analyze performance with different numbers of features."""
        print(f"Performing feature selection analysis for {model_type}...")
        
        if self.feature_importance_df is None:
            print("No feature importance data available")
            return {}
        
        # Test different numbers of top features
        feature_counts = [5, 10, 15, 20, 25, len(feature_names)]
        results = []
        
        for n_features in feature_counts:
            if n_features > len(feature_names):
                continue
                
            # Select top n features
            top_features = self.feature_importance_df.head(n_features)['feature'].tolist()
            feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]
            
            X_selected = X_train[:, feature_indices]
            
            # Train model with selected features
            if model_type == 'classifier':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_selected, y_train)
                score = model.score(X_selected, y_train)
                metric_name = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_selected, y_train)
                y_pred = model.predict(X_selected)
                score = r2_score(y_train, y_pred)
                metric_name = 'r2_score'
            
            results.append({
                'n_features': n_features,
                'features': top_features,
                metric_name: score
            })
            
            print(f"Features: {n_features}, {metric_name}: {score:.3f}")
        
        return {
            'results': results,
            'best_n_features': max(results, key=lambda x: x[metric_name])['n_features']
        }


class AdvancedPreprocessor:
    """Advanced preprocessing with feature selection and engineering."""
    
    def __init__(self):
        self.base_preprocessor = DataPreprocessor()
        self.selected_features = None
        
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key variables."""
        from sklearn.preprocessing import PolynomialFeatures
        
        # Select key features for polynomial expansion
        key_features = ['skin_conductance', 'sleep_duration', 'PSQI_score', 'screen_on_time']
        available_features = [f for f in key_features if f in df.columns]
        
        if not available_features:
            return df
        
        df_poly = df.copy()
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[available_features])
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(available_features)
        
        # Add polynomial features to dataframe
        for i, name in enumerate(poly_feature_names):
            if name not in available_features:  # Don't duplicate original features
                df_poly[f'poly_{name}'] = poly_features[:, i]
        
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful interaction features."""
        df_interact = df.copy()
        
        # Stress-sleep interaction
        if all(col in df.columns for col in ['PSS_score', 'sleep_duration']):
            df_interact['stress_sleep_interaction'] = df['PSS_score'] * (1 / (df['sleep_duration'] + 1))
        
        # Social-mobility interaction
        if all(col in df.columns for col in ['social_load_index', 'mobility_index']):
            df_interact['social_mobility_interaction'] = df['social_load_index'] * df['mobility_index']
        
        # Personality-stress interaction
        personality_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        available_personality = [col for col in personality_cols if col in df.columns]
        
        if len(available_personality) > 0 and 'PSS_score' in df.columns:
            for trait in available_personality:
                df_interact[f'{trait}_stress_interaction'] = df[trait] * df['PSS_score']
        
        return df_interact
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced feature engineering techniques."""
        print("Applying advanced feature engineering...")
        
        # Base preprocessing
        df_processed = self.base_preprocessor.fit_transform(df, include_temporal=True)
        
        # Add polynomial features
        df_processed = self.create_polynomial_features(df_processed)
        
        # Add interaction features
        df_processed = self.create_interaction_features(df_processed)
        
        print(f"Advanced preprocessing completed. Final shape: {df_processed.shape}")
        
        return df_processed


def run_optimization_pipeline(data_path: str = "stress_detection.csv"):
    """Run complete optimization pipeline."""
    print("="*60)
    print("ADVANCED MODEL OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Advanced preprocessing
    advanced_preprocessor = AdvancedPreprocessor()
    df_processed = advanced_preprocessor.advanced_feature_engineering(df)
    
    # Prepare data
    X = df_processed.drop('PSS_score', axis=1)
    y = df_processed['PSS_score']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    feature_names = list(X.columns)
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Optimize classifier
    print("\n" + "="*40)
    print("OPTIMIZING CLASSIFICATION MODEL")
    print("="*40)
    
    # Create stress categories
    stress_classifier = StressClassifier()
    _, y_train_cat = stress_classifier.prepare_data(X_train, y_train)
    _, y_test_cat = stress_classifier.prepare_data(X_test, y_test)
    
    # Optimize classifier
    clf_results = optimizer.optimize_random_forest_classifier(X_train.values, y_train_cat)
    
    # Evaluate optimized classifier
    y_pred_cat = clf_results['best_model'].predict(X_test.values)
    optimized_accuracy = accuracy_score(y_test_cat, y_pred_cat)
    print(f"Optimized classifier accuracy: {optimized_accuracy:.3f}")
    
    # SHAP analysis for classifier
    clf_importance = optimizer.analyze_feature_importance_shap(
        clf_results['best_model'], X_train.values, feature_names
    )
    
    # Optimize regressor
    print("\n" + "="*40)
    print("OPTIMIZING REGRESSION MODEL")
    print("="*40)
    
    reg_results = optimizer.optimize_random_forest_regressor(X_train.values, y_train.values)
    
    # Evaluate optimized regressor
    y_pred_reg = reg_results['best_model'].predict(X_test.values)
    optimized_rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    optimized_r2 = r2_score(y_test, y_pred_reg)
    print(f"Optimized regressor RMSE: {optimized_rmse:.3f}")
    print(f"Optimized regressor R²: {optimized_r2:.3f}")
    
    # SHAP analysis for regressor
    reg_importance = optimizer.analyze_feature_importance_shap(
        reg_results['best_model'], X_train.values, feature_names
    )
    
    # Feature selection analysis
    print("\n" + "="*40)
    print("FEATURE SELECTION ANALYSIS")
    print("="*40)
    
    clf_feature_analysis = optimizer.feature_selection_analysis(
        X_train.values, y_train_cat, feature_names, 'classifier'
    )
    
    reg_feature_analysis = optimizer.feature_selection_analysis(
        X_train.values, y_train.values, feature_names, 'regressor'
    )
    
    # Create visualizations
    optimizer.create_feature_importance_plot('feature_importance_analysis.png')
    
    # Save optimized models
    print("\nSaving optimized models...")
    joblib.dump(clf_results['best_model'], 'models/optimized_classifier.joblib')
    joblib.dump(reg_results['best_model'], 'models/optimized_regressor.joblib')
    joblib.dump(advanced_preprocessor, 'models/advanced_preprocessor.joblib')
    
    # Create optimization report
    optimization_report = f"""
ADVANCED MODEL OPTIMIZATION RESULTS
=====================================

CLASSIFICATION OPTIMIZATION:
- Original Accuracy: ~0.612
- Optimized Accuracy: {optimized_accuracy:.3f}
- Improvement: {optimized_accuracy - 0.612:.3f}
- Best Parameters: {clf_results['best_params']}

REGRESSION OPTIMIZATION:
- Original RMSE: ~6.227
- Optimized RMSE: {optimized_rmse:.3f}
- Improvement: {6.227 - optimized_rmse:.3f}
- Original R²: ~0.478
- Optimized R²: {optimized_r2:.3f}
- Improvement: {optimized_r2 - 0.478:.3f}
- Best Parameters: {reg_results['best_params']}

TOP 10 MOST IMPORTANT FEATURES:
{clf_importance.head(10)[['feature', 'shap_importance']].to_string(index=False)}

FEATURE SELECTION RECOMMENDATIONS:
- Optimal features for classification: {clf_feature_analysis.get('best_n_features', 'N/A')}
- Optimal features for regression: {reg_feature_analysis.get('best_n_features', 'N/A')}

DATASET STATISTICS:
- Total features after advanced engineering: {df_processed.shape[1]}
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
"""
    
    with open('optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(optimization_report)
    
    print("Optimization pipeline completed!")
    print("Reports saved: optimization_report.txt, feature_importance_analysis.png")
    
    return {
        'classifier_results': clf_results,
        'regressor_results': reg_results,
        'feature_importance': clf_importance,
        'optimization_report': optimization_report
    }


if __name__ == "__main__":
    results = run_optimization_pipeline()