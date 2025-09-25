"""
Simple test to verify model loading and basic functionality.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, load_and_explore_data
from stress_models import StressClassifier, StressRegressor, PanicPredictor

def test_model_loading():
    """Test that all models can be loaded successfully."""
    print("Testing model loading...")
    
    try:
        # Load preprocessor
        preprocessor = joblib.load("models/preprocessor.joblib")
        print("✓ Preprocessor loaded successfully")
        
        # Load classifier
        classifier = joblib.load("models/stress_classifier.joblib")
        print("✓ Stress classifier loaded successfully")
        
        # Load regressor
        regressor = joblib.load("models/stress_regressor.joblib")
        print("✓ Stress regressor loaded successfully")
        
        # Load panic predictor
        panic_predictor = joblib.load("models/panic_predictor.joblib")
        print("✓ Panic predictor loaded successfully")
        
        return True, (preprocessor, classifier, regressor, panic_predictor)
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False, None

def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("\nTesting prediction pipeline...")
    
    # Load models
    success, models = test_model_loading()
    if not success:
        return False
    
    preprocessor, classifier, regressor, panic_predictor = models
    
    try:
        # Create sample data
        sample_data = {
            'participant_id': [1],
            'day': [1],
            'PSS_score': [25],
            'Openness': [3.2],
            'Conscientiousness': [4.1],
            'Extraversion': [2.8],
            'Agreeableness': [3.9],
            'Neuroticism': [2.3],
            'sleep_time': [7.5],
            'wake_time': [6.0],
            'sleep_duration': [7.5],
            'PSQI_score': [3],
            'call_duration': [15.5],
            'num_calls': [5],
            'num_sms': [12],
            'screen_on_time': [6.5],
            'skin_conductance': [2.5],
            'accelerometer': [1.2],
            'mobility_radius': [2.1],
            'mobility_distance': [5.3]
        }
        
        df = pd.DataFrame(sample_data)
        print("✓ Sample data created")
        
        # Preprocess data
        df_processed = preprocessor.transform(df, include_temporal=True)
        X = df_processed.drop('PSS_score', axis=1)
        y = df_processed['PSS_score']
        print("✓ Data preprocessing completed")
        
        # Test classification
        X_np, y_encoded = classifier.prepare_data(X, y)
        stress_levels, probabilities = classifier.predict(X_np)
        print(f"✓ Stress classification: {stress_levels[0]} (confidence: {max(probabilities[0]):.3f})")
        
        # Test regression
        pss_pred, uncertainty = regressor.predict_with_confidence(X.values)
        print(f"✓ Stress intensity: {pss_pred[0]:.2f} ± {uncertainty[0]:.2f}")
        
        # Test panic prediction
        panic_prob = panic_predictor.predict_panic_probability(X.values)
        print(f"✓ Panic probability: {panic_prob[0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in prediction pipeline: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing with the original dataset."""
    print("\nTesting data preprocessing with full dataset...")
    
    try:
        # Load original data
        df = pd.read_csv("stress_detection.csv")
        print(f"✓ Original dataset loaded: {df.shape}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        df_processed = preprocessor.fit_transform(df, include_temporal=True)
        print(f"✓ Data processed: {df_processed.shape}")
        
        # Check for NaN values
        nan_count = df_processed.isnull().sum().sum()
        if nan_count == 0:
            print("✓ No NaN values in processed data")
        else:
            print(f"⚠️  {nan_count} NaN values found in processed data")
        
        # Check feature scaling
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'PSS_score']
        
        if len(numerical_cols) > 0:
            means = df_processed[numerical_cols].mean()
            stds = df_processed[numerical_cols].std()
            
            # Check if features are approximately standardized (mean~0, std~1)
            mean_close_to_zero = abs(means.mean()) < 0.1
            std_close_to_one = abs(stds.mean() - 1.0) < 0.1
            
            if mean_close_to_zero and std_close_to_one:
                print("✓ Features properly standardized")
            else:
                print(f"⚠️  Features may not be properly standardized (mean: {means.mean():.3f}, std: {stds.mean():.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data preprocessing test: {e}")
        return False

def run_simple_tests():
    """Run all simple tests."""
    print("="*60)
    print("SIMPLE FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Data Preprocessing", test_data_preprocessing),
        ("Prediction Pipeline", test_prediction_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running {test_name} Test")
        print(f"{'-'*40}")
        
        try:
            if test_name == "Model Loading":
                result, _ = test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return results

if __name__ == "__main__":
    results = run_simple_tests()