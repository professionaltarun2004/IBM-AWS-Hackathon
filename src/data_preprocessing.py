"""
Data preprocessing module for stress detection ML system.
Handles data cleaning, validation, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class DataCleaner:
    """Handles data cleaning and validation operations."""
    
    def __init__(self):
        self.columns_to_remove = ['participant_id', 'day', 'sleep_time', 'wake_time']
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    def remove_non_predictive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-predictive identifier columns."""
        columns_to_drop = [col for col in self.columns_to_remove if col in df.columns]
        return df.drop(columns=columns_to_drop)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using median for numerical and mode for categorical."""
        df_clean = df.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Impute numerical columns
        if len(numerical_cols) > 0:
            df_clean[numerical_cols] = self.numerical_imputer.fit_transform(df_clean[numerical_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            df_clean[categorical_cols] = self.categorical_imputer.fit_transform(df_clean[categorical_cols])
        
        return df_clean
    
    def validate_data_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate data ranges and consistency."""
        validation_report = {
            'valid': True,
            'issues': []
        }
        
        # Check PSS score range (0-40)
        if 'PSS_score' in df.columns:
            invalid_pss = df[(df['PSS_score'] < 0) | (df['PSS_score'] > 40)]
            if len(invalid_pss) > 0:
                validation_report['issues'].append(f"Invalid PSS scores found: {len(invalid_pss)} records")
                validation_report['valid'] = False
        
        # Check personality traits range (typically 1-5)
        personality_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        for col in personality_cols:
            if col in df.columns:
                invalid_vals = df[(df[col] < 0) | (df[col] > 5)]
                if len(invalid_vals) > 0:
                    validation_report['issues'].append(f"Invalid {col} values: {len(invalid_vals)} records")
        
        # Check sleep duration (should be positive and reasonable)
        if 'sleep_duration' in df.columns:
            invalid_sleep = df[(df['sleep_duration'] < 0) | (df['sleep_duration'] > 24)]
            if len(invalid_sleep) > 0:
                validation_report['issues'].append(f"Invalid sleep duration: {len(invalid_sleep)} records")
        
        return validation_report


class FeatureScaler:
    """Handles feature normalization and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_columns = None
    
    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """Fit scaler and transform features using z-score normalization."""
        if exclude_cols is None:
            exclude_cols = ['PSS_score']  # Don't scale target variable
        
        df_scaled = df.copy()
        
        # Identify columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        self.feature_columns = cols_to_scale
        
        if len(cols_to_scale) > 0:
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            self.fitted = True
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        df_scaled = df.copy()
        if self.feature_columns:
            df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_scaled
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers using IQR method."""
        df_clean = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip outliers to bounds
                df_clean[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df_clean


class FeatureEngineer:
    """Creates derived features to enhance predictive power."""
    
    @staticmethod
    def calculate_sleep_efficiency(sleep_duration: pd.Series, psqi_score: pd.Series) -> pd.Series:
        """Calculate sleep efficiency based on duration and quality."""
        # Sleep efficiency = sleep_duration / (24 - psqi_score_normalized)
        # Higher PSQI means worse sleep, so we normalize it
        psqi_normalized = psqi_score / 21  # PSQI range is 0-21
        available_sleep_time = 24 - (psqi_normalized * 12)  # Adjust available time based on quality
        return sleep_duration / np.maximum(available_sleep_time, 1)  # Avoid division by zero
    
    @staticmethod
    def calculate_social_load_index(num_calls: pd.Series, num_sms: pd.Series, call_duration: pd.Series) -> pd.Series:
        """Calculate social interaction load index."""
        return num_calls + num_sms + (call_duration / 60)
    
    @staticmethod
    def calculate_mobility_index(mobility_distance: pd.Series, mobility_radius: pd.Series) -> pd.Series:
        """Calculate mobility activity index."""
        return mobility_distance / np.maximum(mobility_radius, 0.1)  # Avoid division by zero
    
    @staticmethod
    def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features."""
        df_enhanced = df.copy()
        
        # Sleep efficiency
        if all(col in df.columns for col in ['sleep_duration', 'PSQI_score']):
            df_enhanced['sleep_efficiency'] = FeatureEngineer.calculate_sleep_efficiency(
                df['sleep_duration'], df['PSQI_score']
            )
        
        # Social load index
        if all(col in df.columns for col in ['num_calls', 'num_sms', 'call_duration']):
            df_enhanced['social_load_index'] = FeatureEngineer.calculate_social_load_index(
                df['num_calls'], df['num_sms'], df['call_duration']
            )
        
        # Mobility index
        if all(col in df.columns for col in ['mobility_distance', 'mobility_radius']):
            df_enhanced['mobility_index'] = FeatureEngineer.calculate_mobility_index(
                df['mobility_distance'], df['mobility_radius']
            )
        
        # Screen time normalization (hours per day)
        if 'screen_on_time' in df.columns:
            df_enhanced['screen_time_normalized'] = df['screen_on_time'] / 24
        
        # Activity level from accelerometer
        if 'accelerometer' in df.columns:
            df_enhanced['activity_level'] = df['accelerometer']
        
        return df_enhanced
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, participant_col: str = 'participant_id') -> pd.DataFrame:
        """Create temporal features like rolling averages and trends."""
        if participant_col not in df.columns:
            return df
        
        df_temporal = df.copy()
        
        # Sort by participant and day for temporal calculations
        if 'day' in df.columns:
            df_temporal = df_temporal.sort_values([participant_col, 'day'])
        
        # Rolling averages for key features
        rolling_features = ['PSS_score', 'sleep_duration', 'PSQI_score', 'skin_conductance']
        
        for feature in rolling_features:
            if feature in df_temporal.columns:
                # 3-day rolling average
                df_temporal[f'{feature}_3d_avg'] = df_temporal.groupby(participant_col)[feature].rolling(
                    window=3, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # 7-day rolling average
                df_temporal[f'{feature}_7d_avg'] = df_temporal.groupby(participant_col)[feature].rolling(
                    window=7, min_periods=1
                ).mean().reset_index(0, drop=True)
        
        # Lag features (previous day values)
        lag_features = ['PSS_score', 'sleep_duration', 'PSQI_score']
        for feature in lag_features:
            if feature in df_temporal.columns:
                df_temporal[f'{feature}_lag1'] = df_temporal.groupby(participant_col)[feature].shift(1)
        
        # Fill NaN values created by lag and rolling operations
        df_temporal = df_temporal.bfill().ffill()
        
        return df_temporal


class DataPreprocessor:
    """Main preprocessing pipeline that orchestrates all preprocessing steps."""
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.scaler = FeatureScaler()
        self.engineer = FeatureEngineer()
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame, include_temporal: bool = False) -> pd.DataFrame:
        """Complete preprocessing pipeline for training data."""
        print("Starting data preprocessing pipeline...")
        
        # Step 1: Data cleaning
        print("1. Cleaning data and handling missing values...")
        df_clean = self.cleaner.handle_missing_values(df)
        
        # Step 2: Data validation
        print("2. Validating data ranges...")
        validation_report = self.cleaner.validate_data_ranges(df_clean)
        if not validation_report['valid']:
            print(f"Data validation issues found: {validation_report['issues']}")
        
        # Step 3: Feature engineering
        print("3. Creating derived features...")
        df_enhanced = self.engineer.create_derived_features(df_clean)
        
        # Step 4: Temporal features (if requested)
        if include_temporal:
            print("4. Creating temporal features...")
            df_enhanced = self.engineer.create_temporal_features(df_enhanced)
        
        # Step 5: Remove non-predictive columns
        print("5. Removing non-predictive columns...")
        df_processed = self.cleaner.remove_non_predictive_columns(df_enhanced)
        
        # Step 6: Handle outliers
        print("6. Handling outliers...")
        df_processed = self.scaler.handle_outliers(df_processed)
        
        # Step 7: Feature scaling
        print("7. Scaling features...")
        df_scaled = self.scaler.fit_transform(df_processed)
        
        self.is_fitted = True
        print("Preprocessing pipeline completed!")
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame, include_temporal: bool = False) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply same preprocessing steps (without fitting)
        df_clean = self.cleaner.handle_missing_values(df)
        df_enhanced = self.engineer.create_derived_features(df_clean)
        
        if include_temporal:
            df_enhanced = self.engineer.create_temporal_features(df_enhanced)
        
        df_processed = self.cleaner.remove_non_predictive_columns(df_enhanced)
        df_processed = self.scaler.handle_outliers(df_processed)
        df_scaled = self.scaler.transform(df_processed)
        
        return df_scaled
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names after preprocessing."""
        # Simulate preprocessing to get feature names
        df_temp = self.engineer.create_derived_features(df)
        df_temp = self.cleaner.remove_non_predictive_columns(df_temp)
        return [col for col in df_temp.columns if col != 'PSS_score']


def load_and_explore_data(file_path: str) -> pd.DataFrame:
    """Load data and perform initial exploration."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nPSS score distribution:")
    print(df['PSS_score'].describe())
    
    print(f"\nDataset info:")
    print(df.info())
    
    return df


if __name__ == "__main__":
    # Example usage
    df = load_and_explore_data("stress_detection.csv")
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df, include_temporal=True)
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Processed columns: {list(df_processed.columns)}")