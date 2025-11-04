"""Data preprocessing and feature engineering."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_PARAMS
from src.utils.logger import get_logger

logger = get_logger()


class FraudDataPreprocessor:
    """Preprocess fraud detection data."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def load_data(self, filepath: Path = None) -> pd.DataFrame:
        """Load raw data from CSV."""
        if filepath is None:
            # Find the first CSV file in raw data directory
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
            filepath = csv_files[0]
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} transactions with {len(df.columns)} features")
        logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'Time' not in df.columns:
            return df
        
        logger.info("Creating time-based features")
        
        # Convert seconds to hours
        df['Hour'] = (df['Time'] / 3600) % 24
        
        # Create time periods
        df['Time_Period'] = pd.cut(
            df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Cyclical encoding for hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        if 'Amount' not in df.columns:
            return df
        
        logger.info("Creating amount-based features")
        
        # Log transformation
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # Amount categories
        df['Amount_Category'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, np.inf],
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
        
        # Z-score for amount
        df['Amount_Zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from V columns."""
        logger.info("Creating statistical features")
        
        # Get V columns
        v_columns = [col for col in df.columns if col.startswith('V')]
        
        if v_columns:
            # Statistical aggregations
            df['V_Mean'] = df[v_columns].mean(axis=1)
            df['V_Std'] = df[v_columns].std(axis=1)
            df['V_Min'] = df[v_columns].min(axis=1)
            df['V_Max'] = df[v_columns].max(axis=1)
            df['V_Range'] = df['V_Max'] - df['V_Min']
            
            # Count of extreme values
            df['V_Extreme_Count'] = (df[v_columns].abs() > 3).sum(axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features")
        
        if 'Amount' in df.columns and 'Hour' in df.columns:
            df['Amount_Hour_Interaction'] = df['Amount_Log'] * df['Hour']
        
        # Interaction with key V features
        v_columns = [col for col in df.columns if col.startswith('V')]
        if v_columns and 'Amount' in df.columns:
            # Interaction with first few V features
            for v_col in v_columns[:5]:
                df[f'{v_col}_Amount'] = df[v_col] * df['Amount_Log']
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Starting feature engineering")
        
        if FEATURE_PARAMS['time_features']:
            df = self.create_time_features(df)
        
        if FEATURE_PARAMS['amount_features']:
            df = self.create_amount_features(df)
        
        if FEATURE_PARAMS['statistical_features']:
            df = self.create_statistical_features(df)
        
        if FEATURE_PARAMS['interaction_features']:
            df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True):
        """Prepare features for modeling."""
        logger.info("Preparing features for modeling")
        
        # Separate target
        if 'Class' in df.columns:
            y = df['Class']
            X = df.drop('Class', axis=1)
        else:
            y = None
            X = df.copy()
        
        # Drop non-numeric columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Dropping categorical columns: {list(categorical_cols)}")
            X = X.drop(categorical_cols, axis=1)
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            logger.warning(f"Found {X.isnull().sum().sum()} missing values. Filling with median.")
            X = X.fillna(X.median())
        
        # Scale features
        if fit:
            logger.info("Fitting scaler on training data")
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            logger.info("Transforming data with fitted scaler")
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def process_and_save(self, test_size: float = 0.2, random_state: int = 42):
        """Complete preprocessing pipeline and save processed data."""
        logger.info("=" * 50)
        logger.info("Starting complete preprocessing pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_data()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df, fit=True)
        
        # Split data
        logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train fraud rate: {y_train.mean()*100:.2f}%")
        logger.info(f"Test fraud rate: {y_test.mean()*100:.2f}%")
        
        # Save processed data
        logger.info("Saving processed data...")
        
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_path = PROCESSED_DATA_DIR / "train.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved test data to {test_path}")
        
        # Save scaler
        scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_names_path = PROCESSED_DATA_DIR / "feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        logger.info(f"Saved feature names to {feature_names_path}")
        
        logger.info("=" * 50)
        logger.info("✓ Preprocessing completed successfully")
        logger.info("=" * 50)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.process_and_save()