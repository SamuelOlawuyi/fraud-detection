"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocess import FraudDataPreprocessor


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    # Add V features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = FraudDataPreprocessor()
    assert preprocessor is not None
    assert preprocessor.scaler is not None


def test_create_time_features(sample_dataframe):
    """Test time feature creation."""
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.create_time_features(sample_dataframe.copy())
    
    assert 'Hour' in df.columns
    assert 'Hour_Sin' in df.columns
    assert 'Hour_Cos' in df.columns
    assert 'Time_Period' in df.columns
    
    # Check hour range
    assert df['Hour'].min() >= 0
    assert df['Hour'].max() < 24


def test_create_amount_features(sample_dataframe):
    """Test amount feature creation."""
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.create_amount_features(sample_dataframe.copy())
    
    assert 'Amount_Log' in df.columns
    assert 'Amount_Category' in df.columns
    assert 'Amount_Zscore' in df.columns
    
    # Check log transformation
    assert (df['Amount_Log'] >= 0).all()


def test_create_statistical_features(sample_dataframe):
    """Test statistical feature creation."""
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.create_statistical_features(sample_dataframe.copy())
    
    assert 'V_Mean' in df.columns
    assert 'V_Std' in df.columns
    assert 'V_Min' in df.columns
    assert 'V_Max' in df.columns
    assert 'V_Range' in df.columns
    assert 'V_Extreme_Count' in df.columns


def test_engineer_features(sample_dataframe):
    """Test complete feature engineering."""
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.engineer_features(sample_dataframe.copy())
    
    # Should have more features than original
    assert len(df.columns) > len(sample_dataframe.columns)


def test_prepare_features(sample_dataframe):
    """Test feature preparation."""
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.engineer_features(sample_dataframe.copy())
    X, y = preprocessor.prepare_features(df, fit=True)
    
    assert X is not None
    assert y is not None
    assert len(X) == len(y)
    assert 'Class' not in X.columns
    
    # Check that features are scaled
    assert X.mean().mean() < 1.0  # Should be roughly centered


def test_no_missing_values_after_preprocessing(sample_dataframe):
    """Test that preprocessing handles missing values."""
    # Add some missing values
    df = sample_dataframe.copy()
    df.loc[0:5, 'V1'] = np.nan
    
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.engineer_features(df)
    X, y = preprocessor.prepare_features(df, fit=True)
    
    assert X.isnull().sum().sum() == 0