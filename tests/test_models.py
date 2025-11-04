"""Tests for ML models."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import FraudPredictor
from src.config import PROCESSED_DATA_DIR


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return {
        "Time": 12345.0,
        "Amount": 100.50,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053
    }


def test_predictor_initialization():
    """Test that predictor can be initialized."""
    try:
        predictor = FraudPredictor()
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
    except FileNotFoundError:
        pytest.skip("Model files not found. Run training first.")


def test_predict_single(sample_transaction):
    """Test single transaction prediction."""
    try:
        predictor = FraudPredictor()
        result = predictor.predict_single(sample_transaction)
        
        assert "is_fraud" in result
        assert "fraud_probability" in result
        assert "risk_level" in result
        assert "threshold" in result
        
        assert isinstance(result["is_fraud"], bool)
        assert 0 <= result["fraud_probability"] <= 1
        assert result["risk_level"] in ["low", "medium", "high"]
        
    except FileNotFoundError:
        pytest.skip("Model files not found. Run training first.")


def test_predict_batch(sample_transaction):
    """Test batch prediction."""
    try:
        predictor = FraudPredictor()
        transactions = [sample_transaction] * 5
        results = predictor.predict_batch(transactions)
        
        assert len(results) == 5
        for result in results:
            assert "is_fraud" in result
            assert "fraud_probability" in result
            
    except FileNotFoundError:
        pytest.skip("Model files not found. Run training first.")


def test_prediction_consistency(sample_transaction):
    """Test that predictions are consistent for same input."""
    try:
        predictor = FraudPredictor()
        
        result1 = predictor.predict_single(sample_transaction)
        result2 = predictor.predict_single(sample_transaction)
        
        assert result1["fraud_probability"] == result2["fraud_probability"]
        assert result1["is_fraud"] == result2["is_fraud"]
        
    except FileNotFoundError:
        pytest.skip("Model files not found. Run training first.")