"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_predict_endpoint_structure():
    """Test predict endpoint structure (may fail if model not loaded)."""
    # Sample transaction
    transaction = {
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
    
    response = client.post("/predict", json=transaction)
    
    # Either succeeds or returns 503 if model not loaded
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "threshold" in data


def test_predict_invalid_data():
    """Test predict endpoint with invalid data."""
    invalid_transaction = {
        "Time": "invalid",  # Should be float
        "Amount": -100  # Should be >= 0
    }
    
    response = client.post("/predict", json=invalid_transaction)
    assert response.status_code == 422  # Validation error


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model/info")
    # Either succeeds or returns 503 if model not loaded
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "model_path" in data
        assert "threshold" in data


def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    batch = {
        "transactions": [
            {
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
        ]
    }
    
    response = client.post("/predict/batch", json=batch)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "total_transactions" in data
        assert "fraud_count" in data
        assert "fraud_rate" in data