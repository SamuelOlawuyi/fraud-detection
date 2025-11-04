"""Pydantic schemas for API validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    Time: float = Field(..., description="Time in seconds from first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: Optional[float] = 0.0
    V2: Optional[float] = 0.0
    V3: Optional[float] = 0.0
    V4: Optional[float] = 0.0
    V5: Optional[float] = 0.0
    V6: Optional[float] = 0.0
    V7: Optional[float] = 0.0
    V8: Optional[float] = 0.0
    V9: Optional[float] = 0.0
    V10: Optional[float] = 0.0
    V11: Optional[float] = 0.0
    V12: Optional[float] = 0.0
    V13: Optional[float] = 0.0
    V14: Optional[float] = 0.0
    V15: Optional[float] = 0.0
    V16: Optional[float] = 0.0
    V17: Optional[float] = 0.0
    V18: Optional[float] = 0.0
    V19: Optional[float] = 0.0
    V20: Optional[float] = 0.0
    V21: Optional[float] = 0.0
    V22: Optional[float] = 0.0
    V23: Optional[float] = 0.0
    V24: Optional[float] = 0.0
    V25: Optional[float] = 0.0
    V26: Optional[float] = 0.0
    V27: Optional[float] = 0.0
    V28: Optional[float] = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 12345.0,
                "Amount": 100.50,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                # ... other V features
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for fraud prediction."""
    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, or high")
    threshold: float = Field(..., description="Threshold used for classification")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.87,
                "risk_level": "high",
                "threshold": 0.5,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class BatchTransactionInput(BaseModel):
    """Input schema for batch predictions."""
    transactions: List[TransactionInput] = Field(..., description="List of transactions")
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 transactions")
        return v


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)