"""FastAPI application for fraud detection."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    TransactionInput, PredictionOutput, BatchTransactionInput,
    BatchPredictionOutput, HealthResponse, ErrorResponse
)
from src.models.predictor import FraudPredictor
from src.config import settings
from src import __version__

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using machine learning",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    try:
        predictor = FraudPredictor()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("API will start but predictions will fail until model is loaded")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        version=__version__
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_transaction(transaction: TransactionInput):
    """Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data with Time, Amount, and V1-V28 features
    
    Returns:
        Prediction result with fraud probability and risk level
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict_single(transaction_dict)
        
        # Add timestamp
        result["timestamp"] = datetime.now()
        
        return PredictionOutput(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(batch: BatchTransactionInput):
    """Predict fraud for multiple transactions.
    
    Args:
        batch: Batch of transactions (max 1000)
    
    Returns:
        Batch prediction results with statistics
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert transactions to list of dicts
        transactions = [t.dict() for t in batch.transactions]
        
        # Make predictions
        results = predictor.predict_batch(transactions)
        
        # Add timestamps
        predictions = []
        fraud_count = 0
        
        for result in results:
            result["timestamp"] = datetime.now()
            predictions.append(PredictionOutput(**result))
            if result.get("is_fraud"):
                fraud_count += 1
        
        total = len(predictions)
        fraud_rate = fraud_count / total if total > 0 else 0
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_transactions=total,
            fraud_count=fraud_count,
            fraud_rate=fraud_rate
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_path": str(settings.model_path),
        "scaler_path": str(settings.scaler_path),
        "threshold": settings.threshold,
        "feature_count": len(predictor.feature_names) if predictor.feature_names else None,
        "version": __version__
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )