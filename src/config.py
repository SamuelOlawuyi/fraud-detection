"""Configuration management for the fraud detection system."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, OSError):
        # Directory already exists or is a file, skip
        pass

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "4"))
    
    # Model Configuration
    model_path: str = os.getenv("MODEL_PATH", str(PROCESSED_DATA_DIR / "fraud_model.pkl"))
    scaler_path: str = os.getenv("SCALER_PATH", str(PROCESSED_DATA_DIR / "scaler.pkl"))
    threshold: float = float(os.getenv("THRESHOLD", "0.5"))
    
    # Kaggle Configuration
    kaggle_username: Optional[str] = os.getenv("KAGGLE_USERNAME")
    kaggle_key: Optional[str] = os.getenv("KAGGLE_KEY")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Dataset URLs
    creditcard_dataset: str = "mlg-ulb/creditcardfraud"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Model parameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": 1
    }
}

# Feature engineering parameters
FEATURE_PARAMS = {
    "time_features": True,
    "amount_features": True,
    "interaction_features": True,
    "statistical_features": True
}

# SMOTE parameters for handling imbalanced data
SMOTE_PARAMS = {
    "sampling_strategy": 0.5,
    "random_state": 42,
    "k_neighbors": 5
}