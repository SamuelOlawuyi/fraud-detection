"""Fraud prediction module."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union

from src.config import PROCESSED_DATA_DIR, settings
from src.utils.logger import get_logger

logger = get_logger()


class FraudPredictor:
    """Make fraud predictions on new transactions."""
    
    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        """Initialize predictor with model and scaler."""
        if model_path is None:
            model_path = Path(settings.model_path)
        if scaler_path is None:
            scaler_path = Path(settings.scaler_path)
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names if available
        feature_names_path = PROCESSED_DATA_DIR / "feature_names.pkl"
        if feature_names_path.exists():
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            self.feature_names = None
            logger.warning("Feature names not found")
        
        self.threshold = settings.threshold
        logger.info(f"Prediction threshold: {self.threshold}")
    
    def preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
        """Preprocess a single transaction."""
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default value 0
                for feature in missing_features:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        return df
    
    def predict_single(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction."""
        # Preprocess
        df = self.preprocess_transaction(transaction)
        
        # Make prediction
        fraud_probability = self.model.predict_proba(df)[0, 1]
        is_fraud = fraud_probability >= self.threshold
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "low"
        elif fraud_probability < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        result = {
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_probability),
            "risk_level": risk_level,
            "threshold": self.threshold
        }
        
        return result
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for multiple transactions."""
        results = []
        
        for transaction in transactions:
            try:
                result = self.predict_single(transaction)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting transaction: {str(e)}")
                results.append({
                    "is_fraud": None,
                    "fraud_probability": None,
                    "risk_level": "error",
                    "error": str(e)
                })
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud for a DataFrame of transactions."""
        # Make predictions
        fraud_probabilities = self.model.predict_proba(df)[:, 1]
        is_fraud = fraud_probabilities >= self.threshold
        
        # Add predictions to DataFrame
        result_df = df.copy()
        result_df['fraud_probability'] = fraud_probabilities
        result_df['is_fraud'] = is_fraud
        result_df['risk_level'] = pd.cut(
            fraud_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return result_df


if __name__ == "__main__":
    # Example usage
    predictor = FraudPredictor()
    
    # Example transaction (you would need actual feature values)
    example_transaction = {
        "Time": 12345,
        "Amount": 100.50,
        # Add other features...
    }
    
    # Make prediction
    result = predictor.predict_single(example_transaction)
    print(f"Prediction: {result}")