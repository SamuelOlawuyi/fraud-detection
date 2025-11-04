"""Train fraud detection models."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import PROCESSED_DATA_DIR, MODEL_PARAMS, SMOTE_PARAMS
from src.utils.logger import get_logger

logger = get_logger()


class FraudModelTrainer:
    """Train and save fraud detection models."""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        
    def load_data(self):
        """Load preprocessed training data."""
        train_path = PROCESSED_DATA_DIR / "train.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Please run preprocessing first."
            )
        
        logger.info(f"Loading training data from {train_path}")
        df = pd.read_csv(train_path)
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        logger.info(f"Loaded {len(X)} training samples")
        logger.info(f"Features: {len(X.columns)}")
        logger.info(f"Fraud rate: {y.mean()*100:.2f}%")
        
        return X, y
    
    def build_random_forest(self):
        """Build Random Forest model with SMOTE."""
        logger.info("Building Random Forest model")
        
        # Create pipeline with SMOTE
        model = ImbPipeline([
            ('smote', SMOTE(**SMOTE_PARAMS)),
            ('classifier', RandomForestClassifier(**MODEL_PARAMS['random_forest']))
        ])
        
        self.model_name = "random_forest"
        return model
    
    def build_gradient_boosting(self):
        """Build Gradient Boosting model with SMOTE."""
        logger.info("Building Gradient Boosting model")
        
        model = ImbPipeline([
            ('smote', SMOTE(**SMOTE_PARAMS)),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        self.model_name = "gradient_boosting"
        return model
    
    def build_logistic_regression(self):
        """Build Logistic Regression model with SMOTE."""
        logger.info("Building Logistic Regression model")
        
        model = ImbPipeline([
            ('smote', SMOTE(**SMOTE_PARAMS)),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        self.model_name = "logistic_regression"
        return model
    
    def train_model(self, model, X_train, y_train):
        """Train the model."""
        logger.info("=" * 50)
        logger.info(f"Training {self.model_name} model")
        logger.info("=" * 50)
        
        # Train
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        logger.info("✓ Model training completed")
        
        # Training predictions
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        
        # Training metrics
        train_auc = roc_auc_score(y_train, y_proba_train)
        logger.info(f"Training AUC-ROC: {train_auc:.4f}")
        
        return model
    
    def save_model(self, model, model_name: str = None):
        """Save trained model."""
        if model_name is None:
            model_name = self.model_name
        
        model_path = PROCESSED_DATA_DIR / f"{model_name}_model.pkl"
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logger.info("✓ Model saved successfully")
        
        return model_path
    
    def train_and_save(self, model_type: str = "random_forest"):
        """Complete training pipeline."""
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION MODEL TRAINING")
        logger.info("=" * 60)
        
        # Load data
        X_train, y_train = self.load_data()
        
        # Build model
        if model_type == "random_forest":
            model = self.build_random_forest()
        elif model_type == "gradient_boosting":
            model = self.build_gradient_boosting()
        elif model_type == "logistic_regression":
            model = self.build_logistic_regression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model = self.train_model(model, X_train, y_train)
        
        # Save model
        model_path = self.save_model(model)
        
        # Also save as default model
        default_path = PROCESSED_DATA_DIR / "fraud_model.pkl"
        joblib.dump(model, default_path)
        logger.info(f"Saved as default model: {default_path}")
        
        logger.info("=" * 60)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return model


def train_all_models():
    """Train all model types."""
    trainer = FraudModelTrainer()
    models = {}
    
    for model_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
        try:
            logger.info(f"\n\nTraining {model_type}...")
            model = trainer.train_and_save(model_type)
            models[model_type] = model
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
    
    return models


if __name__ == "__main__":
    # Train Random Forest (default)
    trainer = FraudModelTrainer()
    model = trainer.train_and_save(model_type="random_forest")
    
    # Uncomment to train all models
    # train_all_models()