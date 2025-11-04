"""Evaluate fraud detection models."""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from pathlib import Path

from src.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger()


class FraudModelEvaluator:
    """Evaluate fraud detection models."""
    
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = PROCESSED_DATA_DIR / "fraud_model.pkl"
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
    def load_test_data(self):
        """Load test data."""
        test_path = PROCESSED_DATA_DIR / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_path}. "
                "Please run preprocessing first."
            )
        
        logger.info(f"Loading test data from {test_path}")
        df = pd.read_csv(test_path)
        
        X_test = df.drop('Class', axis=1)
        y_test = df['Class']
        
        logger.info(f"Loaded {len(X_test)} test samples")
        logger.info(f"Test fraud rate: {y_test.mean()*100:.2f}%")
        
        return X_test, y_test
    
    def predict(self, X_test):
        """Make predictions."""
        logger.info("Making predictions...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        return y_pred, y_proba
    
    def calculate_metrics(self, y_test, y_pred, y_proba):
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics."""
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("=" * 50)
        logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"Precision:         {metrics['precision']:.4f}")
        logger.info(f"Recall:            {metrics['recall']:.4f}")
        logger.info(f"F1-Score:          {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"Avg Precision:     {metrics['avg_precision']:.4f}")
        logger.info("=" * 50)
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path: Path = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_test, y_proba, save_path: Path = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_test, y_proba, save_path: Path = None):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")
        
        plt.close()
    
    def evaluate(self, save_plots: bool = True):
        """Complete evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Make predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Print classification report
        logger.info("\nDetailed Classification Report:")
        report = classification_report(y_test, y_pred, 
                                      target_names=['Legitimate', 'Fraud'])
        print(report)
        
        # Save plots
        if save_plots:
            plots_dir = PROCESSED_DATA_DIR / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            self.plot_confusion_matrix(y_test, y_pred, 
                                      plots_dir / "confusion_matrix.png")
            self.plot_roc_curve(y_test, y_proba, 
                               plots_dir / "roc_curve.png")
            self.plot_precision_recall_curve(y_test, y_proba, 
                                            plots_dir / "precision_recall_curve.png")
        
        logger.info("=" * 60)
        logger.info("✓ EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return metrics, y_pred, y_proba


if __name__ == "__main__":
    evaluator = FraudModelEvaluator()
    metrics, y_pred, y_proba = evaluator.evaluate(save_plots=True)