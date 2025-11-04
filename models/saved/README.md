# Saved Models

This directory contains the trained models and preprocessing artifacts for the fraud detection system:

- `fraud_detector.joblib`: Main Random Forest model for fraud detection
- `scaler.joblib`: Standard scaler for feature normalization
- `feature_encoder.joblib`: Categorical feature encoder

These files are used by the API for making real-time predictions. They are automatically loaded when the API starts.

## Model Details

- Algorithm: Random Forest Classifier
- SMOTE enabled for handling class imbalance
- Trained on preprocessed credit card transaction data
- Feature engineering applied as per preprocessing pipeline
