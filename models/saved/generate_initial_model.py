"""Generate initial model for repository."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Create synthetic data for initial model
np.random.seed(42)
X = np.random.randn(1000, 30)  # 30 features
y = np.random.choice([0, 1], size=1000, p=[0.99, 0.01])  # 1% fraud rate

# Create and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_scaled, y)

# Save models
save_dir = Path(__file__).parent
joblib.dump(model, save_dir / 'fraud_detector.joblib')
joblib.dump(scaler, save_dir / 'scaler.joblib')

print("Initial models saved successfully!")
print("Note: These are placeholder models for deployment testing.")
print("Replace with actual trained models after training on real data.")
