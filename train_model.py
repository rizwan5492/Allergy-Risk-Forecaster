import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
data = {
    "pollen_exposure": np.random.uniform(0, 12, n_samples),
    "aqi": np.random.uniform(0, 500, n_samples),
    "pollution_Low": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "pollution_Medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "pollution_High": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "pet_exposure": np.random.uniform(0, 8, n_samples),
    "dietary_None": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    "dietary_Occasional": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "dietary_Frequent": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "medication_None": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "medication_Occasional": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "medication_Regular": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "ventilation_Poor": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "ventilation_Average": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "ventilation_Good": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    "stress_Low": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "stress_Medium": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    "stress_High": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "temperature": np.random.uniform(-10, 40, n_samples),
    "humidity": np.random.uniform(0, 100, n_samples),
}

# Simulate risk score (weighted sum + noise)
df = pd.DataFrame(data)
df["risk_score"] = (
    0.3 * df["pollen_exposure"] / 12 +
    0.25 * df["aqi"] / 500 +
    0.15 * (df["pollution_High"] + 0.5 * df["pollution_Medium"]) +
    0.1 * df["pet_exposure"] / 8 +
    0.08 * (df["dietary_Frequent"] + 0.5 * df["dietary_Occasional"]) +
    0.05 * (1 - df["medication_Regular"] - 0.5 * df["medication_Occasional"]) +
    0.04 * (df["ventilation_Poor"] + 0.5 * df["ventilation_Average"]) +
    0.03 * (df["stress_High"] + 0.5 * df["stress_Medium"]) +
    np.random.normal(0, 0.1, n_samples)
) * 10
df["risk_score"] = np.clip(df["risk_score"], 0, 10)

# Prepare features and target
X = df.drop("risk_score", axis=1)
y = df["risk_score"]

# Normalize numeric features
scaler = MinMaxScaler()
X[["pollen_exposure", "aqi", "pet_exposure", "temperature", "humidity"]] = scaler.fit_transform(
    X[["pollen_exposure", "aqi", "pet_exposure", "temperature", "humidity"]]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "allergy_risk_model.pkl")
print("Model trained and saved as allergy_risk_model.pkl")
