# Train & Save California Housing Price Prediction Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Load Dataset
df = pd.read_csv("housing.csv")  # update path if needed

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


# Data Cleaning & Feature Engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["population_per_household"] = df["population"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["bedrooms_per_room"] = df["bedrooms_per_room"].fillna(0)

# One-hot encode categorical
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Features & target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Build Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ XGBRegressor Trained Successfully!")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save Model
joblib.dump(pipeline, MODEL_FILE)
print("Model saved as 'model.pkl'!")
