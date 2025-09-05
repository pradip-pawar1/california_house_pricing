import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # ================= TRAINING PHASE =================
    df = pd.read_csv("housing.csv")  # Update path if needed

    # Feature Engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["bedrooms_per_room"] = df["bedrooms_per_room"].fillna(0)

    # Optional stratified sampling by income category
    df['income_cat'] = pd.cut(df["median_income"], 
                              bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                              labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, _ in split.split(df, df['income_cat']):
        df = df.loc[train_index].drop("income_cat", axis=1)

    # Separate features & target
    y = df["median_house_value"].copy()
    X = df.drop("median_house_value", axis=1)

    # Identify numeric and categorical attributes
    num_attribs = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Build preprocessing pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    X_prepared = pipeline.fit_transform(X)

    # Train model
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_prepared, y)

    # Save model & pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("✅ XGBRegressor trained and saved successfully!")

else:
    # ================= INFERENCE PHASE =================
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")  # Input file for predictions
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")
