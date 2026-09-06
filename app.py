from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained CatBoost model you saved earlier
model_path = os.path.join(os.path.dirname(__file__), 'ev_range_catboost_final_ready.joblib')
artifact = joblib.load(model_path)
model = artifact['model']
NUMERIC_INPUTS = artifact['numeric_inputs']
features_cols = artifact['feature_columns']

# Run the exact math formulas used during training
def build_features(frame):
    x = frame[NUMERIC_INPUTS].copy()
    for c in NUMERIC_INPUTS: 
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["vehicle_volume_m3"] = x["length_mm"] * x["width_mm"] * x["height_mm"] / 1e9
    x["vehicle_footprint_m2"] = x["length_mm"] * x["width_mm"] / 1e6
    x["battery_per_volume"] = x["battery_capacity_kWh"] / x["vehicle_volume_m3"].replace(0, np.nan)
    x["battery_per_seat"] = x["battery_capacity_kWh"] / x["seats"].replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)

@app.post('/api/predict')
async def predict(data: dict):
    raw = pd.DataFrame([data])
    features = build_features(raw).reindex(columns=features_cols)
    pred = max(0.0, round(float(model.predict(features)[0]), 2))
    return {'range_km': pred}