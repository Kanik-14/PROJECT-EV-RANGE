EV RANGE PREDICTOR — WEBSITE INTEGRATION

Files:
- EV_Range_CATBOOST_FINAL_WEBSITE_READY.ipynb : final checked notebook
- ev_range_catboost_final_website_ready.joblib : trained CatBoost model
- ev_range_predictor.py : tiny Python prediction module for a website/backend
- EV_Range_Expanded_Verified_Training_FINAL.csv : training dataset
- EV_Range_Actual_vs_Predicted_FINAL_WEBSITE_READY.csv : untouched hold-out predictions
- EV_Range_FINAL_METRICS.csv : validation metrics

MODEL INPUT JSON (exactly these 11 numeric fields):
{
  "battery_capacity_kWh": 75,
  "torque_nm": 450,
  "top_speed_kmh": 200,
  "acceleration_0_100_s": 4.5,
  "fast_charging_power_kw_dc": 150,
  "towing_capacity_kg": 1000,
  "cargo_volume_l": 450,
  "seats": 5,
  "length_mm": 4700,
  "width_mm": 1850,
  "height_mm": 1600
}

Backend example:
from ev_range_predictor import predict_new_ev
result_km = predict_new_ev(payload)

The module returns a non-negative numeric prediction rounded to 2 decimals.

DO NOT SEND:
- range_km
- efficiency_wh_per_km
- brand
- model
- source_url
- battery_type
- fast_charge_port
- number_of_cells
- drivetrain
- segment
- car_body_type

Validation performed before delivery:
- Notebook executed successfully with all cells passing.
- Model artifact loads successfully.
- Website predictor smoke test passed.
- Missing-input validation passed.
- No efficiency/range leakage in model features.
- Hold-out evaluation was performed before final model training on all rows.
