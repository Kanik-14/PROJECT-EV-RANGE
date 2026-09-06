import os
import joblib
import numpy as np
import pandas as pd
MODEL_PATH=os.path.join(os.path.dirname(__file__),"ev_range_catboost_final_website_ready.joblib")
_ARTIFACT=None
def _load():
 global _ARTIFACT
 if _ARTIFACT is None: _ARTIFACT=joblib.load(MODEL_PATH)
 return _ARTIFACT
def _features(raw,inputs):
 x=raw[inputs].copy()
 for c in inputs: x[c]=pd.to_numeric(x[c],errors="coerce")
 x["vehicle_volume_m3"]=x["length_mm"]*x["width_mm"]*x["height_mm"]/1e9
 x["vehicle_footprint_m2"]=x["length_mm"]*x["width_mm"]/1e6
 x["battery_per_volume"]=x["battery_capacity_kWh"]/x["vehicle_volume_m3"].replace(0,np.nan)
 x["battery_per_seat"]=x["battery_capacity_kWh"]/x["seats"].replace(0,np.nan)
 return x.replace([np.inf,-np.inf],np.nan)
def predict_new_ev(payload=None,**kwargs):
 values=dict(payload or {}); values.update(kwargs); a=_load(); req=a["numeric_inputs"]
 missing=[c for c in req if c not in values]
 if missing: raise ValueError("Missing inputs: "+", ".join(missing))
 raw=pd.DataFrame([{c:values[c] for c in req}])
 for c in req: raw[c]=pd.to_numeric(raw[c],errors="coerce")
 bad=[c for c in req if pd.isna(raw.loc[0,c])]
 if bad: raise ValueError("Invalid numeric inputs: "+", ".join(bad))
 f=_features(raw,req).reindex(columns=a["feature_columns"]); return max(0.0,round(float(a["model"].predict(f)[0]),2))
