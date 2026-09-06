from http.server import BaseHTTPRequestHandler
import json
from catboost_model import apply_catboost_model

NUMERIC_INPUTS = [
    "battery_capacity_kWh",
    "torque_nm",
    "top_speed_kmh",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm",
]


def build_feature_vector(values):
    v = {k: float(values[k]) for k in NUMERIC_INPUTS}

    vehicle_volume_m3 = v["length_mm"] * v["width_mm"] * v["height_mm"] / 1e9
    vehicle_footprint_m2 = v["length_mm"] * v["width_mm"] / 1e6
    battery_per_volume = v["battery_capacity_kWh"] / vehicle_volume_m3 if vehicle_volume_m3 else float("nan")
    battery_per_seat = v["battery_capacity_kWh"] / v["seats"] if v["seats"] else float("nan")

    # Order verified against the trained model's feature_columns
    return [
        v["battery_capacity_kWh"],
        v["torque_nm"],
        v["top_speed_kmh"],
        v["acceleration_0_100_s"],
        v["fast_charging_power_kw_dc"],
        v["towing_capacity_kg"],
        v["cargo_volume_l"],
        v["seats"],
        v["length_mm"],
        v["width_mm"],
        v["height_mm"],
        vehicle_volume_m3,
        vehicle_footprint_m2,
        battery_per_volume,
  battery_per_seat,
    ]


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "EV range predictor API is running"}).encode())

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body or b"{}")

            missing = [c for c in NUMERIC_INPUTS if c not in data]
            if missing:
                self._respond(400, {"error": f"Missing inputs: {', '.join(missing)}"})
                return

            features = build_feature_vector(data)

            if any(f != f for f in features):  # NaN check, no numpy needed
                self._respond(400, {"error": "Invalid numeric input (e.g. seats or dimensions is zero)"})
                return

            raw_prediction = apply_catboost_model(features)
            prediction = max(0.0, round(float(raw_prediction), 2))
            self._respond(200, {"range_km": prediction})

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self._respond(400, {"error": f"Invalid request: {str(e)}"})
        except Exception as e:
            self._respond(500, {"error": f"Server error: {str(e)}"})

    def _respond(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

