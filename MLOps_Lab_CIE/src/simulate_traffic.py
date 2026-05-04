import os
import json
import time
import requests
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_URL = "http://127.0.0.1:9000"

# ── Task 2: test /heartbeat + /score, save step2_s4.json ──────────────────────
print("=== Task 2: Testing API ===")

health_resp = requests.get(f"{API_URL}/heartbeat")
health_data = health_resp.json()
print("Heartbeat:", health_data)

test_input = {
    "money_supply_growth_pct": 6.3,
    "crude_oil_price":         83.5,
    "import_volume_index":     119.8,
    "interest_rate":           5.9,
}
score_resp = requests.post(f"{API_URL}/score", json=test_input)
prediction = score_resp.json()["prediction"]
print(f"Test prediction: {prediction}")

step2 = {
    "health_endpoint":  "/heartbeat",
    "predict_endpoint": "/score",
    "port":             9000,
    "health_response":  health_data,
    "test_input":       test_input,
    "prediction":       prediction,
}
with open(os.path.join(RESULTS_DIR, "step2_s4.json"), "w") as f:
    json.dump(step2, f, indent=2)
print("Saved results/step2_s4.json\n")

# ── Task 3: send 50 requests (30 normal + 20 drifted) ─────────────────────────
print("=== Task 3: Simulating 50 requests (30 normal + 20 drifted) ===")
np.random.seed(42)
FEATURES = ["money_supply_growth_pct", "crude_oil_price", "import_volume_index", "interest_rate"]

normal_requests = []
for _ in range(30):
    normal_requests.append({
        "money_supply_growth_pct": float(round(np.random.uniform(2.0,  15.0),  1)),
        "crude_oil_price":         float(round(np.random.uniform(40.0, 120.0), 1)),
        "import_volume_index":     float(round(np.random.uniform(80.0, 150.0), 1)),
        "interest_rate":           float(round(np.random.uniform(2.0,  10.0),  1)),
    })

new_df           = pd.read_csv(os.path.join(DATA_DIR, "new_data.csv"))
drifted_requests = new_df[FEATURES].to_dict(orient="records")
all_requests     = normal_requests + drifted_requests   # 50 total

print(f"Sending {len(all_requests)} requests...")
for i, payload in enumerate(all_requests):
    try:
        # Clamp to API validation range so requests return 200
        clamped = {
            "money_supply_growth_pct": min(max(payload["money_supply_growth_pct"], 2.0),  15.0),
            "crude_oil_price":         min(max(payload["crude_oil_price"],          40.0), 120.0),
            "import_volume_index":     min(max(payload["import_volume_index"],      80.0), 150.0),
            "interest_rate":           min(max(payload["interest_rate"],            2.0),  10.0),
        }
        requests.post(f"{API_URL}/score", json=clamped, timeout=5)
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")
    time.sleep(0.02)

print("Traffic simulation complete. Run: python src/monitor.py")