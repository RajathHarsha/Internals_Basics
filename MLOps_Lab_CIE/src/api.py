import os
import json
import datetime
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")

# ── Load model ─────────────────────────────────────────────────────────────────
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="EconPulse Inflation Index API")

# ── Input schema with Pydantic validation + correct feature ranges ─────────────
class InflationFeatures(BaseModel):
    money_supply_growth_pct: float = Field(..., ge=2.0,  le=15.0,  description="Money supply growth pct (2–15)")
    crude_oil_price:         float = Field(..., ge=40.0, le=120.0, description="Crude oil price (40–120)")
    import_volume_index:     float = Field(..., ge=80.0, le=150.0, description="Import volume index (80–150)")
    interest_rate:           float = Field(..., ge=2.0,  le=10.0,  description="Interest rate (2–10)")

# ── GET /heartbeat ─────────────────────────────────────────────────────────────
@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "EconPulse inflation_index API"}

# ── POST /score ────────────────────────────────────────────────────────────────
@app.post("/score")
def score(features: InflationFeatures):
    input_dict = features.dict()
    X = np.array([[
        input_dict["money_supply_growth_pct"],
        input_dict["crude_oil_price"],
        input_dict["import_volume_index"],
        input_dict["interest_rate"],
    ]])

    prediction = float(model.predict(X)[0])

    # ── Log every prediction to logs/predictions.jsonl ─────────────────────────
    log_entry = {
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "input":      input_dict,
        "prediction": round(prediction, 4),
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": round(prediction, 4)}

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=False)