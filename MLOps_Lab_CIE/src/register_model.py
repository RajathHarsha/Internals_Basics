import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MLRUNS_DIR  = os.path.join(BASE_DIR, "mlruns")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# ── FIXED MLflow tracking URI (IMPORTANT) ─────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")

EXPERIMENT_NAME = "econpulse-inflation-index"
REGISTERED_NAME = "econpulse-inflation-index-predictor"

# ── Connect MLflow client ─────────────────────────────────────────────────────
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print("ERROR: Experiment not found. Run train.py first.")
    exit(1)

# ── Get best run (lowest RMSE) ────────────────────────────────────────────────
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
)

if not runs:
    print("ERROR: No runs found. Run train.py first.")
    exit(1)

best_run    = runs[0]
best_run_id = best_run.info.run_id
best_rmse   = best_run.data.metrics["rmse"]
best_name   = best_run.data.tags.get("mlflow.runName", "unknown")

print(f"Best run: {best_name}  run_id={best_run_id}  RMSE={best_rmse}")

# ── Decide model path ─────────────────────────────────────────────────────────
if "Lasso" in best_name:
    artifact_path = "lasso_model"
else:
    artifact_path = "rf_model"

model_uri = f"runs:/{best_run_id}/{artifact_path}"
print(f"Registering model from: {model_uri}")

# ── Register model ────────────────────────────────────────────────────────────
result  = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)
version = int(result.version)

print(f"Registered: {REGISTERED_NAME} version={version}")

# ── Save output JSON ──────────────────────────────────────────────────────────
step4 = {
    "registered_model_name": REGISTERED_NAME,
    "version": version,
    "run_id": best_run_id,
    "source_metric": "rmse",
    "source_metric_value": round(best_rmse, 4),
}

with open(os.path.join(RESULTS_DIR, "step4_s6.json"), "w") as f:
    json.dump(step4, f, indent=2)

print("\nSaved results/step4_s6.json")
print(json.dumps(step4, indent=2))