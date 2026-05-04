import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MLRUNS_DIR  = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR,  exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "money_supply_growth_pct",
    "crude_oil_price",
    "import_volume_index",
    "interest_rate"
]
TARGET = "inflation_index"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MLflow setup (FIXED) ───────────────────────────────────────────────────────
EXPERIMENT_NAME = "econpulse-inflation-index"

mlflow.set_tracking_uri("./mlruns")   # ✅ FIXED
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Metrics function ───────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    # Safe MAPE
    mask = y_true != 0
    if np.sum(mask) == 0:
        mape = 0.0
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return mae, rmse, r2, mape

results_models = []
run_ids = {}

# ── Train Lasso ────────────────────────────────────────────────────────────────
lasso_params = {"alpha": 0.1, "max_iter": 1000, "random_state": 42}

with mlflow.start_run(run_name="Lasso") as run:
    lasso = Lasso(**lasso_params)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    mae, rmse, r2, mape = compute_metrics(y_test, y_pred_lasso)

    mlflow.log_params(lasso_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    mlflow.set_tag("domain", "economics")
    mlflow.sklearn.log_model(lasso, "lasso_model")

    run_ids["Lasso"] = run.info.run_id

    results_models.append({
        "name": "Lasso",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "run_id": run.info.run_id
    })

    print(f"Lasso → MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}")

# ── Train RandomForest ─────────────────────────────────────────────────────────
rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

with mlflow.start_run(run_name="RandomForest") as run:
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    mae, rmse, r2, mape = compute_metrics(y_test, y_pred_rf)

    mlflow.log_params(rf_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    mlflow.set_tag("domain", "economics")
    mlflow.sklearn.log_model(rf, "rf_model")

    run_ids["RandomForest"] = run.info.run_id

    results_models.append({
        "name": "RandomForest",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "run_id": run.info.run_id
    })

    print(f"RF → MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}")

# ── Select best model ──────────────────────────────────────────────────────────
best = min(results_models, key=lambda m: m["rmse"])
best_model_name = best["name"]

print(f"\nBest model: {best_model_name} (RMSE={best['rmse']})")

# ── Save best model ────────────────────────────────────────────────────────────
best_model = rf if best_model_name == "RandomForest" else lasso

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)

# ── Save JSON output ───────────────────────────────────────────────────────────
step1 = {
    "experiment_name": EXPERIMENT_NAME,
    "models": [
        {
            "name": m["name"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "mape": m["mape"]
        }
        for m in results_models
    ],
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
    json.dump(step1, f, indent=2)

print("\nSaved results/step1_s1.json")
print(json.dumps(step1, indent=2))