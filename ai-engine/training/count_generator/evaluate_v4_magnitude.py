
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

for df in [train, validation]:
    df["recent"] = df["past_1m_count"]
    df["previous"] = df["past_3m_count"] - df["past_1m_count"]
    df["older"] = df["past_6m_count"] - df["past_3m_count"]

features = ["grid_lat", "grid_lon"]
targets = ["recent", "previous", "older"]

model_dir = ROOT / "models/count_generator/v4"
output_dir = ROOT / "evaluation/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

report = {}

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    model = joblib.load(
        model_dir / f"{target}_magnitude_model.joblib"
    )

    positive_validation = validation[
        validation[target] > 0
    ].copy()

    X = positive_validation[features]
    y = positive_validation[target]

    predictions = np.expm1(model.predict(X))
    predictions = np.maximum(predictions, 0)

    baseline = train.loc[
        train[target] > 0, target
    ].mean()

    baseline_predictions = np.full(len(y), baseline)

    mae = mean_absolute_error(y, predictions)
    baseline_mae = mean_absolute_error(y, baseline_predictions)

    r2 = r2_score(y, predictions)

    print("Validation positive samples:", len(y))
    print("Actual positive mean:", round(y.mean(), 4))
    print("Predicted positive mean:", round(predictions.mean(), 4))
    print("Model MAE:", round(mae, 4))
    print("Baseline MAE:", round(baseline_mae, 4))
    print("R2:", round(r2, 4))

    report[target] = {
        "validation_positive_samples": int(len(y)),
        "actual_mean": float(y.mean()),
        "predicted_mean": float(predictions.mean()),
        "model_mae": float(mae),
        "baseline_mae": float(baseline_mae),
        "r2": float(r2)
    }

with open(output_dir / "magnitude_evaluation.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nMagnitude evaluation saved successfully.")
