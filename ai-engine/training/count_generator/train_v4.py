
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

features = ["grid_lat", "grid_lon"]

train["recent"] = train["past_1m_count"]
train["previous"] = train["past_3m_count"] - train["past_1m_count"]
train["older"] = train["past_6m_count"] - train["past_3m_count"]

base_train = train[train["year"] <= 2016].copy()

calibration = train[
    (train["year"] >= 2017) &
    (train["year"] <= 2019)
].copy()

output_dir = ROOT / "models/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

metadata = {
    "version": "V4",
    "features": features,
    "base_training_period": "2000-2016",
    "calibration_period": "2017-2019",
    "targets": {}
}

for target in ["recent", "previous", "older"]:

    print(f"\nTraining {target} model...")

    y_base = (base_train[target] > 0).astype(int)
    y_cal = (calibration[target] > 0).astype(int)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(base_train[features], y_base)

    calibration_scores = model.predict_proba(
        calibration[features]
    )[:, 1]

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000
    )

    calibrator.fit(
        calibration_scores.reshape(-1, 1),
        y_cal
    )

    model_path = output_dir / f"{target}_occurrence_model.joblib"
    calibration_path = output_dir / f"{target}_calibrator.joblib"

    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibration_path)

    metadata["targets"][target] = {
        "model": model_path.name,
        "calibrator": calibration_path.name,
        "training_positive_rate": float(y_base.mean()),
        "calibration_positive_rate": float(y_cal.mean())
    }

    print("Saved:", model_path.name)
    print("Saved:", calibration_path.name)

with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nV4 occurrence models and calibrators saved successfully.")

