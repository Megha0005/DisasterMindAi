
import pandas as pd
import numpy as np
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

train["recent"] = train["past_1m_count"]
train["previous"] = train["past_3m_count"] - train["past_1m_count"]
train["older"] = train["past_6m_count"] - train["past_3m_count"]

validation["recent"] = validation["past_1m_count"]
validation["previous"] = validation["past_3m_count"] - validation["past_1m_count"]
validation["older"] = validation["past_6m_count"] - validation["past_3m_count"]

features = ["grid_lat", "grid_lon"]

# Chronological calibration split
base_train = train[train["year"] <= 2016]
calibration = train[
    (train["year"] >= 2017) &
    (train["year"] <= 2019)
]

print("Base training rows:", len(base_train))
print("Calibration rows:", len(calibration))
print("Validation rows:", len(validation))

report = {}

for target in ["recent", "previous", "older"]:

    print(f"\n========== {target.upper()} ==========")

    y_base = (base_train[target] > 0).astype(int)
    y_cal = (calibration[target] > 0).astype(int)
    y_val = (validation[target] > 0).astype(int)

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

    validation_scores = model.predict_proba(
        validation[features]
    )[:, 1]

    # Fit global sigmoid mapping using chronological calibration data.
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000
    )

    calibrator.fit(
        calibration_scores.reshape(-1, 1),
        y_cal
    )

    calibrated_scores = calibrator.predict_proba(
        validation_scores.reshape(-1, 1)
    )[:, 1]

    raw = {
        "mean_probability": float(validation_scores.mean()),
        "brier": float(brier_score_loss(y_val, validation_scores)),
        "log_loss": float(log_loss(y_val, validation_scores)),
        "roc_auc": float(roc_auc_score(y_val, validation_scores))
    }

    calibrated = {
        "mean_probability": float(calibrated_scores.mean()),
        "brier": float(brier_score_loss(y_val, calibrated_scores)),
        "log_loss": float(log_loss(y_val, calibrated_scores)),
        "roc_auc": float(roc_auc_score(y_val, calibrated_scores))
    }

    print("Actual rate:", round(y_val.mean(), 4))
    print("Raw:", raw)
    print("Temporal calibrated:", calibrated)

    report[target] = {
        "actual_rate": float(y_val.mean()),
        "raw": raw,
        "temporal_calibrated": calibrated
    }

output = ROOT / "evaluation/count_generator/v4"
output.mkdir(parents=True, exist_ok=True)

with open(output / "temporal_calibration_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nTemporal calibration report saved.")

