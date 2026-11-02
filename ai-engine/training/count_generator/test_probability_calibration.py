
import pandas as pd
import numpy as np
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
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

features = ["grid_lat", "grid_lon"]

targets = {
    "recent": "past_1m_count",
    "previous": None,
    "older": None
}

train["recent"] = train["past_1m_count"]
train["previous"] = train["past_3m_count"] - train["past_1m_count"]
train["older"] = train["past_6m_count"] - train["past_3m_count"]

validation["recent"] = validation["past_1m_count"]
validation["previous"] = validation["past_3m_count"] - validation["past_1m_count"]
validation["older"] = validation["past_6m_count"] - validation["past_3m_count"]

report = {}

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    y_train = (train[target] > 0).astype(int)
    y_val = (validation[target] > 0).astype(int)

    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    base_model.fit(train[features], y_train)

    raw_prob = base_model.predict_proba(
        validation[features]
    )[:, 1]

    # Sigmoid calibration using training-only cross-validation.
    calibrated_model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        method="sigmoid",
        cv=3,
        n_jobs=-1
    )

    calibrated_model.fit(train[features], y_train)

    calibrated_prob = calibrated_model.predict_proba(
        validation[features]
    )[:, 1]

    raw_metrics = {
        "mean_probability": float(raw_prob.mean()),
        "brier": float(brier_score_loss(y_val, raw_prob)),
        "log_loss": float(log_loss(y_val, raw_prob)),
        "roc_auc": float(roc_auc_score(y_val, raw_prob))
    }

    calibrated_metrics = {
        "mean_probability": float(calibrated_prob.mean()),
        "brier": float(brier_score_loss(y_val, calibrated_prob)),
        "log_loss": float(log_loss(y_val, calibrated_prob)),
        "roc_auc": float(roc_auc_score(y_val, calibrated_prob))
    }

    print("Actual rate:", round(y_val.mean(), 4))
    print("Raw:", raw_metrics)
    print("Calibrated:", calibrated_metrics)

    report[target] = {
        "actual_rate": float(y_val.mean()),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics
    }

output = ROOT / "evaluation/count_generator/v4"
output.mkdir(parents=True, exist_ok=True)

with open(output / "sigmoid_calibration_comparison.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nComparison saved successfully.")
