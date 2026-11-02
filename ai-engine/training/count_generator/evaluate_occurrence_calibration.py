
import pandas as pd
import numpy as np
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

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

output_dir = ROOT / "evaluation/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

report = {}

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    y_train = (train[target] > 0).astype(int)
    y_val = (validation[target] > 0).astype(int)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], y_train)

    probabilities = model.predict_proba(
        validation[features]
    )[:, 1]

    brier = brier_score_loss(y_val, probabilities)
    logloss = log_loss(y_val, probabilities)

    # Calibration bins
    fraction_positive, mean_probability = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    print("Actual occurrence rate:", round(y_val.mean(), 4))
    print("Mean predicted probability:", round(probabilities.mean(), 4))
    print("Brier score:", round(brier, 5))
    print("Log loss:", round(logloss, 5))

    print("\nCalibration bins:")
    for predicted, actual in zip(mean_probability, fraction_positive):
        print(
            f"Predicted={predicted:.4f} | "
            f"Actual={actual:.4f}"
        )

    report[target] = {
        "actual_rate": float(y_val.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "calibration_bins": [
            {
                "predicted_probability": float(p),
                "actual_frequency": float(a)
            }
            for p, a in zip(mean_probability, fraction_positive)
        ]
    }

with open(output_dir / "occurrence_calibration.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nCalibration report saved successfully.")
