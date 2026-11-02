
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


DATA_DIR = Path("data/processed/splits")
MODEL_FILE = Path("models/baseline/logistic_regression.joblib")
OUTPUT_DIR = Path("evaluation/baseline")


def main():
    print("Loading validation dataset and trained model...")

    df = pd.read_csv(DATA_DIR / "validation.csv")
    model = joblib.load(MODEL_FILE)

    features = [
        "grid_lat",
        "grid_lon",
        "year",
        "month_sin",
        "month_cos",
        "past_1m_count",
        "past_3m_count",
        "past_6m_count",
    ]

    X = df[
        [
            "grid_lat",
            "grid_lon",
            "year",
            "month",
            "past_1m_count",
            "past_3m_count",
            "past_6m_count",
        ]
    ].copy()

    X["month_sin"] = np.sin(
        2 * np.pi * X["month"] / 12
    )

    X["month_cos"] = np.cos(
        2 * np.pi * X["month"] / 12
    )

    for column in [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count",
    ]:
        X[column] = np.log1p(X[column])

    X = X[features]
    y = df["target_next_month"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]

    print("\nROC-AUC:", round(roc_auc_score(y, probabilities), 4))
    print(
        "PR-AUC:",
        round(average_precision_score(y, probabilities), 4)
    )

    precision, recall, thresholds = precision_recall_curve(
        y,
        probabilities
    )

    # Calculate F1 for every available threshold.
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

    print("\n========== THRESHOLD COMPARISON ==========")
    print("Threshold       Precision       Recall       F1-score")

    selected_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    for threshold in selected_thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = ((predictions == 1) & (y == 1)).sum()
        fp = ((predictions == 1) & (y == 0)).sum()
        fn = ((predictions == 0) & (y == 1)).sum()

        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0

        print(
            f"{threshold:<16.2f}"
            f"{p:<16.4f}"
            f"{r:<13.4f}"
            f"{f1:.4f}"
        )

    print("\n========== BEST VALIDATION THRESHOLD ==========")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"Precision: {precision[best_index]:.4f}")
    print(f"Recall: {recall[best_index]:.4f}")
    print(f"F1-score: {f1_scores[best_index]:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1_score": f1_scores,
    }).to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False
    )

    # Save the selected validation threshold separately.
    pd.DataFrame([{
        "selected_threshold": best_threshold,
        "validation_precision": precision[best_index],
        "validation_recall": recall[best_index],
        "validation_f1": f1_scores[best_index],
    }]).to_csv(
        OUTPUT_DIR / "selected_threshold.csv",
        index=False
    )

    print("\nThreshold analysis saved successfully!")


if __name__ == "__main__":
    main()
