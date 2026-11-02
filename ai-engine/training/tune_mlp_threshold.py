
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from models.neural_network.mlp_model import DisasterMLP


VAL_FILE = Path("data/processed/splits/validation.csv")
MODEL_FILE = Path("models/neural_network/mlp_best.pth")
SCALER_FILE = Path("models/neural_network/mlp_scaler.joblib")

OUTPUT_DIR = Path("evaluation/neural_network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_features(df):
    month_period = pd.PeriodIndex(df["month_period"], freq="M")

    X = pd.DataFrame(index=df.index)

    X["grid_lat"] = df["grid_lat"]
    X["grid_lon"] = df["grid_lon"]
    X["year"] = month_period.year
    X["month_sin"] = np.sin(2 * np.pi * month_period.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month_period.month / 12)

    X["past_1m_count"] = np.log1p(df["past_1m_count"])
    X["past_3m_count"] = np.log1p(df["past_3m_count"])
    X["past_6m_count"] = np.log1p(df["past_6m_count"])

    return X.astype("float32")


def main():
    print("Loading validation dataset...")

    df = pd.read_csv(VAL_FILE)

    X = prepare_features(df)
    y_true = df["target_next_month"].to_numpy(dtype=int)

    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(X).astype(np.float32)

    model = DisasterMLP(input_features=8).to(device)

    state_dict = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Device:", device)
    print("Validation rows:", f"{len(X_scaled):,}")

    print("\nGenerating predictions...")

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(X_scaled), 4096):
            batch = torch.from_numpy(
                X_scaled[start:start + 4096]
            ).to(device)

            logits = model(batch).squeeze(1)
            batch_probabilities = torch.sigmoid(logits)

            probabilities.extend(
                batch_probabilities.cpu().numpy()
            )

    probabilities = np.asarray(probabilities)

    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    # Find threshold maximizing validation F1.
    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(y_true, probabilities)
    )

    f1_values = (
        2 * precision_curve[:-1] * recall_curve[:-1]
        / (
            precision_curve[:-1]
            + recall_curve[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    print("\n========== MLP VALIDATION RESULTS ==========")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nThreshold comparison:")

    threshold_results = []

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, best_threshold]:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        threshold_results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        })

        print(
            f"Threshold: {threshold:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    best_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        best_predictions,
        labels=[0, 1]
    ).ravel()

    print("\nBest threshold:", f"{best_threshold:.6f}")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    pd.DataFrame(threshold_results).to_csv(
        OUTPUT_DIR / "mlp_threshold_analysis.csv",
        index=False
    )

    pd.DataFrame({
        "actual": y_true,
        "probability": probabilities,
        "prediction": best_predictions
    }).to_csv(
        OUTPUT_DIR / "mlp_validation_predictions.csv",
        index=False
    )

    config = {
        "model_name": "DisasterMLP",
        "version": "1.0.0",
        "target": "target_next_month",
        "prediction_horizon": "1 month",
        "grid_resolution_degrees": 2,
        "historical_windows_months": [1, 3, 6],
        "classification_threshold": best_threshold,
        "threshold_selection_metric": "validation_f1",
        "validation_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "precision": float(
                precision_score(y_true, best_predictions)
            ),
            "recall": float(
                recall_score(y_true, best_predictions)
            ),
            "f1_score": float(
                f1_score(y_true, best_predictions)
            )
        }
    }

    with open(
        OUTPUT_DIR / "mlp_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4)

    print("\nSaved threshold analysis.")
    print("Saved validation predictions.")
    print("Saved MLP configuration.")
    print("\nThreshold tuning completed!")


if __name__ == "__main__":
    main()
