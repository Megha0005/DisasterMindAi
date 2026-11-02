
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from models.neural_network.mlp_model import DisasterMLP


TEST_FILE = Path("data/processed/splits/test.csv")
MODEL_FILE = Path("models/neural_network/mlp_best.pth")
SCALER_FILE = Path("models/neural_network/mlp_scaler.joblib")
CONFIG_FILE = Path("evaluation/neural_network/mlp_config.json")

OUTPUT_DIR = Path("evaluation/final")
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
    print("Loading test dataset...")

    df = pd.read_csv(TEST_FILE)
    X = prepare_features(df)

    y_true = df["target_next_month"].to_numpy(dtype=int)

    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(X).astype(np.float32)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = config["classification_threshold"]

    model = DisasterMLP(input_features=8).to(device)

    state_dict = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Test rows:", f"{len(X_scaled):,}")
    print("Classification threshold:", threshold)

    print("\nGenerating test predictions...")

    all_probabilities = []

    with torch.no_grad():
        for start in range(0, len(X_scaled), 4096):
            batch = torch.from_numpy(
                X_scaled[start:start + 4096]
            ).to(device)

            logits = model(batch).squeeze(1)
            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    probabilities = np.asarray(all_probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "model_name": "DisasterMLP",
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    print("\n========== MLP FINAL TEST RESULTS ==========")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    pd.DataFrame({
        "actual": y_true,
        "probability": probabilities,
        "prediction": predictions
    }).to_csv(
        OUTPUT_DIR / "mlp_test_predictions.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "mlp_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved test predictions.")
    print("Saved test metrics.")
    print("\nMLP test evaluation completed!")


if __name__ == "__main__":
    main()
