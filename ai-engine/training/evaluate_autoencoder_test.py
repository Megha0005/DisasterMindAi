
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from models.autoencoder.autoencoder_model import DisasterAutoencoder


TEST_FILE = Path("data/processed/splits/test.csv")
MODEL_FILE = Path("models/autoencoder/autoencoder_best.pth")
SCALER_FILE = Path("models/autoencoder/autoencoder_scaler.joblib")
CONFIG_FILE = Path("evaluation/autoencoder/anomaly_metrics.json")

OUTPUT_DIR = Path("evaluation/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4096
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


def calculate_errors(model, features):
    model.eval()
    errors = []

    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.from_numpy(
                features[start:start + BATCH_SIZE]
            ).to(device)

            reconstructed, _ = model(batch)

            batch_errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors)


def main():
    print("Loading test dataset...")

    df = pd.read_csv(TEST_FILE)
    X = prepare_features(df)
    y = df["target_next_month"].to_numpy(dtype=int)

    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(X).astype(np.float32)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = config["threshold"]

    model = DisasterAutoencoder(
        input_features=8,
        latent_dim=3
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device,
            weights_only=True
        )
    )

    print("Device:", device)
    print("Test rows:", f"{len(X_scaled):,}")
    print("Fixed anomaly threshold:", f"{threshold:.6f}")

    print("\nCalculating test reconstruction errors...")

    errors = calculate_errors(model, X_scaled)
    anomalies = errors > threshold

    anomaly_count = int(anomalies.sum())
    positive_count = int(y.sum())

    anomaly_positive = int(((anomalies) & (y == 1)).sum())
    anomaly_negative = int(((anomalies) & (y == 0)).sum())

    anomaly_precision = (
        anomaly_positive / anomaly_count
        if anomaly_count else 0
    )

    anomaly_recall = (
        anomaly_positive / positive_count
        if positive_count else 0
    )

    overall_positive_rate = float(y.mean())
    anomaly_positive_rate = anomaly_precision

    enrichment = (
        anomaly_positive_rate / overall_positive_rate
        if overall_positive_rate else 0
    )

    results = {
        "threshold": float(threshold),
        "test_samples": len(y),
        "test_positive_samples": positive_count,
        "overall_positive_rate": overall_positive_rate,
        "mean_reconstruction_error": float(errors.mean()),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": float(100 * anomalies.mean()),
        "anomaly_positive_count": anomaly_positive,
        "anomaly_negative_count": anomaly_negative,
        "anomaly_positive_rate": anomaly_positive_rate,
        "anomaly_recall": anomaly_recall,
        "enrichment_ratio": float(enrichment)
    }

    pd.DataFrame({
        "month_period": df["month_period"],
        "grid_lat": df["grid_lat"],
        "grid_lon": df["grid_lon"],
        "reconstruction_error": errors,
        "is_anomaly": anomalies,
        "target_next_month": y
    }).to_csv(
        OUTPUT_DIR / "autoencoder_test_anomalies.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "autoencoder_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n========== AUTOENCODER TEST RESULTS ==========")

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()
