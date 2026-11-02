
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from models.autoencoder.autoencoder_model import DisasterAutoencoder


TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

MODEL_FILE = Path("models/autoencoder/autoencoder_best.pth")
SCALER_FILE = Path("models/autoencoder/autoencoder_scaler.joblib")

OUTPUT_DIR = Path("evaluation/autoencoder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4096
ANOMALY_PERCENTILE = 99

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def prepare_features(df):
    month_period = pd.PeriodIndex(
        df["month_period"],
        freq="M"
    )

    X = pd.DataFrame(index=df.index)

    X["grid_lat"] = df["grid_lat"]
    X["grid_lon"] = df["grid_lon"]
    X["year"] = month_period.year

    X["month_sin"] = np.sin(
        2 * np.pi * month_period.month / 12
    )

    X["month_cos"] = np.cos(
        2 * np.pi * month_period.month / 12
    )

    X["past_1m_count"] = np.log1p(
        df["past_1m_count"]
    )

    X["past_3m_count"] = np.log1p(
        df["past_3m_count"]
    )

    X["past_6m_count"] = np.log1p(
        df["past_6m_count"]
    )

    return X.astype("float32")


def calculate_errors(model, features):
    model.eval()

    errors = []

    with torch.no_grad():

        for start in range(0, len(features), BATCH_SIZE):

            batch = torch.from_numpy(
                features[start:start + BATCH_SIZE]
            ).to(device)

            reconstructed, latent = model(batch)

            # Mean squared reconstruction error per sample.
            batch_errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            errors.extend(
                batch_errors.cpu().numpy()
            )

    return np.asarray(errors)


def main():

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    scaler = joblib.load(SCALER_FILE)

    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    model = DisasterAutoencoder(
        input_features=8,
        latent_dim=3
    ).to(device)

    state_dict = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print("Model loaded successfully.")

    print("\nCalculating training reconstruction errors...")

    train_errors = calculate_errors(
        model,
        X_train
    )

    print("Calculating validation reconstruction errors...")

    val_errors = calculate_errors(
        model,
        X_val
    )

    # Threshold derived only from training errors.
    threshold = float(
        np.percentile(
            train_errors,
            ANOMALY_PERCENTILE
        )
    )

    train_anomalies = train_errors > threshold
    val_anomalies = val_errors > threshold

    # Validation target is used only for descriptive analysis.
    y_val = val_df["target_next_month"].to_numpy(dtype=int)

    anomaly_df = pd.DataFrame({
        "month_period": val_df["month_period"].to_numpy(),
        "grid_lat": val_df["grid_lat"].to_numpy(),
        "grid_lon": val_df["grid_lon"].to_numpy(),
        "reconstruction_error": val_errors,
        "is_anomaly": val_anomalies,
        "target_next_month": y_val
    })

    anomaly_df.to_csv(
        OUTPUT_DIR / "validation_anomaly_results.csv",
        index=False
    )

    metrics = {
        "model_name": "DisasterAutoencoder",
        "threshold_method": "99th percentile of training reconstruction errors",
        "threshold": threshold,
        "training_reconstruction": {
            "mean": float(np.mean(train_errors)),
            "median": float(np.median(train_errors)),
            "std": float(np.std(train_errors)),
            "anomaly_count": int(train_anomalies.sum()),
            "anomaly_percentage": float(
                100 * train_anomalies.mean()
            )
        },
        "validation_reconstruction": {
            "mean": float(np.mean(val_errors)),
            "median": float(np.median(val_errors)),
            "std": float(np.std(val_errors)),
            "anomaly_count": int(val_anomalies.sum()),
            "anomaly_percentage": float(
                100 * val_anomalies.mean()
            )
        },
        "validation_anomaly_overlap": {
            "anomalies_with_positive_next_month": int(
                ((val_anomalies) & (y_val == 1)).sum()
            ),
            "anomalies_with_negative_next_month": int(
                ((val_anomalies) & (y_val == 0)).sum()
            )
        }
    }

    with open(
        OUTPUT_DIR / "anomaly_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\n========== ANOMALY DETECTION RESULTS ==========")

    print(f"\nTraining mean error: {np.mean(train_errors):.6f}")
    print(f"Validation mean error: {np.mean(val_errors):.6f}")

    print(
        f"\nAnomaly threshold ({ANOMALY_PERCENTILE}th percentile): "
        f"{threshold:.6f}"
    )

    print(
        f"Training anomalies: {train_anomalies.sum():,} "
        f"({100 * train_anomalies.mean():.2f}%)"
    )

    print(
        f"Validation anomalies: {val_anomalies.sum():,} "
        f"({100 * val_anomalies.mean():.2f}%)"
    )

    print(
        "\nValidation anomalies with positive next-month label:",
        int(((val_anomalies) & (y_val == 1)).sum())
    )

    print(
        "Validation anomalies with negative next-month label:",
        int(((val_anomalies) & (y_val == 0)).sum())
    )

    print("\nSaved:", OUTPUT_DIR / "validation_anomaly_results.csv")
    print("Saved:", OUTPUT_DIR / "anomaly_metrics.json")

    print("\nAutoencoder evaluation completed!")


if __name__ == "__main__":
    main()
