
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models.autoencoder.autoencoder_model import DisasterAutoencoder


TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

MODEL_DIR = Path("models/autoencoder")
EVAL_DIR = Path("evaluation/autoencoder")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2048
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 4

RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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


def calculate_validation_loss(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for features in loader:

            features = features[0].to(
                device,
                non_blocking=True
            )

            reconstructed, _ = model(features)

            loss = criterion(reconstructed, features)

            total_loss += loss.item() * len(features)
            total_samples += len(features)

    return total_loss / total_samples


def main():

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    print("Training samples:", f"{len(X_train):,}")
    print("Validation samples:", f"{len(X_val):,}")
    print("Input features:", len(X_train.columns))

    # Fit scaler only on training features.
    print("\nStandardizing features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    joblib.dump(
        scaler,
        MODEL_DIR / "autoencoder_scaler.joblib"
    )

    # Unsupervised datasets: no target labels.
    train_dataset = TensorDataset(
        torch.from_numpy(X_train)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = DisasterAutoencoder(
        input_features=8,
        latent_dim=3
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Autoencoder training...")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0
        total_samples = 0

        for batch in train_loader:

            features = batch[0].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            reconstructed, latent = model(features)

            loss = criterion(
                reconstructed,
                features
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(features)
            total_samples += len(features)

        train_loss = total_train_loss / total_samples

        val_loss = calculate_validation_loss(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "autoencoder_best.pth"
            )

            print("  Best model saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    summary = {
        "model_name": "DisasterAutoencoder",
        "input_features": 8,
        "latent_dimensions": 3,
        "best_epoch": best_epoch,
        "best_validation_mse": float(best_val_loss),
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset)
    }

    import json

    with open(
        EVAL_DIR / "training_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== AUTOENCODER TRAINING SUMMARY ==========")
    print("Best epoch:", best_epoch)
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print("Model saved:", MODEL_DIR / "autoencoder_best.pth")
    print("Scaler saved:", MODEL_DIR / "autoencoder_scaler.joblib")
    print("\nAutoencoder training completed!")


if __name__ == "__main__":
    main()
