
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from models.neural_network.mlp_model import DisasterMLP


# Configuration
TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

MODEL_DIR = Path("models/neural_network")
EVAL_DIR = Path("evaluation/neural_network")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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


def evaluate(model, loader):
    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)

            logits = model(features).squeeze(1)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.numpy())

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)

    roc_auc = roc_auc_score(targets, probabilities)
    pr_auc = average_precision_score(targets, probabilities)

    return roc_auc, pr_auc


def main():
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    y_train = train_df["target_next_month"].to_numpy(dtype=np.float32)
    y_val = val_df["target_next_month"].to_numpy(dtype=np.float32)

    print("Training rows:", f"{len(X_train):,}")
    print("Validation rows:", f"{len(X_val):,}")

    print("\nStandardizing features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    joblib.dump(
        scaler,
        MODEL_DIR / "mlp_scaler.joblib"
    )

    print("Scaler saved.")

    # Convert arrays into PyTorch tensors.
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
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

    # Handle class imbalance.
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)

    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=device
    )

    print(f"\nPositive training samples: {int(positives):,}")
    print(f"Negative training samples: {int(negatives):,}")
    print(f"Positive class weight: {pos_weight.item():.4f}")

    model = DisasterMLP(input_features=8).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting MLP training...")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for features, targets in train_loader:

            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features).squeeze(1)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(features)

        average_loss = total_loss / len(train_dataset)

        val_roc_auc, val_pr_auc = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {average_loss:.5f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f}"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "mlp_best.pth"
            )

            print("  Best model saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n========== MLP TRAINING SUMMARY ==========")
    print("Best epoch:", best_epoch)
    print(f"Best validation PR-AUC: {best_pr_auc:.4f}")
    print("Model saved:", MODEL_DIR / "mlp_best.pth")
    print("Scaler saved:", MODEL_DIR / "mlp_scaler.joblib")
    print("\nMLP training completed!")


if __name__ == "__main__":
    main()
