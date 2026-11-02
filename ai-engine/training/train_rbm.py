
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from models.rbm.rbm_model import DisasterRBM


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "grid_lat", "grid_lon", "year",
    "month_sin", "month_cos",
    "past_1m_count", "past_3m_count", "past_6m_count"
]

BATCH_SIZE = 2048
EPOCHS = 30
LEARNING_RATE = 0.0005
HIDDEN_UNITS = 32
CD_STEPS = 1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "train.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "validation.csv"
)

SAVE_DIR = os.path.join(BASE_DIR, "models", "rbm")
os.makedirs(SAVE_DIR, exist_ok=True)


# Load data
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


def prepare_features(df):
    df = df.copy()

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for col in ["past_1m_count", "past_3m_count", "past_6m_count"]:
        df[col] = np.log1p(df[col])

    return df[FEATURES]


X_train = prepare_features(train_df)
X_val = prepare_features(val_df)

# Scale using training data only
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_tensor = torch.tensor(X_train, dtype=torch.float32)
val_tensor = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    TensorDataset(val_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE.type == "cuda")
)


# Initialize RBM
model = DisasterRBM(
    visible_units=8,
    hidden_units=HIDDEN_UNITS
).to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)


# CD-1 training
print("\nStarting RBM training...")
print("Device:", DEVICE)

for epoch in range(EPOCHS):

    model.train()

    train_error = 0
    train_free_energy = 0
    train_samples = 0

    for (batch,) in train_loader:

        batch = batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Positive phase
        positive_hidden_prob, _ = model.sample_hidden(batch)

        # Negative phase
        _, visible_mean, negative_visible, negative_hidden_prob = (
            model.contrastive_divergence(batch, k=CD_STEPS)
        )

        # Contrastive Divergence parameter updates
        positive_association = torch.matmul(
            batch.T, positive_hidden_prob
        )

        negative_association = torch.matmul(
            negative_visible.T, negative_hidden_prob
        )

        batch_size = batch.size(0)

        weight_gradient = (
            positive_association - negative_association
        ) / batch_size

        visible_gradient = (
            batch - negative_visible
        ).mean(dim=0)

        hidden_gradient = (
            positive_hidden_prob - negative_hidden_prob
        ).mean(dim=0)

        model.W.grad = -weight_gradient
        model.visible_bias.grad = -visible_gradient
        model.hidden_bias.grad = -hidden_gradient

        optimizer.step()

        # Monitoring reconstruction error
        reconstruction_error = torch.mean(
            (batch - visible_mean) ** 2
        )

        train_error += reconstruction_error.item() * batch_size

        train_free_energy += (
            model.free_energy(batch).mean().item() * batch_size
        )

        train_samples += batch_size

    # Validation
    model.eval()

    val_error = 0
    val_samples = 0

    with torch.no_grad():

        for (batch,) in val_loader:

            batch = batch.to(DEVICE, non_blocking=True)

            hidden_prob, hidden_sample = model.sample_hidden(batch)

            visible_mean, _ = model.sample_visible(hidden_prob)

            error = torch.mean((batch - visible_mean) ** 2)

            val_error += error.item() * batch.size(0)
            val_samples += batch.size(0)

    train_avg = train_error / train_samples
    val_avg = val_error / val_samples
    free_energy_avg = train_free_energy / train_samples

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Reconstruction: {train_avg:.5f} "
        f"Validation Reconstruction: {val_avg:.5f} "
        f"Free Energy: {free_energy_avg:.5f}"
    )

    # Save checkpoint
    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, "rbm_latest.pth")
    )

    joblib.dump(
        scaler,
        os.path.join(SAVE_DIR, "rbm_scaler.joblib")
    )


print("\nRBM training completed successfully!")
