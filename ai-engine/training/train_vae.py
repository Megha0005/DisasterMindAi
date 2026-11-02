
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.vae.vae_model import DisasterVAE


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "grid_lat",
    "grid_lon",
    "year",
    "month_sin",
    "month_cos",
    "past_1m_count",
    "past_3m_count",
    "past_6m_count"
]

BATCH_SIZE = 2048
EPOCHS = 30
LEARNING_RATE = 0.001
BETA = 0.1
LATENT_DIM = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "train.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "validation.csv"
)

SAVE_DIR = os.path.join(BASE_DIR, "models", "vae")
os.makedirs(SAVE_DIR, exist_ok=True)


# Load datasets
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.copy()
X_val = val_df.copy()

# Generate cyclical month features
for df in [X_train, X_val]:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

X_train = X_train[FEATURES].copy()
X_val = X_val[FEATURES].copy()

# Apply log transformation to count features
for col in ["past_1m_count", "past_3m_count", "past_6m_count"]:
    X_train[col] = np.log1p(X_train[col])
    X_val[col] = np.log1p(X_val[col])

# Standardize using training data only
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to tensors
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


# Initialize model
model = DisasterVAE(
    input_features=len(FEATURES),
    latent_dim=LATENT_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# VAE loss
def vae_loss(reconstructed, original, mu, logvar):

    reconstruction_loss = nn.functional.mse_loss(
        reconstructed,
        original,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + BETA * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# Training
best_val_loss = float("inf")

print("\nStarting VAE training...")
print("Device:", DEVICE)

for epoch in range(EPOCHS):

    model.train()

    train_total = 0
    train_reconstruction = 0
    train_kl = 0
    train_samples = 0

    for (batch,) in train_loader:

        batch = batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        reconstructed, mu, logvar = model(batch)

        loss, recon_loss, kl_loss = vae_loss(
            reconstructed, batch, mu, logvar
        )

        loss.backward()

        optimizer.step()

        train_total += loss.item()
        train_reconstruction += recon_loss.item()
        train_kl += kl_loss.item()
        train_samples += batch.size(0)

    # Validation
    model.eval()

    val_total = 0
    val_reconstruction = 0
    val_kl = 0
    val_samples = 0

    with torch.no_grad():

        for (batch,) in val_loader:

            batch = batch.to(DEVICE, non_blocking=True)

            reconstructed, mu, logvar = model(batch)

            loss, recon_loss, kl_loss = vae_loss(
                reconstructed, batch, mu, logvar
            )

            val_total += loss.item()
            val_reconstruction += recon_loss.item()
            val_kl += kl_loss.item()
            val_samples += batch.size(0)

    train_avg = train_total / train_samples
    val_avg = val_total / val_samples

    train_recon_avg = train_reconstruction / train_samples
    train_kl_avg = train_kl / train_samples

    val_recon_avg = val_reconstruction / val_samples
    val_kl_avg = val_kl / val_samples

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train: {train_avg:.5f} "
        f"Val: {val_avg:.5f} "
        f"Val Recon: {val_recon_avg:.5f} "
        f"Val KL: {val_kl_avg:.5f}"
    )

    # Save best model
    if val_avg < best_val_loss:

        best_val_loss = val_avg

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "vae_best.pth")
        )

        joblib.dump(
            scaler,
            os.path.join(SAVE_DIR, "vae_scaler.joblib")
        )

        print("Best VAE model saved!")

print("\nVAE training completed.")
print("Best validation loss:", best_val_loss)
