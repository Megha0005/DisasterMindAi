
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models.rbm.mixed_rbm_model import MixedDisasterRBM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2048
EPOCHS = 30
LEARNING_RATE = 0.0005
HIDDEN_UNITS = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data/processed/splits/train.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR, "data/processed/splits/validation.csv"
)

SAVE_DIR = os.path.join(BASE_DIR, "models/rbm/mixed")
os.makedirs(SAVE_DIR, exist_ok=True)


# Prepare features
def prepare_features(df):

    X = pd.DataFrame(index=df.index)

    X["grid_lat"] = df["grid_lat"]
    X["grid_lon"] = df["grid_lon"]
    X["year"] = df["year"]

    X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for col in ["past_1m_count", "past_3m_count", "past_6m_count"]:
        X[col + "_active"] = (df[col] > 0).astype(float)
        X[col + "_intensity"] = np.log1p(df[col])

    gaussian_cols = [
        "grid_lat", "grid_lon", "year",
        "month_sin", "month_cos",
        "past_1m_count_intensity",
        "past_3m_count_intensity",
        "past_6m_count_intensity"
    ]

    binary_cols = [
        "past_1m_count_active",
        "past_3m_count_active",
        "past_6m_count_active"
    ]

    return X[gaussian_cols], X[binary_cols]


print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

G_train, B_train = prepare_features(train_df)
G_val, B_val = prepare_features(val_df)

# Scale Gaussian features only
scaler = StandardScaler()

G_train = scaler.fit_transform(G_train)
G_val = scaler.transform(G_val)

train_g = torch.tensor(G_train, dtype=torch.float32)
train_b = torch.tensor(B_train.values, dtype=torch.float32)

val_g = torch.tensor(G_val, dtype=torch.float32)
val_b = torch.tensor(B_val.values, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_g, train_b),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    TensorDataset(val_g, val_b),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE.type == "cuda")
)


# Initialize model
model = MixedDisasterRBM(
    gaussian_units=8,
    binary_units=3,
    hidden_units=HIDDEN_UNITS
).to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)


def train_cd(batch_g, batch_b):

    # Positive phase
    positive_prob, _ = model.sample_hidden(batch_g, batch_b)

    # Negative phase: CD-1
    _, hidden = model.sample_hidden(batch_g, batch_b)

    (
        negative_g_mean,
        negative_g,
        negative_b_prob,
        negative_b
    ) = model.sample_visible(hidden)

    negative_hidden_prob, _ = model.sample_hidden(
        negative_g, negative_b
    )

    n = batch_g.size(0)

    # Parameter gradients
    grad_wg = (
        batch_g.T @ positive_prob
        - negative_g.T @ negative_hidden_prob
    ) / n

    grad_wb = (
        batch_b.T @ positive_prob
        - negative_b.T @ negative_hidden_prob
    ) / n

    grad_bg = (batch_g - negative_g).mean(dim=0)
    grad_bb = (batch_b - negative_b).mean(dim=0)

    grad_h = (positive_prob - negative_hidden_prob).mean(dim=0)

    optimizer.zero_grad()

    model.W_gaussian.grad = -grad_wg
    model.W_binary.grad = -grad_wb
    model.visible_bias_gaussian.grad = -grad_bg
    model.visible_bias_binary.grad = -grad_bb
    model.hidden_bias.grad = -grad_h

    optimizer.step()

    return negative_g_mean, negative_b_prob


# Training
print("\nStarting Mixed RBM training...")
print("Device:", DEVICE)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_g_error = 0
    train_b_error = 0
    train_samples = 0

    for batch_g, batch_b in train_loader:

        batch_g = batch_g.to(DEVICE, non_blocking=True)
        batch_b = batch_b.to(DEVICE, non_blocking=True)

        negative_g_mean, negative_b_prob = train_cd(
            batch_g, batch_b
        )

        g_error = torch.mean(
            (batch_g - negative_g_mean) ** 2
        )

        b_error = torch.nn.functional.binary_cross_entropy(
            negative_b_prob,
            batch_b
        )

        n = batch_g.size(0)

        train_g_error += g_error.item() * n
        train_b_error += b_error.item() * n
        train_samples += n

    # Validation
    model.eval()

    val_g_error = 0
    val_b_error = 0
    val_samples = 0

    with torch.no_grad():

        for batch_g, batch_b in val_loader:

            batch_g = batch_g.to(DEVICE, non_blocking=True)
            batch_b = batch_b.to(DEVICE, non_blocking=True)

            hidden_prob, _ = model.sample_hidden(batch_g, batch_b)

            g_mean, _, b_prob, _ = model.sample_visible(hidden_prob)

            g_loss = torch.mean((batch_g - g_mean) ** 2)

            b_loss = torch.nn.functional.binary_cross_entropy(
                b_prob, batch_b
            )

            n = batch_g.size(0)

            val_g_error += g_loss.item() * n
            val_b_error += b_loss.item() * n
            val_samples += n

    train_g_avg = train_g_error / train_samples
    train_b_avg = train_b_error / train_samples

    val_g_avg = val_g_error / val_samples
    val_b_avg = val_b_error / val_samples

    total_val = val_g_avg + val_b_avg

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Gaussian: {train_g_avg:.5f} "
        f"Train Binary: {train_b_avg:.5f} "
        f"Val Gaussian: {val_g_avg:.5f} "
        f"Val Binary: {val_b_avg:.5f}"
    )

    if total_val < best_val_loss:

        best_val_loss = total_val

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "mixed_rbm_best.pth")
        )

        joblib.dump(
            scaler,
            os.path.join(SAVE_DIR, "mixed_rbm_scaler.joblib")
        )

        print("Best mixed RBM saved!")

print("\nMixed RBM training completed!")
print("Best validation loss:", best_val_loss)
