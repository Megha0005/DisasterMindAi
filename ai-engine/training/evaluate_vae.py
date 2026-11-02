
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.vae.vae_model import DisasterVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "grid_lat", "grid_lon", "year",
    "month_sin", "month_cos",
    "past_1m_count", "past_3m_count", "past_6m_count"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "test.csv"
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "vae", "vae_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "models", "vae", "vae_scaler.joblib")

# Load test data
print("Loading test dataset...")

df = pd.read_csv(TEST_PATH)

X = df.copy()

for feature in ["past_1m_count", "past_3m_count", "past_6m_count"]:
    X[feature] = np.log1p(X[feature])

X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

X = X[FEATURES]

scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

tensor = torch.tensor(X_scaled, dtype=torch.float32)

loader = DataLoader(
    TensorDataset(tensor),
    batch_size=2048,
    shuffle=False
)

# Load trained model
model = DisasterVAE(input_features=8, latent_dim=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

reconstruction_errors = []
latent_vectors = []

print("Evaluating VAE...")

with torch.no_grad():
    for (batch,) in loader:

        batch = batch.to(DEVICE)

        # Use mean instead of random sampling for stable evaluation
        mu, logvar = model.encode(batch)
        reconstructed = model.decode(mu)

        errors = F.mse_loss(
            reconstructed,
            batch,
            reduction="none"
        ).mean(dim=1)

        reconstruction_errors.extend(errors.cpu().numpy())
        latent_vectors.extend(mu.cpu().numpy())

errors = np.array(reconstruction_errors)
latent_vectors = np.array(latent_vectors)

# Training-derived anomaly threshold
train_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "processed", "splits", "train.csv")
)

X_train = train_df.copy()

for feature in ["past_1m_count", "past_3m_count", "past_6m_count"]:
    X_train[feature] = np.log1p(X_train[feature])

X_train["month_sin"] = np.sin(2 * np.pi * X_train["month"] / 12)
X_train["month_cos"] = np.cos(2 * np.pi * X_train["month"] / 12)

X_train = scaler.transform(X_train[FEATURES])

train_tensor = torch.tensor(X_train, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=2048,
    shuffle=False
)

train_errors = []

with torch.no_grad():
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        mu, _ = model.encode(batch)
        reconstructed = model.decode(mu)

        batch_errors = F.mse_loss(
            reconstructed,
            batch,
            reduction="none"
        ).mean(dim=1)

        train_errors.extend(batch_errors.cpu().numpy())

threshold = np.percentile(train_errors, 99)

# Anomaly detection
anomalies = errors > threshold

df["reconstruction_error"] = errors
df["is_anomaly"] = anomalies
df["latent_1"] = latent_vectors[:, 0]
df["latent_2"] = latent_vectors[:, 1]
df["latent_3"] = latent_vectors[:, 2]

# Results
positive = df["target_next_month"].astype(bool)

anomaly_count = anomalies.sum()
anomaly_positive = (anomalies & positive.to_numpy()).sum()

print("\n========== VAE TEST RESULTS ==========")
print("Test rows:", len(df))
print("Threshold:", round(float(threshold), 6))
print("Anomalies:", anomaly_count)
print("Anomaly percentage:", round(anomaly_count / len(df) * 100, 2), "%")
print("Mean reconstruction error:", round(float(errors.mean()), 6))
print("Overall positive rate:", round(float(positive.mean() * 100), 2), "%")

if anomaly_count > 0:
    anomaly_rate = anomaly_positive / anomaly_count
    recall = anomaly_positive / max(positive.sum(), 1)
    enrichment = anomaly_rate / max(float(positive.mean()), 1e-12)

    print("Anomaly positive rate:", round(anomaly_rate * 100, 2), "%")
    print("Anomaly recall:", round(recall * 100, 2), "%")
    print("Enrichment ratio:", round(enrichment, 3), "x")

# Save results
output_dir = os.path.join(BASE_DIR, "evaluation", "vae")
os.makedirs(output_dir, exist_ok=True)

df.to_csv(
    os.path.join(output_dir, "vae_test_predictions.csv"),
    index=False
)

print("\nResults saved to evaluation/vae/vae_test_predictions.csv")
