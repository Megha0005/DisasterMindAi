
import os
import joblib
import numpy as np
import pandas as pd
import torch

from scipy.stats import ks_2samp
from models.rbm.rbm_model import DisasterRBM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = [
    "grid_lat", "grid_lon", "year",
    "month_sin", "month_cos",
    "past_1m_count", "past_3m_count", "past_6m_count"
]

# Load full training data
df = pd.read_csv(
    os.path.join(BASE_DIR, "data/processed/splits/train.csv")
)

X = df.copy()

X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

for col in ["past_1m_count", "past_3m_count", "past_6m_count"]:
    X[col] = np.log1p(X[col])

scaler = joblib.load(
    os.path.join(BASE_DIR, "models/rbm/rbm_scaler.joblib")
)

real = scaler.transform(X[FEATURES])

# Load model
model = DisasterRBM(8, 32).to(DEVICE)

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models/rbm/rbm_latest.pth"),
        map_location=DEVICE
    )
)

model.eval()

# Generate synthetic samples
with torch.no_grad():
    synthetic = model.generate(
        num_samples=50000,
        steps=2000
    ).cpu().numpy()

# Use equal-sized random samples
rng = np.random.default_rng(42)

real_sample = real[
    rng.choice(len(real), 50000, replace=False)
]

print("\n========== DISTRIBUTION COMPARISON ==========")

ks_values = []

for i, feature in enumerate(FEATURES):

    statistic, p_value = ks_2samp(
        real_sample[:, i],
        synthetic[:, i]
    )

    ks_values.append(statistic)

    print(
        f"{feature}: "
        f"KS={statistic:.4f}, "
        f"p-value={p_value:.6g}"
    )

print("\nMean KS statistic:", round(float(np.mean(ks_values)), 4))

# Correlation comparison
real_corr = np.corrcoef(real_sample, rowvar=False)
synthetic_corr = np.corrcoef(synthetic, rowvar=False)

correlation_error = np.mean(
    np.abs(real_corr - synthetic_corr)
)

print(
    "Mean absolute correlation difference:",
    round(float(correlation_error), 4)
)

print("\n========== QUALITY CHECK COMPLETE ==========")
