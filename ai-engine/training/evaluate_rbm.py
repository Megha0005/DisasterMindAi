
import os
import joblib
import numpy as np
import pandas as pd
import torch

from models.rbm.rbm_model import DisasterRBM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "train.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "rbm", "rbm_latest.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "rbm", "rbm_scaler.joblib"
)

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

print("Loading RBM and training data...")

df = pd.read_csv(TRAIN_PATH)

X = df.copy()

X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

for col in ["past_1m_count", "past_3m_count", "past_6m_count"]:
    X[col] = np.log1p(X[col])

X = X[FEATURES]

scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)

real_tensor = torch.tensor(
    X_scaled[:50000],
    dtype=torch.float32,
    device=DEVICE
)

model = DisasterRBM(
    visible_units=8,
    hidden_units=32
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Generating synthetic samples...")

with torch.no_grad():
    synthetic = model.generate(
        num_samples=50000,
        steps=500
    )

synthetic = synthetic.cpu().numpy()

# Compare distributions
real = real_tensor.cpu().numpy()

print("\n========== RBM GENERATION RESULTS ==========")

print("Real samples:", real.shape)
print("Synthetic samples:", synthetic.shape)

print("\nFeature-wise comparison:")

for i, feature in enumerate(FEATURES):

    real_mean = np.mean(real[:, i])
    synthetic_mean = np.mean(synthetic[:, i])

    real_std = np.std(real[:, i])
    synthetic_std = np.std(synthetic[:, i])

    print(f"\n{feature}")
    print("Real mean:", round(float(real_mean), 4))
    print("Synthetic mean:", round(float(synthetic_mean), 4))
    print("Real std:", round(float(real_std), 4))
    print("Synthetic std:", round(float(synthetic_std), 4))

# Overall diagnostics
print("\n========== QUALITY CHECK ==========")

print("NaN values:", np.isnan(synthetic).sum())
print("Infinite values:", np.isinf(synthetic).sum())

print("Real overall mean:", round(float(real.mean()), 4))
print("Synthetic overall mean:", round(float(synthetic.mean()), 4))

print("Real overall std:", round(float(real.std()), 4))
print("Synthetic overall std:", round(float(synthetic.std()), 4))

# Save generated data in original feature scale
synthetic_original = scaler.inverse_transform(synthetic)

output_dir = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(output_dir, exist_ok=True)

synthetic_df = pd.DataFrame(
    synthetic_original,
    columns=FEATURES
)

synthetic_df.to_csv(
    os.path.join(output_dir, "rbm_synthetic_samples.csv"),
    index=False
)

print("\nSynthetic samples saved successfully!")
