
import os
import joblib
import numpy as np
import pandas as pd
import torch

from scipy.stats import ks_2samp
from models.rbm.mixed_rbm_model import MixedDisasterRBM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data/processed/splits/train.csv"
)

MODEL_PATH = "models/rbm/mixed/mixed_rbm_best.pth"
SCALER_PATH = "models/rbm/mixed/mixed_rbm_scaler.joblib"

GAUSSIAN_FEATURES = [
    "grid_lat", "grid_lon", "year",
    "month_sin", "month_cos",
    "past_1m_count_intensity",
    "past_3m_count_intensity",
    "past_6m_count_intensity"
]

BINARY_FEATURES = [
    "past_1m_count_active",
    "past_3m_count_active",
    "past_6m_count_active"
]


def prepare_data(df):

    result = pd.DataFrame()

    result["grid_lat"] = df["grid_lat"]
    result["grid_lon"] = df["grid_lon"]
    result["year"] = df["year"]

    result["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    for col in [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ]:

        result[col + "_active"] = (
            df[col] > 0
        ).astype(float)

        result[col + "_intensity"] = np.log1p(
            df[col]
        )

    return result


print("Loading training dataset...")

df = pd.read_csv(TRAIN_PATH)
data = prepare_data(df)

scaler = joblib.load(SCALER_PATH)

real_g = scaler.transform(
    data[GAUSSIAN_FEATURES]
)

real_b = data[BINARY_FEATURES].to_numpy(
    dtype=np.float32
)

# Load model
model = MixedDisasterRBM(
    gaussian_units=8,
    binary_units=3,
    hidden_units=32
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Generating synthetic samples...")

with torch.no_grad():

    generated_g, generated_b = model.generate(
        num_samples=50000,
        steps=2000
    )

generated_g = generated_g.cpu().numpy()
generated_b = generated_b.cpu().numpy()

# Sample equal number of real observations
rng = np.random.default_rng(42)

indices = rng.choice(
    len(real_g),
    size=50000,
    replace=False
)

real_g_sample = real_g[indices]
real_b_sample = real_b[indices]

# Gaussian distribution comparison
print("\n========== GAUSSIAN FEATURE QUALITY ==========")

ks_values = []

for i, feature in enumerate(GAUSSIAN_FEATURES):

    statistic, p_value = ks_2samp(
        real_g_sample[:, i],
        generated_g[:, i]
    )

    ks_values.append(statistic)

    print(
        f"{feature}: KS={statistic:.4f}, "
        f"p-value={p_value:.6g}"
    )

print(
    "\nMean Gaussian KS:",
    round(float(np.mean(ks_values)), 4)
)

# Binary occurrence comparison
print("\n========== BINARY FEATURE QUALITY ==========")

for i, feature in enumerate(BINARY_FEATURES):

    real_rate = real_b_sample[:, i].mean()
    generated_rate = generated_b[:, i].mean()

    print(
        f"{feature}: "
        f"Real={real_rate:.4f}, "
        f"Generated={generated_rate:.4f}"
    )

# Correlation comparison
real_combined = np.column_stack(
    [real_g_sample, real_b_sample]
)

generated_combined = np.column_stack(
    [generated_g, generated_b]
)

real_corr = np.corrcoef(
    real_combined, rowvar=False
)

generated_corr = np.corrcoef(
    generated_combined, rowvar=False
)

correlation_error = np.mean(
    np.abs(real_corr - generated_corr)
)

print("\nMean absolute correlation difference:",
      round(float(correlation_error), 4))

# Save generated data
generated_original = scaler.inverse_transform(generated_g)

output = pd.DataFrame(
    generated_original,
    columns=GAUSSIAN_FEATURES
)

for i, feature in enumerate(BINARY_FEATURES):
    output[feature] = generated_b[:, i]

output_dir = os.path.join(
    BASE_DIR, "data", "synthetic"
)

os.makedirs(output_dir, exist_ok=True)

output.to_csv(
    os.path.join(
        output_dir,
        "mixed_rbm_synthetic_samples.csv"
    ),
    index=False
)

print("\nSynthetic data saved successfully!")
