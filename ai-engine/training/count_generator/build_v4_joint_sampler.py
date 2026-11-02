
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

required = [
    "grid_lat",
    "grid_lon",
    "past_1m_count",
    "past_3m_count",
    "past_6m_count"
]

train = train[required].copy()

train["recent"] = train["past_1m_count"]
train["previous"] = (
    train["past_3m_count"] - train["past_1m_count"]
)
train["older"] = (
    train["past_6m_count"] - train["past_3m_count"]
)

if (
    train[["recent", "previous", "older"]] < 0
).any().any():
    raise ValueError("Negative increments detected.")

# Geographic pooling
RADIUS = 12

train["lat_region"] = (
    np.floor(train["grid_lat"] / RADIUS) * RADIUS
)

train["lon_region"] = (
    np.floor(train["grid_lon"] / RADIUS) * RADIUS
)

sampler_data = {}

for key, group in train.groupby(
    ["lat_region", "lon_region"]
):
    sampler_data[key] = group[
        ["recent", "previous", "older"]
    ].to_numpy(dtype=np.int64)

output_dir = ROOT / "models/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    sampler_data,
    output_dir / "joint_sampler_12deg.joblib"
)

metadata = {
    "version": "V4",
    "pooling_radius_degrees": RADIUS,
    "training_rows": len(train),
    "regions": len(sampler_data),
    "sampling_method": "Joint empirical row sampling",
    "preserves_increment_hierarchy": True,
    "fallback": "Global empirical joint distribution"
}

joblib.dump(
    train[["recent", "previous", "older"]].to_numpy(
        dtype=np.int64
    ),
    output_dir / "global_joint_distribution.joblib"
)

with open(output_dir / "joint_sampler_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("V4 joint sampler saved.")
print("Pooling radius:", RADIUS)
print("Regions:", len(sampler_data))
print("Training rows:", len(train))
print("Hierarchy verification:", bool(
    (
        train["recent"] <=
        train["recent"] + train["previous"]
    ).all()
    and
    (
        train["recent"] + train["previous"] <=
        train["recent"] + train["previous"] + train["older"]
    ).all()
))
