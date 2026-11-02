
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

train["recent"] = train["past_1m_count"]
train["previous"] = (
    train["past_3m_count"] - train["past_1m_count"]
)
train["older"] = (
    train["past_6m_count"] - train["past_3m_count"]
)

# Use the same historical period as the occurrence models.
base_train = train[train["year"] <= 2016].copy()

features = ["grid_lat", "grid_lon"]
targets = ["recent", "previous", "older"]

output_dir = ROOT / "models/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

metadata = {}

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    positive_data = base_train[
        base_train[target] > 0
    ].copy()

    X = positive_data[features]
    y = np.log1p(positive_data[target])

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    model_path = output_dir / f"{target}_magnitude_model.joblib"

    joblib.dump(model, model_path)

    print("Positive training observations:", len(positive_data))
    print("Mean positive count:", round(positive_data[target].mean(), 4))
    print("Saved:", model_path.name)

    metadata[target] = {
        "model": model_path.name,
        "target_transform": "log1p",
        "training_observations": int(len(positive_data)),
        "training_period": "2000-2016"
    }

with open(output_dir / "magnitude_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nV4 magnitude models saved successfully.")
