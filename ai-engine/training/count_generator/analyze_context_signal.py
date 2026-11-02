
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

for df in [train, validation]:
    df["recent"] = df["past_1m_count"]
    df["previous"] = (
        df["past_3m_count"] - df["past_1m_count"]
    )
    df["older"] = (
        df["past_6m_count"] - df["past_3m_count"]
    )

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

targets = ["recent", "previous", "older"]

feature_sets = {
    "Geography only": [
        "grid_lat", "grid_lon"
    ],
    "Time only": [
        "year", "month_sin", "month_cos"
    ],
    "Geography + time": [
        "grid_lat", "grid_lon",
        "year", "month_sin", "month_cos"
    ]
}

for name, features in feature_sets.items():

    print(f"\n========== {name} ==========")

    for target in targets:

        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )

        model.fit(train[features], train[target])

        predictions = model.predict(validation[features])
        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(
            validation[target], predictions
        )

        r2 = r2_score(
            validation[target], predictions
        )

        print(
            f"{target}: MAE={mae:.4f}, R2={r2:.4f}"
        )
