
import pandas as pd
import numpy as np

from pathlib import Path

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

radii = [4, 8, 12, 20]

for target in ["recent", "previous", "older"]:

    print(f"\n========== {target.upper()} ==========")

    for radius in radii:

        data = train.copy()

        data["lat_region"] = (
            np.floor(data["grid_lat"] / radius) * radius
        )

        data["lon_region"] = (
            np.floor(data["grid_lon"] / radius) * radius
        )

        grouped = data.groupby(
            ["lat_region", "lon_region"]
        )[target]

        stats = grouped.agg(
            observations="size",
            positives=lambda x: (x > 0).sum(),
            tail_10=lambda x: (x >= 10).sum(),
            tail_25=lambda x: (x >= 25).sum()
        )

        print(f"\nRadius: {radius} degrees")
        print("Regions:", len(stats))
        print("Median observations:", stats.observations.median())
        print("Median positives:", stats.positives.median())
        print("Regions with fewer than 20 positives:", (stats.positives < 20).sum())
        print("Regions with fewer than 5 tail-10 samples:", (stats.tail_10 < 5).sum())
        print("Regions with fewer than 5 tail-25 samples:", (stats.tail_25 < 5).sum())

