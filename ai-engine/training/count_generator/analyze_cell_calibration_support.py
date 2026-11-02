
import pandas as pd
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation)
]:
    print(f"\n========== {name} ==========")

    df = df.copy()

    df["recent"] = df["past_1m_count"]
    df["previous"] = df["past_3m_count"] - df["past_1m_count"]
    df["older"] = df["past_6m_count"] - df["past_3m_count"]

    for target in ["recent", "previous", "older"]:

        cell_stats = df.groupby(
            ["grid_lat", "grid_lon"]
        )[target].agg(
            observations="size",
            positives=lambda x: (x > 0).sum(),
            negatives=lambda x: (x == 0).sum()
        )

        print(f"\n{target.upper()}")
        print("Cells:", len(cell_stats))
        print("Cells with zero positives:", (cell_stats.positives == 0).sum())
        print("Cells with fewer than 5 positives:", (cell_stats.positives < 5).sum())
        print("Cells with fewer than 5 negatives:", (cell_stats.negatives < 5).sum())
        print("Median positives per cell:", cell_stats.positives.median())
        print("Median negatives per cell:", cell_stats.negatives.median())

