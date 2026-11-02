
import pandas as pd
import numpy as np
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

for name, df in [("train", train), ("validation", validation)]:

    print(f"\n========== {name.upper()} ==========")

    for target in ["past_1m_count", "past_3m_count", "past_6m_count"]:

        stats = df.groupby(
            ["grid_lat", "grid_lon"]
        )[target].agg(
            observations="size",
            positives=lambda x: (x > 0).sum(),
            tail_10=lambda x: (x >= 10).sum(),
            tail_25=lambda x: (x >= 25).sum(),
            tail_50=lambda x: (x >= 50).sum()
        )

        print(f"\n{target}")

        for column in ["positives", "tail_10", "tail_25", "tail_50"]:
            print(
                f"{column}: "
                f"zero cells={(stats[column] == 0).sum()}, "
                f"fewer than 5={(stats[column] < 5).sum()}, "
                f"median={stats[column].median()}"
            )

