
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

files = {
    "TRAIN": ROOT / "data/processed/count_generator_train.csv",
    "VALIDATION": ROOT / "data/processed/count_generator_validation.csv"
}

for name, path in files.items():

    df = pd.read_csv(
        path,
        usecols=[
            "grid_lat",
            "grid_lon",
            "past_1m_count",
            "past_3m_count",
            "past_6m_count"
        ]
    )

    df["region"] = (
        df["grid_lat"].astype(str)
        + "_"
        + df["grid_lon"].astype(str)
    )

    region_stats = df.groupby("region").agg(
        observations=("past_6m_count", "size"),
        mean_1m=("past_1m_count", "mean"),
        mean_3m=("past_3m_count", "mean"),
        mean_6m=("past_6m_count", "mean")
    )

    print(f"\n========== {name} ==========")

    print("Unique geographic cells:", len(region_stats))
    print("Total observations:", len(df))

    print("\nRegional mean 6-month count:")
    print(region_stats.mean_6m.describe().round(4))

    print("\nMost active 10 regions:")
    print(
        region_stats.sort_values(
            "mean_6m",
            ascending=False
        ).head(10).round(4)
    )
