
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
            "year",
            "month",
            "past_1m_count",
            "past_3m_count",
            "past_6m_count"
        ]
    )

    df["recent_1m"] = df["past_1m_count"]
    df["previous_2m"] = (
        df["past_3m_count"] - df["past_1m_count"]
    )
    df["older_3m"] = (
        df["past_6m_count"] - df["past_3m_count"]
    )

    print(f"\n========== {name} ==========")

    print("\nActivity by year:")

    yearly = df.groupby("year").agg(
        mean_1m=("recent_1m", "mean"),
        mean_2m=("previous_2m", "mean"),
        mean_3m=("older_3m", "mean")
    )

    print(yearly.round(4))

    print("\nActivity by calendar month:")

    monthly = df.groupby("month").agg(
        mean_1m=("recent_1m", "mean"),
        mean_2m=("previous_2m", "mean"),
        mean_3m=("older_3m", "mean")
    )

    print(monthly.round(4))

    print("\nYearly observation count:")
    print(df.groupby("year").size().describe())

