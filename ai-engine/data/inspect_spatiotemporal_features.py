
from pathlib import Path

import pandas as pd


FILE = Path(
    "data/processed/earthquake_spatiotemporal_features.csv"
)


def main():
    df = pd.read_csv(FILE)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum().to_string())

    print("\n========== DUPLICATE CELL-MONTH ROWS ==========")
    duplicates = df.duplicated(
        subset=["grid_lat", "grid_lon", "month_period"]
    ).sum()
    print(f"Duplicates: {duplicates}")

    print("\n========== MONTH RANGE ==========")
    print("First month:", df["month_period"].min())
    print("Last month:", df["month_period"].max())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["target_next_month"].value_counts().to_string())

    print("\n========== TARGET BY YEAR ==========")
    yearly = (
        df.groupby("year")["target_next_month"]
        .agg(["count", "sum", "mean"])
    )
    yearly["positive_percentage"] = (
        yearly["mean"] * 100
    ).round(2)
    print(yearly.to_string())

    print("\n========== HISTORICAL FEATURE STATISTICS ==========")
    features = [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ]
    print(df[features].describe().to_string())

    print("\n========== SAMPLE RECORDS ==========")
    print(df.head(10).to_string(index=False))

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
