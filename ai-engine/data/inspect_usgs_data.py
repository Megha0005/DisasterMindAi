
from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/raw/usgs_earthquakes_2000_2025_m4.csv"
)


def inspect_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    print("\n========== FIRST FIVE RECORDS ==========")
    print(df.head().to_string())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes.to_string())

    print("\n========== MISSING VALUES ==========")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2)
    })

    print(
        missing.sort_values(
            "missing_percentage",
            ascending=False
        ).to_string()
    )

    print("\n========== DUPLICATE RECORDS ==========")
    print(f"Exact duplicates: {df.duplicated().sum()}")
    print(f"Duplicate event IDs: {df['id'].duplicated().sum()}")

    print("\n========== DATE RANGE ==========")
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True
    )

    print(f"First event: {df['time'].min()}")
    print(f"Last event: {df['time'].max()}")
    print(f"Invalid timestamps: {df['time'].isna().sum()}")

    print("\n========== MAGNITUDE STATISTICS ==========")
    print(df["mag"].describe().to_string())

    print("\n========== DEPTH STATISTICS ==========")
    print(df["depth"].describe().to_string())

    print("\n========== GEOGRAPHIC COVERAGE ==========")
    print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
    print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")

    print("\n========== MAGNITUDE TYPES ==========")
    print(df["magType"].value_counts(dropna=False).head(15).to_string())

    print("\n========== EVENT TYPES ==========")
    print(df["type"].value_counts(dropna=False).to_string())

    print("\n========== EVENT STATUS ==========")
    print(df["status"].value_counts(dropna=False).to_string())

    print("\n========== DATA VALIDATION ==========")
    print("Magnitude below 4:", (df["mag"] < 4).sum())
    print("Missing magnitude:", df["mag"].isna().sum())
    print("Invalid latitude:", (~df["latitude"].between(-90, 90)).sum())
    print("Invalid longitude:", (~df["longitude"].between(-180, 180)).sum())
    print("Negative depth:", (df["depth"] < 0).sum())

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    inspect_dataset()
