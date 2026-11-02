
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/raw/usgs_earthquakes_2000_2025_m4.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "usgs_earthquakes_cleaned.csv"


def clean_usgs_dataset():
    print("Loading USGS dataset...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    original_count = len(df)

    # Convert timestamps.
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True
    )

    # Remove records without essential information.
    essential_columns = [
        "time",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "id"
    ]

    df = df.dropna(subset=essential_columns)

    # Keep only earthquake events.
    df = df[df["type"] == "earthquake"].copy()

    # Remove duplicate event IDs.
    df = df.drop_duplicates(subset=["id"])

    # Enforce the original collection threshold.
    df = df[df["mag"] >= 4.0].copy()

    # Validate geographic coordinates.
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    # Preserve negative depths; flag them for analysis.
    df["negative_depth_flag"] = (
        df["depth"] < 0
    ).astype("int8")

    # Extract time-based features.
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_of_week

    # Sort chronologically.
    df = df.sort_values("time").reset_index(drop=True)

    # Save cleaned data separately.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== CLEANING REPORT ==========")
    print(f"Original records: {original_count:,}")
    print(f"Cleaned records: {len(df):,}")
    print(f"Records removed: {original_count - len(df):,}")
    print(
        "Negative depth records retained:",
        int(df["negative_depth_flag"].sum())
    )
    print(f"Final columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")

    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    clean_usgs_dataset()
