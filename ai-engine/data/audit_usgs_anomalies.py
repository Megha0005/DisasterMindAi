
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/raw/usgs_earthquakes_2000_2025_m4.csv"
)

OUTPUT_DIR = Path("data/processed/audit")


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Convert occurrence time to datetime.
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True
    )

    # Find unusual records.
    negative_depth = df[df["depth"] < 0].copy()
    low_magnitude = df[df["mag"] < 4].copy()

    # Save records for manual investigation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    negative_depth.to_csv(
        OUTPUT_DIR / "negative_depth_records.csv",
        index=False
    )

    low_magnitude.to_csv(
        OUTPUT_DIR / "below_threshold_records.csv",
        index=False
    )

    print("\n========== NEGATIVE DEPTH RECORDS ==========")
    print(f"Count: {len(negative_depth)}")

    if not negative_depth.empty:
        print(
            negative_depth[
                ["id", "time", "latitude", "longitude",
                 "depth", "mag", "magType", "place"]
            ].to_string(index=False)
        )

    print("\n========== MAGNITUDE BELOW 4 ==========")
    print(f"Count: {len(low_magnitude)}")

    if not low_magnitude.empty:
        print(
            low_magnitude[
                ["id", "time", "depth", "mag",
                 "magType", "place"]
            ].to_string(index=False)
        )

    print("\n========== MAGNITUDE TYPE COUNTS ==========")
    print(
        df["magType"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n========== AUDIT FILES ==========")
    print(f"Saved in: {OUTPUT_DIR.resolve()}")

    print("\nAudit completed. Original dataset unchanged.")


if __name__ == "__main__":
    main()
