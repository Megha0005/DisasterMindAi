
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/usgs_earthquakes_cleaned.csv"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR / "earthquake_spatiotemporal_features.csv"
)

GRID_SIZE = 2
HISTORY_MONTHS = 6

# December 2025 is excluded as a target month.
# October 2025 is the last prediction month.
FIRST_FEATURE_MONTH = "2000-01"
LAST_FEATURE_MONTH = "2025-10"


def build_features():
    print("Loading cleaned earthquake data...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=["time", "latitude", "longitude", "mag"]
    ).copy()

    # Assign geographic grid cells.
    df["grid_lat"] = (
        np.floor(df["latitude"] / GRID_SIZE) * GRID_SIZE
    )

    df["grid_lon"] = (
        np.floor(df["longitude"] / GRID_SIZE) * GRID_SIZE
    )

    df["month_period"] = (
        df["time"]
        .dt.tz_convert(None)
        .dt.to_period("M")
    )

    print("Aggregating monthly activity...")

    monthly = (
        df.groupby(
            ["grid_lat", "grid_lon", "month_period"],
            observed=True
        )
        .agg(
            earthquake_count=("id", "count"),
            average_magnitude=("mag", "mean"),
            maximum_magnitude=("mag", "max"),
            average_depth=("depth", "mean")
        )
        .reset_index()
    )

    # Define a common calendar for every observed cell.
    calendar = pd.period_range(
        FIRST_FEATURE_MONTH,
        LAST_FEATURE_MONTH,
        freq="M"
    )

    cells = monthly[
        ["grid_lat", "grid_lon"]
    ].drop_duplicates()

    print(f"Observed geographic cells: {len(cells):,}")

    # Create a complete cell-month panel.
    cells["_key"] = 1

    calendar_df = pd.DataFrame({
        "month_period": calendar,
        "_key": 1
    })

    panel = cells.merge(
        calendar_df,
        on="_key"
    ).drop(columns="_key")

    panel = panel.merge(
        monthly,
        on=["grid_lat", "grid_lon", "month_period"],
        how="left"
    )

    # Zero-event months receive zero event counts.
    panel["earthquake_count"] = (
        panel["earthquake_count"].fillna(0)
    )

    # Missing magnitudes and depths remain missing in the
    # event-level aggregates. They are not real zero values.
    panel = panel.sort_values(
        ["grid_lat", "grid_lon", "month_period"]
    ).reset_index(drop=True)

    grouped = panel.groupby(
        ["grid_lat", "grid_lon"],
        sort=False
    )["earthquake_count"]

    # Historical activity only: never use current-month counts.
    for window in [1, 3, 6]:
        panel[f"past_{window}m_count"] = grouped.transform(
            lambda series: (
                series.shift(1)
                .rolling(window, min_periods=window)
                .sum()
            )
        )

    # Label from the following calendar month.
    panel["next_month_count"] = grouped.transform(
        lambda series: series.shift(-1)
    )

    panel["target_next_month"] = (
        panel["next_month_count"] > 0
    ).astype("int8")

    # Remove the first six months, which lack sufficient history,
    # and the final month, which has no next-month label.
    panel = panel.dropna(
        subset=[
            "past_1m_count",
            "past_3m_count",
            "past_6m_count",
            "next_month_count"
        ]
    ).copy()

    panel["year"] = panel["month_period"].dt.year
    panel["month"] = panel["month_period"].dt.month

    panel["month_period"] = (
        panel["month_period"].astype(str)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_FILE, index=False)

    print("\n========== REVISED FEATURE REPORT ==========")
    print(f"Final rows: {len(panel):,}")

    print(
        "Geographic cells:",
        panel[["grid_lat", "grid_lon"]]
        .drop_duplicates().shape[0]
    )

    print("First feature month:", panel["month_period"].min())
    print("Last feature month:", panel["month_period"].max())

    print("\nTarget distribution:")
    print(panel["target_next_month"].value_counts().to_string())

    print("\nTarget percentages:")
    print(
        panel["target_next_month"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_FILE.resolve()}")
    print("Revised feature engineering completed!")


if __name__ == "__main__":
    build_features()
