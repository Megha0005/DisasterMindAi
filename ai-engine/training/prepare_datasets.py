
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/earthquake_spatiotemporal_features.csv"
)

OUTPUT_DIR = Path("data/processed/splits")


def main():
    print("Loading engineered earthquake features...")

    df = pd.read_csv(INPUT_FILE)

    # Convert month to a comparable monthly period.
    df["month_period"] = pd.PeriodIndex(
        df["month_period"],
        freq="M"
    )

    # Chronological split.
    train = df[
        (df["month_period"] >= pd.Period("2000-07", freq="M"))
        & (df["month_period"] <= pd.Period("2019-12", freq="M"))
    ].copy()

    validation = df[
        (df["month_period"] >= pd.Period("2020-01", freq="M"))
        & (df["month_period"] <= pd.Period("2022-12", freq="M"))
    ].copy()

    test = df[
        (df["month_period"] >= pd.Period("2023-01", freq="M"))
        & (df["month_period"] <= pd.Period("2025-09", freq="M"))
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\n========== CHRONOLOGICAL SPLIT REPORT ==========")

    for name, dataset in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        print(f"\n{name}")
        print(f"Rows: {len(dataset):,}")

        if not dataset.empty:
            print(
                "Period:",
                dataset["month_period"].min(),
                "to",
                dataset["month_period"].max()
            )

            print(
                "Positive rate:",
                round(dataset["target_next_month"].mean() * 100, 2),
                "%"
            )

    print("\n========== OUTPUT FILES ==========")
    print(f"Saved in: {OUTPUT_DIR.resolve()}")

    print("\nChronological splitting completed!")


if __name__ == "__main__":
    main()
