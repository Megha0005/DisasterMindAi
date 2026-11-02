
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

for df in [train, validation]:
    df["recent"] = df["past_1m_count"]
    df["previous"] = df["past_3m_count"] - df["past_1m_count"]
    df["older"] = df["past_6m_count"] - df["past_3m_count"]

report = {}

for period, df in [
    ("train", train),
    ("validation", validation)
]:

    print(f"\n========== {period.upper()} ==========")

    for target in ["recent", "previous", "older"]:

        print(f"\n--- {target.upper()} ---")

        positive = df[df[target] > 0].copy()

        for threshold in [10, 25, 50, 100]:

            tail = positive[positive[target] >= threshold]

            if len(tail) == 0:
                continue

            cell_counts = tail.groupby(
                ["grid_lat", "grid_lon"]
            ).size()

            total_cells = df[
                ["grid_lat", "grid_lon"]
            ].drop_duplicates().shape[0]

            top_10_share = (
                cell_counts.sort_values(ascending=False)
                .head(10)
                .sum() / len(tail)
            )

            print(
                f"Count >= {threshold}: "
                f"samples={len(tail)}, "
                f"active_cells={len(cell_counts)}, "
                f"top10_share={top_10_share:.4f}"
            )

            report[f"{period}_{target}_{threshold}"] = {
                "samples": int(len(tail)),
                "active_cells": int(len(cell_counts)),
                "top10_cell_share": float(top_10_share)
            }

output = ROOT / "evaluation/count_generator/v4"
output.mkdir(parents=True, exist_ok=True)

with open(output / "tail_geography_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nGeographic tail analysis saved.")

