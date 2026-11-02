
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

keys = ["grid_lat", "grid_lon"]

train_stats = (
    train.groupby(keys)["past_6m_count"]
    .mean()
    .rename("train_mean_6m")
)

validation_stats = (
    validation.groupby(keys)["past_6m_count"]
    .mean()
    .rename("validation_mean_6m")
)

comparison = pd.concat(
    [train_stats, validation_stats],
    axis=1
).dropna()

comparison["absolute_change"] = abs(
    comparison["validation_mean_6m"]
    - comparison["train_mean_6m"]
)

comparison["relative_change"] = (
    comparison["absolute_change"]
    / (comparison["train_mean_6m"] + 0.01)
)

print("Geographic cells:", len(comparison))

print("\nSpearman rank correlation:")
print(
    spearmanr(
        comparison["train_mean_6m"],
        comparison["validation_mean_6m"]
    )
)

print("\nAbsolute change:")
print(comparison["absolute_change"].describe().round(4))

print("\nRelative change:")
print(comparison["relative_change"].describe(
    percentiles=[.5, .9, .95, .99]
).round(4))

print("\nLargest absolute changes:")
print(
    comparison.sort_values(
        "absolute_change",
        ascending=False
    ).head(15).round(4)
)

output = ROOT / "evaluation/count_generator"
output.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output / "geographic_shift.csv"
)

print("\nSaved: evaluation/count_generator/geographic_shift.csv")
