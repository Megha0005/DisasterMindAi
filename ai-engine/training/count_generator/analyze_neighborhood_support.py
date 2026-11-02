
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv",
    usecols=[
        "grid_lat",
        "grid_lon",
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ]
)

df["recent"] = df["past_1m_count"]
df["previous"] = df["past_3m_count"] - df["past_1m_count"]
df["older"] = df["past_6m_count"] - df["past_3m_count"]

df["tuple"] = list(zip(
    df["recent"],
    df["previous"],
    df["older"]
))

df["region_lat"] = (df["grid_lat"] // 6) * 6
df["region_lon"] = (df["grid_lon"] // 6) * 6

def calculate_ess(series):
    probabilities = series.value_counts(normalize=True).values
    return 1.0 / np.sum(probabilities ** 2)

regional = df.groupby(
    ["region_lat", "region_lon"]
).agg(
    observations=("tuple", "size"),
    unique_patterns=("tuple", "nunique"),
    effective_sample_size=("tuple", calculate_ess)
)

print("Total regions:", len(regional))

print("\nEffective sample size:")
print(
    regional["effective_sample_size"]
    .describe(percentiles=[.25, .5, .75, .9, .95])
    .round(2)
)

print("\nESS thresholds:")

for threshold in [5, 10, 20, 30, 50, 100]:
    count = (
        regional["effective_sample_size"] >= threshold
    ).sum()
    print(f"ESS >= {threshold}: {count}/{len(regional)}")

print("\nLowest ESS regions:")
print(
    regional.sort_values("effective_sample_size").head(10)
)

print("\nHighest ESS regions:")
print(
    regional.sort_values(
        "effective_sample_size",
        ascending=False
    ).head(10)
)

output = ROOT / "evaluation/count_generator"
output.mkdir(parents=True, exist_ok=True)

regional.to_csv(output / "regional_effective_sample_size.csv")

print("\nSaved regional ESS analysis.")
