
import pandas as pd
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

df["recent_1m"] = df["past_1m_count"]
df["previous_2m"] = (
    df["past_3m_count"] - df["past_1m_count"]
)
df["older_3m"] = (
    df["past_6m_count"] - df["past_3m_count"]
)

increment_cols = [
    "recent_1m",
    "previous_2m",
    "older_3m"
]

df["increment_tuple"] = list(
    zip(*(df[col] for col in increment_cols))
)

regional = df.groupby(
    ["grid_lat", "grid_lon"]
).agg(
    observations=("recent_1m", "size"),
    unique_tuples=("increment_tuple", "nunique"),
    mean_recent=("recent_1m", "mean"),
    mean_previous=("previous_2m", "mean"),
    mean_older=("older_3m", "mean")
)

print("Total regions:", len(regional))

print("\nObservations per region:")
print(regional["observations"].describe())

print("\nActual unique increment tuples per region:")
print(regional["unique_tuples"].describe())

print("\nRegional tuple support thresholds:")

for threshold in [5, 10, 20, 30, 50, 100]:
    count = (regional["unique_tuples"] >= threshold).sum()
    print(f"At least {threshold}: {count}/{len(regional)}")

print("\nRegions with fewest unique tuples:")
print(regional.sort_values("unique_tuples").head(15))

output = ROOT / "evaluation/count_generator"
output.mkdir(parents=True, exist_ok=True)

regional.to_csv(output / "regional_support.csv")

print("\nSaved:", output / "regional_support.csv")
