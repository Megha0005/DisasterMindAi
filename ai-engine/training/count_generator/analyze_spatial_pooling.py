
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
df["previous_2m"] = df["past_3m_count"] - df["past_1m_count"]
df["older_3m"] = df["past_6m_count"] - df["past_3m_count"]

regional = df.groupby(
    ["grid_lat", "grid_lon"]
).agg(
    observations=("recent_1m", "size"),
    active_months=("past_6m_count", lambda x: (x > 0).sum()),
    mean_activity=("past_6m_count", "mean")
).reset_index()

latitudes = sorted(regional.grid_lat.unique())
longitudes = sorted(regional.grid_lon.unique())

print("Latitude grid spacing:", sorted(set(
    round(b - a, 4) for a, b in zip(latitudes, latitudes[1:])
))[:10])

print("Longitude grid spacing:", sorted(set(
    round(b - a, 4) for a, b in zip(longitudes, longitudes[1:])
))[:10])

print("\nActivity distribution across cells:")
print(regional["active_months"].describe(
    percentiles=[.25, .50, .75, .90, .95, .99]
).round(2))

print("\nCells with active months:")
for threshold in [1, 5, 10, 20, 50, 100]:
    print(
        f"At least {threshold}:",
        int((regional.active_months >= threshold).sum())
    )

print("\nSaved regional activity summary.")

output = ROOT / "evaluation/count_generator"
regional.to_csv(output / "spatial_pooling_support.csv", index=False)
