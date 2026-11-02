
import pandas as pd
import numpy as np
import json

from pathlib import Path
from scipy.stats import ks_2samp

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

targets = ["recent", "previous", "older"]
radii = [8, 12]

rng = np.random.default_rng(42)
results = {}

for target in targets:
    print(f"\n========== {target.upper()} ==========")

    real = validation[target].to_numpy()
    results[target] = {}

    for radius in radii:
        print(f"\nRadius: {radius} degrees")

        tr = train.copy()
        va = validation.copy()

        for df in [tr, va]:
            df["lat_region"] = (
                np.floor(df["grid_lat"] / radius) * radius
            )
            df["lon_region"] = (
                np.floor(df["grid_lon"] / radius) * radius
            )

        regional_distributions = {}

        for key, group in tr.groupby(
            ["lat_region", "lon_region"]
        ):
            values = group[target].to_numpy()
            regional_distributions[key] = values

        generated = np.zeros(len(va), dtype=int)
        fallback_count = 0

        region_keys = list(
            zip(va["lat_region"], va["lon_region"])
        )

        for i, key in enumerate(region_keys):
            distribution = regional_distributions.get(key)

            if distribution is None or len(distribution) == 0:
                distribution = tr[target].to_numpy()
                fallback_count += 1

            generated[i] = rng.choice(distribution)

        ks = ks_2samp(real, generated).statistic

        print("Real mean:", round(real.mean(), 4))
        print("Synthetic mean:", round(generated.mean(), 4))
        print("Real std:", round(real.std(), 4))
        print("Synthetic std:", round(generated.std(), 4))
        print("KS statistic:", round(ks, 5))
        print("Fallback samples:", fallback_count)

        print("Real quantiles:")
        print(np.quantile(real, [0.50, 0.90, 0.95, 0.99]))

        print("Synthetic quantiles:")
        print(np.quantile(generated, [0.50, 0.90, 0.95, 0.99]))

        results[target][str(radius)] = {
            "real_mean": float(real.mean()),
            "synthetic_mean": float(generated.mean()),
            "real_std": float(real.std()),
            "synthetic_std": float(generated.std()),
            "ks_statistic": float(ks),
            "fallback_samples": int(fallback_count),
            "real_quantiles": np.quantile(
                real, [0.50, 0.90, 0.95, 0.99]
            ).tolist(),
            "synthetic_quantiles": np.quantile(
                generated, [0.50, 0.90, 0.95, 0.99]
            ).tolist()
        }

output = ROOT / "evaluation/count_generator/v4"
output.mkdir(parents=True, exist_ok=True)

with open(output / "pooling_candidate_report.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nPooling candidate evaluation saved.")
