
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

sampler_data = joblib.load(
    ROOT / "models/count_generator/v4/joint_sampler_12deg.joblib"
)

global_data = joblib.load(
    ROOT / "models/count_generator/v4/global_joint_distribution.joblib"
)

RADIUS = 12
rng = np.random.default_rng(42)

validation["lat_region"] = (
    np.floor(validation["grid_lat"] / RADIUS) * RADIUS
)

validation["lon_region"] = (
    np.floor(validation["grid_lon"] / RADIUS) * RADIUS
)

generated = np.zeros((len(validation), 3), dtype=np.int64)
fallback_count = 0

for i, row in enumerate(
    validation[["lat_region", "lon_region"]].itertuples(index=False)
):
    key = (row.lat_region, row.lon_region)

    distribution = sampler_data.get(key)

    if distribution is None or len(distribution) == 0:
        distribution = global_data
        fallback_count += 1

    selected = rng.integers(0, len(distribution))
    generated[i] = distribution[selected]

real = np.column_stack([
    validation["past_1m_count"].to_numpy(),
    (
        validation["past_3m_count"] -
        validation["past_1m_count"]
    ).to_numpy(),
    (
        validation["past_6m_count"] -
        validation["past_3m_count"]
    ).to_numpy()
])

names = ["recent", "previous", "older"]

report = {}

print("Fallback samples:", fallback_count)

for i, name in enumerate(names):

    print(f"\n========== {name.upper()} ==========")

    ks = ks_2samp(real[:, i], generated[:, i]).statistic

    print("Real mean:", round(real[:, i].mean(), 4))
    print("Synthetic mean:", round(generated[:, i].mean(), 4))
    print("Real std:", round(real[:, i].std(), 4))
    print("Synthetic std:", round(generated[:, i].std(), 4))
    print("KS statistic:", round(ks, 5))

    report[name] = {
        "real_mean": float(real[:, i].mean()),
        "synthetic_mean": float(generated[:, i].mean()),
        "real_std": float(real[:, i].std()),
        "synthetic_std": float(generated[:, i].std()),
        "ks_statistic": float(ks)
    }

real_corr = np.corrcoef(real, rowvar=False)
generated_corr = np.corrcoef(generated, rowvar=False)

correlation_error = np.abs(real_corr - generated_corr)

print("\nReal correlation matrix:")
print(real_corr)

print("\nSynthetic correlation matrix:")
print(generated_corr)

print(
    "\nMean absolute correlation difference:",
    round(
        correlation_error[np.triu_indices(3, k=1)].mean(),
        5
    )
)

hierarchy_ok = bool(
    (generated[:, 0] <= generated[:, 0] + generated[:, 1]).all()
    and
    (
        generated[:, 0] + generated[:, 1]
        <= generated.sum(axis=1)
    ).all()
)

print("Hierarchy preserved:", hierarchy_ok)

report["fallback_samples"] = fallback_count
report["real_correlation"] = real_corr.tolist()
report["synthetic_correlation"] = generated_corr.tolist()
report["mean_absolute_correlation_difference"] = float(
    correlation_error[np.triu_indices(3, k=1)].mean()
)
report["hierarchy_preserved"] = hierarchy_ok

output_dir = ROOT / "evaluation/count_generator/v4"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "joint_sampler_validation.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nJoint sampler validation saved.")

