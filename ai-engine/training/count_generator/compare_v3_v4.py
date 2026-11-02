
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

# V3 model
v3_model = joblib.load(
    ROOT / "models/count_generator/v3/increment_generator.joblib"
)

# V4 regional joint sampler
v4_regions = joblib.load(
    ROOT / "models/count_generator/v4/joint_sampler_12deg.joblib"
)

v4_global = joblib.load(
    ROOT / "models/count_generator/v4/global_joint_distribution.joblib"
)

N = len(validation)
RADIUS = 12

rng = np.random.default_rng(42)

# Real validation increments
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

# Generate V3 samples
v3_df = v3_model.sample(N)

v3 = np.column_stack([
    v3_df["past_1m_count"].to_numpy(),
    (
        v3_df["past_3m_count"] -
        v3_df["past_1m_count"]
    ).to_numpy(),
    (
        v3_df["past_6m_count"] -
        v3_df["past_3m_count"]
    ).to_numpy()
])

# Generate V4 samples
va = validation.copy()

va["lat_region"] = (
    np.floor(va["grid_lat"] / RADIUS) * RADIUS
)

va["lon_region"] = (
    np.floor(va["grid_lon"] / RADIUS) * RADIUS
)

v4 = np.zeros((N, 3), dtype=np.int64)
fallback_count = 0

for i, row in enumerate(
    va[["lat_region", "lon_region"]].itertuples(index=False)
):
    key = (row.lat_region, row.lon_region)

    distribution = v4_regions.get(key)

    if distribution is None or len(distribution) == 0:
        distribution = v4_global
        fallback_count += 1

    v4[i] = distribution[
        rng.integers(0, len(distribution))
    ]

names = ["recent", "previous", "older"]

results = {}

for index, name in enumerate(names):

    print(f"\n========== {name.upper()} ==========")

    results[name] = {}

    for model_name, generated in [
        ("V3", v3),
        ("V4", v4)
    ]:

        ks = ks_2samp(
            real[:, index],
            generated[:, index]
        ).statistic

        mean_error = abs(
            real[:, index].mean() -
            generated[:, index].mean()
        )

        std_error = abs(
            real[:, index].std() -
            generated[:, index].std()
        )

        tail_rate_real = np.mean(real[:, index] >= 10)
        tail_rate_generated = np.mean(
            generated[:, index] >= 10
        )

        print(
            f"{model_name}: "
            f"KS={ks:.5f}, "
            f"Mean error={mean_error:.5f}, "
            f"Std error={std_error:.5f}, "
            f"Tail >=10 real={tail_rate_real:.5f}, "
            f"generated={tail_rate_generated:.5f}"
        )

        results[name][model_name] = {
            "ks_statistic": float(ks),
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "real_tail_rate_ge_10": float(tail_rate_real),
            "generated_tail_rate_ge_10": float(
                tail_rate_generated
            )
        }

# Correlation comparison
real_corr = np.corrcoef(real, rowvar=False)
v3_corr = np.corrcoef(v3, rowvar=False)
v4_corr = np.corrcoef(v4, rowvar=False)

upper = np.triu_indices(3, k=1)

v3_corr_error = np.abs(
    real_corr[upper] - v3_corr[upper]
).mean()

v4_corr_error = np.abs(
    real_corr[upper] - v4_corr[upper]
).mean()

print("\n========== CORRELATION COMPARISON ==========")
print("V3 correlation error:", round(v3_corr_error, 5))
print("V4 correlation error:", round(v4_corr_error, 5))

results["correlation"] = {
    "real": real_corr.tolist(),
    "V3": v3_corr.tolist(),
    "V4": v4_corr.tolist(),
    "V3_error": float(v3_corr_error),
    "V4_error": float(v4_corr_error)
}

# Hierarchy checks
for model_name, generated in [("V3", v3), ("V4", v4)]:
    valid = bool(
        (generated[:, 0] >= 0).all()
        and (generated[:, 1] >= 0).all()
        and (generated[:, 2] >= 0).all()
    )

    print(f"{model_name} hierarchy valid:", valid)

results["V4_fallback_samples"] = fallback_count

output_dir = ROOT / "evaluation/count_generator/comparison"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "v3_v4_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nV3 vs V4 comparison saved successfully.")
