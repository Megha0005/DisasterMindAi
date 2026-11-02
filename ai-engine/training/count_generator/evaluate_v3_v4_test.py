
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]

test_path = ROOT / "data/processed/splits/test.csv"

test = pd.read_csv(test_path)

required = [
    "grid_lat", "grid_lon",
    "past_1m_count", "past_3m_count", "past_6m_count"
]

missing = set(required) - set(test.columns)
if missing:
    raise ValueError(f"Missing test columns: {missing}")

test = test[required].copy()

test["recent"] = test["past_1m_count"]
test["previous"] = (
    test["past_3m_count"] - test["past_1m_count"]
)
test["older"] = (
    test["past_6m_count"] - test["past_3m_count"]
)

real = test[
    ["recent", "previous", "older"]
].to_numpy(dtype=np.int64)

if (real < 0).any():
    raise ValueError("Negative increments in test data.")

# Load frozen generators
v3_model = joblib.load(
    ROOT / "models/count_generator/v3/increment_generator.joblib"
)

v4_regions = joblib.load(
    ROOT / "models/count_generator/v4/joint_sampler_12deg.joblib"
)

v4_global = joblib.load(
    ROOT / "models/count_generator/v4/global_joint_distribution.joblib"
)

N = len(test)
RADIUS = 12
rng = np.random.default_rng(2026)

# V3 global empirical generation
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

# V4 geographic joint generation
test["lat_region"] = (
    np.floor(test["grid_lat"] / RADIUS) * RADIUS
)
test["lon_region"] = (
    np.floor(test["grid_lon"] / RADIUS) * RADIUS
)

v4 = np.zeros((N, 3), dtype=np.int64)
fallback_count = 0

for i, row in enumerate(
    test[["lat_region", "lon_region"]].itertuples(index=False)
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

print("Test observations:", N)
print("Fallback samples:", fallback_count)

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

        real_tail = np.mean(real[:, index] >= 10)
        generated_tail = np.mean(generated[:, index] >= 10)

        print(
            f"{model_name}: "
            f"KS={ks:.5f}, "
            f"Mean error={mean_error:.5f}, "
            f"Std error={std_error:.5f}, "
            f"Tail real={real_tail:.5f}, "
            f"Tail generated={generated_tail:.5f}"
        )

        results[name][model_name] = {
            "ks": float(ks),
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "real_tail_rate_ge_10": float(real_tail),
            "generated_tail_rate_ge_10": float(generated_tail)
        }

# Joint correlation evaluation
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

print("\n========== CORRELATION ==========")
print("Real:\n", real_corr)
print("V3:\n", v3_corr)
print("V4:\n", v4_corr)
print("V3 correlation error:", round(v3_corr_error, 5))
print("V4 correlation error:", round(v4_corr_error, 5))

results["correlation"] = {
    "real": real_corr.tolist(),
    "V3": v3_corr.tolist(),
    "V4": v4_corr.tolist(),
    "V3_error": float(v3_corr_error),
    "V4_error": float(v4_corr_error)
}

for model_name, generated in [("V3", v3), ("V4", v4)]:
    hierarchy_ok = bool(
        (generated >= 0).all()
        and (generated[:, 0] <= generated[:, 0] + generated[:, 1]).all()
        and (
            generated[:, 0] + generated[:, 1]
            <= generated.sum(axis=1)
        ).all()
    )

    print(f"{model_name} hierarchy valid:", hierarchy_ok)
    results[f"{model_name}_hierarchy_valid"] = hierarchy_ok

results["test_rows"] = N
results["fallback_samples"] = fallback_count
results["test_period"] = "January 2023 - September 2025"

output_dir = ROOT / "evaluation/count_generator/final"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "v3_v4_untouched_test_report.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nUntouched test evaluation saved successfully.")
