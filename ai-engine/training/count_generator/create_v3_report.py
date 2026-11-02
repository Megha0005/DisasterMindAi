
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/count_generator/v3/increment_generator.joblib"
TRAIN_PATH = ROOT / "data/processed/count_generator_train.csv"
VAL_PATH = ROOT / "data/processed/count_generator_validation.csv"

OUTPUT = ROOT / "evaluation/count_generator/v3"
OUTPUT.mkdir(parents=True, exist_ok=True)

model = joblib.load(MODEL_PATH)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)

synthetic = model.sample(len(validation))

columns = [
    "past_1m_count",
    "past_3m_count",
    "past_6m_count"
]

results = {}

for col in columns:

    ks = ks_2samp(
        validation[col].sample(50000, random_state=42),
        synthetic[col].sample(50000, random_state=42)
    ).statistic

    results[col] = {
        "validation_mean": float(validation[col].mean()),
        "synthetic_mean": float(synthetic[col].mean()),
        "validation_std": float(validation[col].std()),
        "synthetic_std": float(synthetic[col].std()),
        "validation_p99": float(validation[col].quantile(.99)),
        "synthetic_p99": float(synthetic[col].quantile(.99)),
        "ks_statistic": float(ks)
    }

real_corr = validation[columns].corr().values
synthetic_corr = synthetic[columns].corr().values

correlation_difference = float(
    np.abs(real_corr - synthetic_corr).sum() / 6
)

report = {
    "model": "IncrementGenerator",
    "version": "V3",
    "training_rows": len(train),
    "validation_rows": len(validation),
    "unique_increment_tuples": len(model.increment_tuples),
    "distribution_results": results,
    "mean_absolute_off_diagonal_correlation_difference":
        correlation_difference,
    "hierarchy_valid": bool(
        (synthetic.past_1m_count <= synthetic.past_3m_count).all()
        and
        (synthetic.past_3m_count <= synthetic.past_6m_count).all()
    ),
    "evaluation_type": "Unseen temporal validation"
}

report_path = OUTPUT / "v3_validation_report.json"

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))
print("\nReport saved:", report_path)
