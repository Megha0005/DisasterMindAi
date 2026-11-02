
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

targets = ["recent", "previous", "older"]

report = {}

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    train_positive = train.loc[train[target] > 0, target]
    val_positive = validation.loc[validation[target] > 0, target]

    print("Training positive samples:", len(train_positive))
    print("Validation positive samples:", len(val_positive))

    train_quantiles = train_positive.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )

    val_quantiles = val_positive.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )

    print("\nTraining quantiles:")
    print(train_quantiles)

    print("\nValidation quantiles:")
    print(val_quantiles)

    print("\nTail probabilities:")
    thresholds = [5, 10, 25, 50, 100]

    for threshold in thresholds:
        train_rate = (train_positive >= threshold).mean()
        val_rate = (val_positive >= threshold).mean()

        print(
            f"Count >= {threshold}: "
            f"train={train_rate:.5f}, "
            f"validation={val_rate:.5f}"
        )

    report[target] = {
        "training_quantiles": train_quantiles.to_dict(),
        "validation_quantiles": val_quantiles.to_dict()
    }

output = ROOT / "evaluation/count_generator/v4"
output.mkdir(parents=True, exist_ok=True)

with open(output / "count_tail_analysis.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nTail analysis saved successfully.")
