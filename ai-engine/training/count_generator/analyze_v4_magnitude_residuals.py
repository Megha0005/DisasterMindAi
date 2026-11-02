
import pandas as pd
import numpy as np
import joblib

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

features = ["grid_lat", "grid_lon"]
targets = ["recent", "previous", "older"]

model_dir = ROOT / "models/count_generator/v4"

for target in targets:

    print(f"\n========== {target.upper()} ==========")

    model = joblib.load(
        model_dir / f"{target}_magnitude_model.joblib"
    )

    data = validation[validation[target] > 0].copy()

    actual = data[target].to_numpy()
    predicted = np.expm1(
        model.predict(data[features])
    )

    predicted = np.maximum(predicted, 0)

    data["actual"] = actual
    data["predicted"] = predicted
    data["absolute_error"] = np.abs(actual - predicted)

    data["count_group"] = pd.cut(
        data["actual"],
        bins=[0, 1, 2, 5, 10, 25, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-25", "26+"]
    )

    grouped = data.groupby(
        "count_group",
        observed=False
    ).agg(
        samples=("actual", "size"),
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        mean_absolute_error=("absolute_error", "mean")
    )

    print(grouped.to_string())

    print("\nOverall prediction quantiles:")
    print(
        pd.Series(predicted).quantile(
            [0.50, 0.90, 0.95, 0.99, 1.00]
        )
    )

