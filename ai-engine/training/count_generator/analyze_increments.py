
import pandas as pd

files = {
    "TRAIN": "data/processed/count_generator_train.csv",
    "VALIDATION": "data/processed/count_generator_validation.csv"
}

for name, path in files.items():

    df = pd.read_csv(path, usecols=[
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ])

    df["recent_1m"] = df["past_1m_count"]
    df["previous_2m"] = (
        df["past_3m_count"] - df["past_1m_count"]
    )
    df["older_3m"] = (
        df["past_6m_count"] - df["past_3m_count"]
    )

    increments = [
        "recent_1m",
        "previous_2m",
        "older_3m"
    ]

    print(f"\n========== {name} ==========")

    print("\nDistribution:")
    print(
        df[increments]
        .describe(percentiles=[0.50, 0.90, 0.95, 0.99])
        .round(4)
    )

    print("\nZero rates:")
    print(df[increments].eq(0).mean().round(4))

    print("\nCorrelation:")
    print(df[increments].corr().round(4))

    print(
        "\nNegative increments:",
        int((df[increments] < 0).sum().sum())
    )
