
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "count_generator_validation.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "count_generator",
    "conditional_count_generator.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "synthetic",
    "count_generator", "synthetic_counts.csv"
)

def main():
    print("Loading model and real data...")

    model = joblib.load(MODEL_PATH)
    real = pd.read_csv(DATA_PATH)

    print("Generating synthetic samples...")
    synthetic = model.sample(50000)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    synthetic.to_csv(OUTPUT_PATH, index=False)

    cols = [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ]

    print("\n========== DISTRIBUTION COMPARISON ==========")

    for col in cols:
        print(f"\n{col}")

        print("Real:")
        print(real[col].describe(
            percentiles=[.5, .9, .95, .99]
        ))

        print("Synthetic:")
        print(synthetic[col].describe(
            percentiles=[.5, .9, .95, .99]
        ))

        ks = __import__(
            "scipy.stats", fromlist=["ks_2samp"]
        ).ks_2samp(
            real[col].sample(
                50000, random_state=42
            ),
            synthetic[col]
        )

        print("KS statistic:", round(ks.statistic, 5))

    print("\n========== CORRELATION COMPARISON ==========")

    real_corr = real[cols].corr()
    synthetic_corr = synthetic[cols].corr()

    print("\nReal correlation:")
    print(real_corr.round(4))

    print("\nSynthetic correlation:")
    print(synthetic_corr.round(4))

    difference = np.abs(
        real_corr.values - synthetic_corr.values
    )

    print(
        "\nMean absolute correlation difference:",
        round(difference.mean(), 5)
    )

    print("\n========== CONSTRAINT VALIDATION ==========")

    print(
        "1m <= 3m:",
        bool(
            (synthetic.past_1m_count <=
             synthetic.past_3m_count).all()
        )
    )

    print(
        "3m <= 6m:",
        bool(
            (synthetic.past_3m_count <=
             synthetic.past_6m_count).all()
        )
    )

    print("\nSynthetic dataset saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
