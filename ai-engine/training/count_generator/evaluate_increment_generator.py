
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp


def main():

    root = Path(__file__).resolve().parents[2]

    model_path = (
        root / "models/count_generator/v3/increment_generator.joblib"
    )

    validation_path = (
        root / "data/processed/count_generator_validation.csv"
    )

    output_path = (
        root / "data/synthetic/count_generator/v3"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading V3 generator...")
    model = joblib.load(model_path)

    print("Loading unseen validation data...")
    real = pd.read_csv(validation_path)

    print("Generating synthetic samples...")
    synthetic = model.sample(len(real))

    real_cols = [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count"
    ]

    print("\n========== DISTRIBUTION COMPARISON ==========")

    for col in real_cols:

        print(f"\n{col}")

        print("Real:")
        print(real[col].describe(
            percentiles=[0.50, 0.90, 0.95, 0.99]
        ).round(4))

        print("Synthetic:")
        print(synthetic[col].describe(
            percentiles=[0.50, 0.90, 0.95, 0.99]
        ).round(4))

        ks = ks_2samp(
            real[col].sample(
                n=min(50000, len(real)),
                random_state=42
            ),
            synthetic[col]
        ).statistic

        print(f"KS statistic: {ks:.5f}")

    print("\n========== CORRELATION COMPARISON ==========")

    real_corr = real[real_cols].corr()
    synthetic_corr = synthetic[real_cols].corr()

    print("\nReal correlation:")
    print(real_corr.round(4))

    print("\nSynthetic correlation:")
    print(synthetic_corr.round(4))

    corr_diff = np.abs(
        real_corr.values - synthetic_corr.values
    ).sum() / 6

    print(
        "\nMean absolute off-diagonal correlation difference:",
        round(float(corr_diff), 5)
    )

    print("\n========== CONSTRAINT VALIDATION ==========")

    print(
        "1m <= 3m:",
        bool((synthetic.past_1m_count <=
              synthetic.past_3m_count).all())
    )

    print(
        "3m <= 6m:",
        bool((synthetic.past_3m_count <=
              synthetic.past_6m_count).all())
    )

    save_path = output_path / "synthetic_counts_v3.csv"
    synthetic.to_csv(save_path, index=False)

    print("\nSynthetic data saved:", save_path)


if __name__ == "__main__":
    main()
