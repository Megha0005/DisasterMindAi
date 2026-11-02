
import os
import json
import joblib
import pandas as pd

from models.count_generator.conditional_count_model import (
    ConditionalCountGenerator
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "count_generator_train.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "count_generator"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "conditional_count_generator.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR, "metadata.json"
)

MIN_SUPPORT = 30
RANDOM_STATE = 42

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading training data...")
    data = pd.read_csv(DATA_PATH)

    print("Training Conditional Count Generator...")

    model = ConditionalCountGenerator(
        min_support=MIN_SUPPORT,
        random_state=RANDOM_STATE
    )

    model.fit(data)

    print("\nGenerating a sample for verification...")
    samples = model.sample(10000)

    # Validate generated count constraints
    assert (samples >= 0).all().all()
    assert (
        samples.past_1m_count <= samples.past_3m_count
    ).all()
    assert (
        samples.past_3m_count <= samples.past_6m_count
    ).all()

    print("Generated samples:", len(samples))
    print("Count hierarchy validation: PASSED")

    print("\nSaving model...")
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model": "ConditionalCountGenerator",
        "minimum_support": MIN_SUPPORT,
        "random_state": RANDOM_STATE,
        "training_rows": len(data),
        "conditional_3m_groups": len(
            model.three_month_tables
        ),
        "conditional_1m_groups": len(
            model.one_month_joint
        ),
        "verification_samples": len(samples)
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining completed successfully!")
    print("Model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)
    print("\nMetadata:")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    main()
