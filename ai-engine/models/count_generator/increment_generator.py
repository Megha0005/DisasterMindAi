
import numpy as np
import pandas as pd


class IncrementGenerator:

    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.increment_tuples = None
        self.probabilities = None
        self.fitted = False

    def fit(self, data):

        required = [
            "past_1m_count",
            "past_3m_count",
            "past_6m_count"
        ]

        df = data[required].copy()

        if df.isnull().any().any():
            raise ValueError("Missing count values.")

        df["recent_1m"] = df["past_1m_count"]
        df["previous_2m"] = (
            df["past_3m_count"] - df["past_1m_count"]
        )
        df["older_3m"] = (
            df["past_6m_count"] - df["past_3m_count"]
        )

        increments = df[
            ["recent_1m", "previous_2m", "older_3m"]
        ]

        if (increments < 0).any().any():
            raise ValueError("Negative increments detected.")

        grouped = increments.value_counts().sort_index()

        self.increment_tuples = np.asarray(
            grouped.index.tolist(),
            dtype=np.int64
        )

        self.probabilities = (
            grouped.to_numpy(dtype=float) / grouped.sum()
        )

        self.fitted = True
        return self

    def sample(self, n_samples):

        if not self.fitted:
            raise RuntimeError("Generator is not fitted.")

        indices = self.rng.choice(
            len(self.increment_tuples),
            size=n_samples,
            replace=True,
            p=self.probabilities
        )

        increments = self.increment_tuples[indices]

        recent = increments[:, 0]
        previous = increments[:, 1]
        older = increments[:, 2]

        return pd.DataFrame({
            "past_1m_count": recent,
            "past_3m_count": recent + previous,
            "past_6m_count": recent + previous + older
        })
