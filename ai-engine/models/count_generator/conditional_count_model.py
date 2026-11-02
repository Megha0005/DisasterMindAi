
import numpy as np
import pandas as pd


class ConditionalCountGenerator:

    def __init__(self, min_support=30, random_state=42):
        self.min_support = min_support
        self.rng = np.random.default_rng(random_state)

        self.six_month_dist = None
        self.three_month_global = None
        self.one_month_global = None

        self.three_month_tables = {}
        self.one_month_joint = {}
        self.one_month_3m = {}
        self.one_month_6m = {}

        self.fallback_counts = {}
        self.fitted = False

    @staticmethod
    def _distribution(values):
        counts = pd.Series(values).value_counts().sort_index()
        return (
            counts.index.to_numpy(dtype=np.int64),
            (counts / counts.sum()).to_numpy(dtype=float)
        )

    def fit(self, data):

        cols = [
            "past_1m_count",
            "past_3m_count",
            "past_6m_count"
        ]

        df = data[cols].copy()

        if df.isnull().any().any() or (df < 0).any().any():
            raise ValueError("Invalid count data.")

        if (df.past_1m_count > df.past_3m_count).any():
            raise ValueError("Invalid 1m <= 3m constraint.")

        if (df.past_3m_count > df.past_6m_count).any():
            raise ValueError("Invalid 3m <= 6m constraint.")

        self.six_month_dist = self._distribution(
            df.past_6m_count
        )

        self.three_month_global = self._distribution(
            df.past_3m_count
        )

        self.one_month_global = self._distribution(
            df.past_1m_count
        )

        for six, group in df.groupby("past_6m_count"):
            if len(group) >= self.min_support:
                self.three_month_tables[int(six)] = (
                    self._distribution(group.past_3m_count)
                )

        for (six, three), group in df.groupby(
            ["past_6m_count", "past_3m_count"]
        ):
            if len(group) >= self.min_support:
                self.one_month_joint[(int(six), int(three))] = (
                    self._distribution(group.past_1m_count)
                )

        for three, group in df.groupby("past_3m_count"):
            if len(group) >= self.min_support:
                self.one_month_3m[int(three)] = (
                    self._distribution(group.past_1m_count)
                )

        for six, group in df.groupby("past_6m_count"):
            if len(group) >= self.min_support:
                self.one_month_6m[int(six)] = (
                    self._distribution(group.past_1m_count)
                )

        self.fitted = True
        return self

    def _sample(self, distribution):
        values, probabilities = distribution
        return int(self.rng.choice(values, p=probabilities))

    def sample(self, n_samples):

        if not self.fitted:
            raise RuntimeError("Model is not fitted.")

        self.fallback_counts = {
            "three_conditional": 0,
            "three_global": 0,
            "one_joint": 0,
            "one_3m": 0,
            "one_6m": 0,
            "one_global": 0
        }

        output = []

        for _ in range(n_samples):

            six = self._sample(self.six_month_dist)

            if six in self.three_month_tables:
                dist3 = self.three_month_tables[six]
                self.fallback_counts["three_conditional"] += 1
            else:
                dist3 = self.three_month_global
                self.fallback_counts["three_global"] += 1

            three = min(self._sample(dist3), six)

            key = (six, three)

            if key in self.one_month_joint:
                dist1 = self.one_month_joint[key]
                self.fallback_counts["one_joint"] += 1

            elif three in self.one_month_3m:
                dist1 = self.one_month_3m[three]
                self.fallback_counts["one_3m"] += 1

            elif six in self.one_month_6m:
                dist1 = self.one_month_6m[six]
                self.fallback_counts["one_6m"] += 1

            else:
                dist1 = self.one_month_global
                self.fallback_counts["one_global"] += 1

            one = min(self._sample(dist1), three)

            output.append((one, three, six))

        return pd.DataFrame(
            output,
            columns=[
                "past_1m_count",
                "past_3m_count",
                "past_6m_count"
            ]
        )
