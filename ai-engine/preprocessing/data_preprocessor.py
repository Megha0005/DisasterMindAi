
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None

    def load_data(self, file_path):
        """Load a CSV dataset."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("The dataset is empty.")

        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def clean_data(self, df, target_column):
        """Remove duplicates and rows with missing target values."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        df = df.copy()

        original_rows = len(df)
        df = df.drop_duplicates()
        print(f"Duplicates removed: {original_rows - len(df)}")

        missing_targets = df[target_column].isna().sum()
        df = df.dropna(subset=[target_column])

        print(f"Rows with missing targets removed: {missing_targets}")

        if df.empty:
            raise ValueError("No rows remain after cleaning.")

        return df

    def split_data(self, df, target_column):
        """Split data before fitting preprocessing transformations."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if X.shape[1] == 0:
            raise ValueError("The dataset has no input features.")

        # Stratify when the target appears to be categorical
        # and every class has enough samples.
        stratify = None

        if y.dtype == "object" or y.dtype.name == "category" or y.dtype == bool:
            if y.value_counts().min() >= 2:
                stratify = y

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

    def build_pipeline(self, X_train):
        """Create numerical and categorical preprocessing pipelines."""
        numerical_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers = []

        if numerical_columns:
            transformers.append(
                ("numerical", numerical_pipeline, numerical_columns)
            )

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        if not transformers:
            raise ValueError("No supported input features were found.")

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

    def process(self, file_path, target_column, output_dir="data/processed"):
        """Run the complete preprocessing workflow."""
        df = self.load_data(file_path)
        df = self.clean_data(df, target_column)

        X_train, X_test, y_train, y_test = self.split_data(
            df,
            target_column
        )

        self.build_pipeline(X_train)

        # Fit only on training data to prevent data leakage.
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        feature_names = self.preprocessor.get_feature_names_out()

        X_train_df = pd.DataFrame(
            X_train_processed,
            columns=feature_names,
            index=X_train.index
        )

        X_test_df = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index
        )

        train_df = X_train_df.copy()
        train_df[target_column] = y_train

        test_df = X_test_df.copy()
        test_df[target_column] = y_test

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_path / "train_processed.csv", index=False)
        test_df.to_csv(output_path / "test_processed.csv", index=False)

        joblib.dump(
            self.preprocessor,
            output_path / "preprocessor.joblib"
        )

        print("\nPreprocessing completed successfully!")
        print(f"Training samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")
        print(f"Processed features: {len(feature_names)}")
        print(f"Output directory: {output_path.resolve()}")

        return train_df, test_df


if __name__ == "__main__":
    processor = DataPreprocessor()

    # Example usage:
    # Replace this with your actual dataset and target column.
    # train_data, test_data = processor.process(
    #     file_path="data/raw/disaster_data.csv",
    #     target_column="risk_level"
    # )
