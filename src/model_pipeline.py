"""Model pipeline utilities."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_model(n_estimators: int = 300, max_depth: int | None = None, multi_output: bool = True) -> Pipeline:

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    if multi_output:
        classifier = MultiOutputClassifier(classifier)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
    return pipeline
