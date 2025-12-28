"""Training entrypoint for the diabetes risk predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import glob

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, DEFAULT_MODEL_PATH, TARGET_COLUMN
from .data_loader import load_dataset
from .model_pipeline import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the diabetes risk model")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to data folder")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Destination for the trained pipeline",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    data_folder = Path(args.data)
    csv_files = list(data_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")
    
    csv_path = csv_files[0]
    print(f"Loading data from dataset...")
    
    features, targets = load_dataset(csv_path, multi_output=True)
    print(f"Dataset shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    if isinstance(targets, pd.DataFrame):
        print(f"Diabetes distribution: {dict(targets['Diabetes_012'].value_counts())}")
        print(f"Obesity distribution: {dict(targets['Obesity'].value_counts())}")
    else:
        print(f"Target distribution: {dict(targets.value_counts())}")
    
    pipeline = build_model(n_estimators=args.estimators, max_depth=args.max_depth, multi_output=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=args.test_size, random_state=42
    )

    print("Training model...")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    # Handle multi-output metrics
    if isinstance(targets, pd.DataFrame):
        # Multi-output case: calculate metrics for each output separately
        from sklearn.metrics import accuracy_score, f1_score as sklearn_f1
        
        diabetes_pred = predictions[:, 0]
        obesity_pred = predictions[:, 1]
        diabetes_true = y_test['Diabetes_012'].values
        obesity_true = y_test['Obesity'].values
        
        diabetes_f1 = float(sklearn_f1(diabetes_true, diabetes_pred, average="weighted", zero_division=0))
        obesity_f1 = float(sklearn_f1(obesity_true, obesity_pred, average="weighted", zero_division=0))
        
        diabetes_acc = float(accuracy_score(diabetes_true, diabetes_pred))
        obesity_acc = float(accuracy_score(obesity_true, obesity_pred))
        
        # Overall metrics
        metrics = {
            "train_size": len(x_train),
            "test_size": len(x_test),
            "f1_weighted": (diabetes_f1 + obesity_f1) / 2,  # Average F1
            "f1_macro": (diabetes_f1 + obesity_f1) / 2,
            "accuracy": (diabetes_acc + obesity_acc) / 2,
            "diabetes_f1": diabetes_f1,
            "obesity_f1": obesity_f1,
        }
        
        # Classification reports
        diabetes_report = classification_report(
            diabetes_true, diabetes_pred, output_dict=True, zero_division=0
        )
        obesity_report = classification_report(
            obesity_true, obesity_pred, output_dict=True, zero_division=0
        )
        
        metrics["diabetes_report"] = diabetes_report
        metrics["obesity_report"] = obesity_report
    else:
        # Single output case
        metrics = {
            "train_size": len(x_train),
            "test_size": len(x_test),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        }
        
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Model saved to: {args.model_path}")

    metrics_path = args.model_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
