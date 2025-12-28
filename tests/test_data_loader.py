from pathlib import Path

from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMNS
from src.data_loader import load_dataset


def test_load_dataset_shapes():
    features, targets = load_dataset(Path(DATA_PATH))
    assert list(features.columns) == FEATURE_COLUMNS
    assert list(targets.columns) == TARGET_COLUMNS
    assert len(features) == len(targets) > 0
