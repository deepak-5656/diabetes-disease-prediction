from src.config import DATA_PATH, TARGET_COLUMNS
from src.data_loader import load_dataset
from src.model_pipeline import build_model


def test_pipeline_train_and_predict():
    features, targets = load_dataset(DATA_PATH)
    model = build_model(n_estimators=10, max_depth=5)
    sample_features = features.head(10)
    sample_targets = targets.head(10)
    model.fit(sample_features, sample_targets)
    preds = model.predict(sample_features)
    assert preds.shape == (len(sample_features), len(TARGET_COLUMNS))
