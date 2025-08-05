import glob
import pickle

import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# your evaluate_model helper
def evaluate_model(pkl_path, X, y):
    try:
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return acc, cm
    except Exception as e:
        pytest.skip(f"Skipping {pkl_path}: {e}")

@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("data/iris.csv")
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    return X, y

# 1) Parametrized test for each model file found
@pytest.mark.parametrize("model_path", sorted(glob.glob("models/decision_tree_model_*.pkl")))
def test_model_accuracy_threshold(model_path, data):
    X, y = data
    acc, cm = evaluate_model(model_path, X, y)
    # sanity-check shape
    n = len(y)
    assert cm.shape == (3, 3), f"{model_path}: expected 3×3 CM, got {cm.shape}"
    assert cm.sum() == n,   f"{model_path}: confusion matrix sums to {cm.sum()} ≠ {n}"
    # accuracy threshold
    assert acc >= 0.7, f"{model_path} accuracy too low: {acc:.3f}"

# 2) Test that versions strictly improve
def test_version_improvement(data):
    X, y = data
    # collect accuracies in version order
    accs = []
    files = sorted(glob.glob("models/decision_tree_model_*.pkl"))
    for path in files:
        acc, _ = evaluate_model(path, X, y)
        accs.append(acc)
    # ensure accs is increasing
    for earlier, later, path_earlier, path_later in zip(accs, accs[1:], files, files[1:]):
        assert later >= earlier, (
            f"Accuracy did not improve from {path_earlier} ({earlier:.3f}) "
            f"to {path_later} ({later:.3f})"
        )

