import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.datasets import make_classification, make_regression
from sklearn.compose import make_column_transformer

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.model_trainer import run_experiment

@pytest.fixture
def config():
    # Set min_samples_for_split high enough to force CV if needed, or low to force split.
    # Default is 50.
    return AnalysisConfig(min_samples_for_split=50)

@pytest.fixture
def state():
    return AnalysisState()

@pytest.fixture
def dummy_preprocessor():
    # A dummy preprocessor that just passes everything through.
    # We need to assume the input data is already numeric for the models to work directly if we use passthrough,
    # or ensure the models can handle it.
    # The make_classification/regression returns numeric data.
    return make_column_transformer(
        ("passthrough", slice(None)),
        remainder="drop"
    )

def test_run_experiment_classification(config, state, dummy_preprocessor):
    # Generate classification data
    X_np, y_np = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X_np, columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(y_np, name='target')
    
    # Run experiment
    # 100 samples > 50 (default min_samples_for_split), so usage of train_test_split
    pipeline = run_experiment(X, y, dummy_preprocessor, config, state)
    
    # Assertions
    assert state.selected_model is not None
    assert isinstance(state.selected_model, str)
    assert len(state.model_scores) == 3 # RF, GB, Logistic
    
    # Check if F1-Macro is present
    for model_name, scores in state.model_scores.items():
        assert 'F1-Macro' in scores
        assert isinstance(scores['F1-Macro'], float)
        assert 0 <= scores['F1-Macro'] <= 1

    # Check pipeline is fitted (predict should work)
    assert hasattr(pipeline, "predict")
    pipeline.predict(X.iloc[:5])

def test_run_experiment_regression(config, state, dummy_preprocessor):
    # Generate regression data
    X_np, y_np = make_regression(n_samples=40, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X_np, columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(y_np, name='target')
    
    # Run experiment
    # 40 samples < 50, so usage of CV
    pipeline = run_experiment(X, y, dummy_preprocessor, config, state)
    
    # Assertions
    assert state.selected_model is not None
    assert len(state.model_scores) == 3 # RF, GB, Linear
    
    # Check if MAE is present
    for model_name, scores in state.model_scores.items():
        assert 'MAE' in scores
        assert isinstance(scores['MAE'], float)
        assert scores['MAE'] >= 0

    # Check pipeline is fitted
    pipeline.predict(X.iloc[:5])

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
