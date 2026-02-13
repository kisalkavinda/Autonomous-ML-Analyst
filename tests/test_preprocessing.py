import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state import AnalysisState
from src.preprocessing import build_preprocessor
from sklearn.compose import ColumnTransformer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'num1': [1, 2, np.nan, 4],
        'num2': [0.1, 0.2, 0.3, 0.4],
        'cat1': ['a', 'b', 'a', np.nan],
        'target': [0, 1, 0, 1]
    })

def test_build_preprocessor_structure(sample_data):
    state = AnalysisState()
    target_col = 'target'
    
    preprocessor, X, y = build_preprocessor(sample_data, target_col, state)
    
    # Check return types
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Check shape separation
    assert 'target' not in X.columns
    assert X.shape == (4, 3) # num1, num2, cat1
    assert y.name == 'target'

    # Check state logging
    assert len(state.preprocessing_steps) > 0
    log_text = " ".join(state.preprocessing_steps)
    assert "numeric features" in log_text
    assert "categorical features" in log_text

def test_pipeline_fitting(sample_data):
    state = AnalysisState()
    target_col = 'target'
    preprocessor, X, y = build_preprocessor(sample_data, target_col, state)
    
    # Ensure the pipeline interprets data correctly
    X_transformed = preprocessor.fit_transform(X)
    
    # Output shape: 
    # num1 (1), num2 (1) -> 2 numeric
    # cat1: 'a', 'b', nan. Imputed 'most_frequent' (likely 'a'). 
    # OneHot: 'a', 'b' -> 2 columns (sparse=False)
    # Total cols: 2 + 2 = 4
    
    assert X_transformed.shape == (4, 4)
    assert not np.isnan(X_transformed).any()

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
