import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import InsufficientDataError, TargetConstantError, ExcessiveMissingDataError
from src.data_validator import validate_dataset

@pytest.fixture
def config():
    return AnalysisConfig(
        min_samples_absolute=5,
        max_missing_percentage=0.2, # 20%
        max_cardinality_ratio=0.5   # 50%
    )

@pytest.fixture
def state():
    return AnalysisState()

def test_insufficient_data_error(config, state):
    """Test that InsufficientDataError is raised when len(df) < min_samples_absolute."""
    df = pd.DataFrame({'a': [1, 2], 'target': [0, 1]})
    with pytest.raises(InsufficientDataError):
        validate_dataset(df, 'target', config, state)

def test_target_constant_error(config, state):
    """Test that TargetConstantError is raised when target has <= 1 unique value."""
    # Sufficient rows (6 > 5), but target is constant
    df = pd.DataFrame({'a': range(6), 'target': [1] * 6})
    with pytest.raises(TargetConstantError):
        validate_dataset(df, 'target', config, state)

def test_excessive_missing_data_error(config, state):
    """Test that ExcessiveMissingDataError is raised when target missing % > max."""
    # 6 rows. max_missing=0.2 (1.2 rows). So 2 missing is > 20% (it's 33%)
    df = pd.DataFrame({
        'a': range(6),
        'target': [1, 0, 1, 0, None, None]
    })
    with pytest.raises(ExcessiveMissingDataError):
        validate_dataset(df, 'target', config, state)

def test_drop_cols_and_state_mutation(config, state):
    """Test that high cardinality/null columns are dropped and state is updated."""
    # min_samples=5, max_cardinality=0.5
    df = pd.DataFrame({
        'good_col': [1, 2, 3, 4, 5, 6],
        'all_null': [None] * 6,
        'high_card': ['a', 'b', 'c', 'd', 'e', 'f'], # 6/6 = 1.0 > 0.5
        'low_card': ['x', 'x', 'y', 'y', 'z', 'z'], # 3/6 = 0.5 <= 0.5
        'target': [0, 1, 0, 1, 0, 1]
    })
    
    cleaned_df = validate_dataset(df, 'target', config, state)
    
    # Assertions for dataframe columns
    assert 'all_null' not in cleaned_df.columns
    assert 'high_card' not in cleaned_df.columns
    assert 'good_col' in cleaned_df.columns
    assert 'low_card' in cleaned_df.columns
    assert 'target' in cleaned_df.columns
    
    # Assertions for state updates
    dropped_reasons = {d['col']: d['reason'] for d in state.dropped_columns}
    
    assert 'all_null' in dropped_reasons
    assert dropped_reasons['all_null'] == "100% missing"
    
    assert 'high_card' in dropped_reasons
    # The reason string contains the ratio, checking for partial match
    assert "High cardinality" in dropped_reasons['high_card']

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
