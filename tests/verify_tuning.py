
import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import engineer_features, build_preprocessor
from src.model_trainer import run_experiment
from src.config import AnalysisConfig
from src.state import AnalysisState

def test_engineer_features():
    print("\n--- Testing Feature Engineering ---")
    df = pd.DataFrame({
        'A': [1, 2, 3, 100],  # Skewed?
        'B': [2, 4, 6, 8],
        'C': ['X', 'Y', 'X', 'Y'],
        'Target': [10, 20, 30, 40]
    })
    
    # Expected: 
    # A_squared, B_squared (poly)
    # A_x_B (interaction)
    # A_log? (skewness might be high enough for A due to 100)
    
    df_eng = engineer_features(df, target_col='Target')
    print("Engineered Columns:", df_eng.columns.tolist())
    
    assert 'A_squared' in df_eng.columns, "Missing A_squared"
    assert 'B_squared' in df_eng.columns, "Missing B_squared"
    assert 'A_x_B' in df_eng.columns, "Missing Interaction Term A_x_B"
    # A_log might depend on exact skew value, let's just check it runs
    
    print("✅ Feature Engineering verification passed!")

def test_tuning():
    print("\n--- Testing Hyperparameter Tuning ---")
    # Tiny dataset
    df = pd.DataFrame({
        'A': np.random.rand(50),
        'B': np.random.rand(50),
        'Target': np.random.rand(50)
    })
    
    state = AnalysisState()
    # Config is frozen (dataclass(frozen=True) probably), so instantiate with value
    config = AnalysisConfig(min_samples_for_split=20)
    
    # Preprocess
    df_eng = engineer_features(df, target_col='Target')
    preprocessor, X, y = build_preprocessor(df_eng, 'Target', state)
    
    # Run Experiment (Should trigger RandomizedSearchCV)
    best_pipeline = run_experiment(X, y, preprocessor, config, state)
    
    print("Best Model Name:", state.selected_model)
    print("Scores:", state.model_scores)
    
    assert isinstance(best_pipeline, Pipeline)
    assert state.selected_model is not None
    
    # Check if we can predict
    pred = best_pipeline.predict(X.head())
    print("Predictions:", pred)
    
    print("✅ Hyperparameter Tuning verification passed!")

if __name__ == "__main__":
    test_engineer_features()
    test_tuning()
