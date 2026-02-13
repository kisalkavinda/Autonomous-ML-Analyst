import pandas as pd
import numpy as np
import sys
import os
from sklearn.datasets import make_classification

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset
from src.preprocessing import build_preprocessor
from src.model_trainer import run_experiment
from src.report_generator import generate_markdown_report

def test_happy_path():
    print("--- Test 3: The Happy Path ---")
    # Generate clean classification dataset (100 samples)
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    df['target'] = y
    
    config = AnalysisConfig()
    state = AnalysisState()
    target_col = 'target'
    
    try:
        print("1. Validating...")
        df_clean = validate_dataset(df, target_col, config, state)
        
        print("2. Preprocessing...")
        preprocessor, X_train, y_train = build_preprocessor(df_clean, target_col, state)
        
        print("3. Training...")
        best_model = run_experiment(X_train, y_train, preprocessor, config, state)
        
        print("4. Reporting...")
        report = generate_markdown_report(state, target_col)
        
        print("\n--- Final Report Preview ---")
        print(report[:500] + "...\n(truncated)")
        
        if state.selected_model and len(state.model_scores) > 0:
            print(f"PASS: Pipeline completed. Selected model: {state.selected_model}")
        else:
            print("FAIL: Pipeline ran but no model selected.")
            
    except Exception as e:
        print(f"FAIL: Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_happy_path()
