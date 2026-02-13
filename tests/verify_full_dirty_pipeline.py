import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset
from src.preprocessing import build_preprocessor
from src.model_trainer import run_experiment
from src.report_generator import generate_markdown_report

def test_full_dirty_pipeline():
    print("--- Test: Full Pipeline with Dirty Target ---")
    
    # Create dataset based on the user's description (best_selling_video_games.csv autoclave)
    # We need enough samples for the pipeline to run (default min 30)
    
    # 50 rows of valid numeric data (as strings to mimic 'object' dtype initially)
    good_data = {
        'Rank': range(50),
        'other_feat': [float(i) for i in range(50)],
        'Sales(millions)': [str(float(i)) for i in range(50)]
    }
    
    # 5 rows of "corrupted" text data in target
    bad_data = {
        'Rank': ['The Sims'] * 5,
        'other_feat': [100.0] * 5,
        'Sales(millions)': ['Cyberpunk'] * 5 # The Dirty Target
    }
    
    df_good = pd.DataFrame(good_data)
    df_bad = pd.DataFrame(bad_data)
    
    df = pd.concat([df_good, df_bad], ignore_index=True)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    target_col = 'Sales(millions)'
    print(f"Original Target Dtype: {df[target_col].dtype}")
    print(f"Contains 'Cyberpunk': {'Cyberpunk' in df[target_col].values}")
    
    config = AnalysisConfig()
    state = AnalysisState()
    
    try:
        # 1. Validation (Should coerce target)
        print("\n1. Running Validation...")
        df_clean = validate_dataset(df, target_col, config, state)
        
        print(f"Cleaned Target Dtype: {df_clean[target_col].dtype}")
        print(f"Warnings Logged: {len(state.warnings)}")
        for w in state.warnings:
            print(f"  - {w}")
            
        # 2. Preprocessing
        print("\n2. Building Preprocessor...")
        preprocessor, X, y = build_preprocessor(df_clean, target_col, state)
        
        # 3. Training
        print("\n3. Training Models...")
        # Since target is float, should be regression
        best_model = run_experiment(X, y, preprocessor, config, state)
        
        # 4. Reporting
        print("\n4. Generating Report...")
        report = generate_markdown_report(state, target_col)
        
        print("\n" + "="*40)
        print("FINAL REPORT SNIPPET")
        print("="*40)
        
        # Print the Model Leaderboard section
        if "## 🏆 Model Leaderboard" in report:
            start_idx = report.find("## 🏆 Model Leaderboard")
            print(report[start_idx:])
        else:
            print("Leaderboard not found in report!")
            
    except Exception as e:
        print(f"FAIL: Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_dirty_pipeline()
