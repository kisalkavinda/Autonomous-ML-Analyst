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
    
    # Create dataset for Semantic Leakage Verification
    # Target: Revenue
    # Leakers: Cost, Profit, Gross_Profit (substring match)
    
    n = 100
    cost = np.random.uniform(50, 150, n)
    profit = cost * 0.1 + np.random.normal(0, 5, n) # Moderate correlation
    revenue = cost + profit # Perfect linear combination
    
    data = {
        'Cost': cost,
        'Profit': profit,
        'Gross_Profit': profit, # Substring match test
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'] * 20, # Harmless feature
        'Revenue': revenue
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    target_col = 'Revenue'
    print(f"Target: {target_col}")
    print(f"Features: {[c for c in df.columns if c != target_col]}")
    
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
        print("\n" + "="*40)
        print("FINAL REPORT START")
        print("="*40)
        print(report[:1000]) # Print first 1000 chars to see Overview and Validation sections
        print("..." + "="*40)
            
        # VERIFICATION ASSERTIONS
        print("\n--- Verification Results ---")
        dropped_cols = [item['col'] for item in state.dropped_columns]
        print(f"Dropped Columns: {dropped_cols}")
        
        # Check Semantic Leakage
        if 'Gross_Profit' in dropped_cols:
             print("✅ PASS: 'Gross_Profit' correctly dropped (Substring Match).")
        else:
             print("❌ FAIL: 'Gross_Profit' was NOT dropped.")
             
        if 'Profit' in dropped_cols:
             print("✅ PASS: 'Profit' correctly dropped (Exact Term).")
        else:
             print("❌ FAIL: 'Profit' was NOT dropped.")
             
        if 'Cost' in dropped_cols:
             print("✅ PASS: 'Cost' correctly dropped.")
        else:
             print("❌ FAIL: 'Cost' was NOT dropped.")
             
        # Check harmless feature
        if 'City' in df_clean.columns:
             print("✅ PASS: 'City' was correctly preserved.")
        else:
             print("❌ FAIL: 'City' was incorrectly dropped.")
            
    except Exception as e:
        print(f"FAIL: Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_dirty_pipeline()
