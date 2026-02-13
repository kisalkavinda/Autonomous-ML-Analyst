import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset

def test_dirty_target():
    print("--- Test: The Dirty Target ---")
    
    # Create dataset based on the user's description (best_selling_video_games.csv autoclave)
    # Row 11: Normal
    # Row 12: Corrupted (Rank missing, shifted left)
    
    data = {
        'Rank': ['11', 'The Sims'], # Mixed
        'Title': ['Overwatch', '50'],
        'Releaseyear': ['2016', 'Maxis'],
        'Sales(millions)': ['50', 'The Sims'] # Target: Mixed Numeric/Text
    }
    
    # Let's make a larger dataset to pass min_samples_absolute=30
    # 50 rows of good data, 5 rows of bad data
    
    good_data = {
        'Rank': range(50),
        'Title': [f'Game {i}' for i in range(50)],
        'Sales(millions)': [float(i) for i in range(50)]
    }
    
    bad_data = {
        'Rank': ['The Sims'] * 5,
        'Title': ['50'] * 5,
        'Sales(millions)': ['Cyberpunk'] * 5 # Text in target
    }
    
    df_good = pd.DataFrame(good_data)
    df_bad = pd.DataFrame(bad_data)
    
    df = pd.concat([df_good, df_bad], ignore_index=True)
    
    # Shuffle to mix them
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure float column is object due to 'Cyberpunk'
    df['Sales(millions)'] = df['Sales(millions)'].astype(str)
    
    print(f"Original Dtype: {df['Sales(millions)'].dtype}")
    
    config = AnalysisConfig()
    state = AnalysisState()
    target_col = 'Sales(millions)'
    
    try:
        df_clean = validate_dataset(df, target_col, config, state)
        
        print(f"Cleaned Dtype: {df_clean[target_col].dtype}")
        
        # Verify coercion
        if pd.api.types.is_numeric_dtype(df_clean[target_col]):
            print("PASS: Target column coerced to numeric.")
        else:
            print("FAIL: Target column remains object.")
            
        # Verify Warning
        warnings = [w for w in state.warnings if "mixed text/numbers" in w]
        if warnings:
            print(f"PASS: Warning logged: {warnings[0]}")
        else:
            print("FAIL: No warning logged for mixed type.")
            
        # Verify bad rows are NaN (which implies they are dropped effectively by models later, 
        # or we check if they are dropped?
        # The prompt says: "Proceed to the Missing Values check (which automatically drops the newly created NaN rows)."
        # Wait, the missing value check in validator is:
        # 1. Check missing percentage (step 3) -> Step 3 is executed BEFORE step 4 (where I inserted coercion?)
        
        # WAIT. The user instruction said: "Open src/data_validator.py and add this block right before the Missing Values check (around Step 3)"
        # Step 3 in my file is "Check missing percentage".
        # Step 4 is "Drop columns that are 100% null".
        
        # Let's check where I inserted it.
        # I inserted it at 'StartLine: 25', which is before 'Step 4'.
        # The prompt said "right before the Missing Values check (around Step 3)".
        # Step 3 starts at line 20.
        
        # IF I inserted it at Step 4, it's AFTER the missing value check (Step 3).
        # This might be WRONG order if I wanted the missing check to run on the coerced values?
        # "Missing Values check (which automatically drops the newly created NaN rows)"
        
        # Actually my validator DOES NOT have a "Drop Missing Rows" step.
        # It has:
        # 1. Min Samples
        # 2. Target Constant
        # 3. Missing Percentage Check (Raises error if too high)
        # 4. Drop Cols 100% Null.
        
        # It does NOT drop rows with missing values in the validator.
        # The Preprocessor (imputer) handles missing values usually.
        # UNLESS the user implies that `run_experiment` or `preprocessor` handles them.
        # OR "Proceed to the Missing Values check" meant Step 3 check?
        
        # If I convert 'Cyberpunk' to NaN, then missing_pct increases.
        # If I do this BEFORE Step 3, then ExcessiveMissingDataError might trigger if too many are bad.
        # This seems correct behavior.
        
        # User said: "add this block right before the Missing Values check (around Step 3)"
        # My tool call inserted it at Line 25, which is AFTER Step 3 (Line 20-23).
        # I MISSED THE INSTRUCTION LOCATION.
        # I need to MOVE it to before Step 3.
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dirty_target()
