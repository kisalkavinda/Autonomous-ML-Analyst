import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset

def test_numeric_id():
    print("--- Test: Numeric ID Dropping ---")
    
    # Create dataset resembling gender_submission.csv
    # PassengerId (100% unique int), Survived (Target)
    
    data = {
        'PassengerId': range(1, 101),       # 1 to 100
        'Survived': [0, 1] * 50             # Target
    }
    
    df = pd.DataFrame(data)
    
    config = AnalysisConfig()
    state = AnalysisState()
    target_col = 'Survived'
    
    try:
        print(f"Columns before: {df.columns.tolist()}")
        df_clean = validate_dataset(df, target_col, config, state)
        print(f"Columns after: {df_clean.columns.tolist()}")
        
        # Verify PassengerId is dropped
        if 'PassengerId' not in df_clean.columns:
            print("PASS: 'PassengerId' was dropped.")
            # Verify log
            dropped = [d for d in state.dropped_columns if d['col'] == 'PassengerId']
            if dropped:
                print(f"PASS: Logged reason: {dropped[0]['reason']}")
            else:
                print("FAIL: Log missing.")
        else:
            print("FAIL: 'PassengerId' was NOT dropped.")
            
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_numeric_id()
