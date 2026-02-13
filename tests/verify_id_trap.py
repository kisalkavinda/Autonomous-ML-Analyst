import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset

def test_id_trap():
    print("--- Test 2: The ID Trap ---")
    # Create dataset with 100 rows.
    # 'PassengerId' is unique for every row (cardinality 1.0 > 0.9 default)
    # 'Category' has low cardinality
    data = {
        'PassengerId': [str(i) for i in range(100)],
        'Category': ['A' if i % 2 == 0 else 'B' for i in range(100)],
        'target': [0] * 50 + [1] * 50
    }
    df = pd.DataFrame(data)
    
    config = AnalysisConfig(min_samples_absolute=30)
    state = AnalysisState()
    
    # Run validation
    df_clean = validate_dataset(df, 'target', config, state)
    
    # Check if PassengerId was dropped
    if 'PassengerId' not in df_clean.columns:
        print("PASS: 'PassengerId' was dropped.")
        # Verify state log
        dropped_logs = [d for d in state.dropped_columns if d['col'] == 'PassengerId']
        if dropped_logs:
            print(f"PASS: State log confirms drop: {dropped_logs[0]}")
        else:
            print("FAIL: 'PassengerId' dropped but not logged in state.")
    else:
        print(f"FAIL: 'PassengerId' was NOT dropped. Columns: {df_clean.columns.tolist()}")

if __name__ == "__main__":
    test_id_trap()
