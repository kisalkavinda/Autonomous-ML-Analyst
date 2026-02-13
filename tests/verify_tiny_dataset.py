import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset
from src.exceptions import InsufficientDataError

def test_tiny_dataset():
    print("--- Test 1: The Tiny Dataset ---")
    # Create dataset with 10 rows (below default min of 30)
    df = pd.DataFrame({'a': range(10), 'target': range(10)})
    config = AnalysisConfig()
    state = AnalysisState()
    
    try:
        validate_dataset(df, 'target', config, state)
        print("FAIL: Validation did not raise InsufficientDataError.")
    except InsufficientDataError as e:
        print(f"PASS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_tiny_dataset()
