import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.data_validator import validate_dataset
from src.preprocessing import build_preprocessor
from src.model_trainer import run_experiment
from src.exceptions import InsufficientDataError

def test_no_features():
    print("--- Test: No Features Left (Gender Submission) ---")
    
    # gender_submission.csv typically has: PassengerId, Survived
    # PassengerId is 100% unique numeric -> Dropped by Phase 6.4 fix
    # Survived is Target
    # Result: X has 0 columns.
    
    data = {
        'PassengerId': range(1, 41),        # 40 rows (enough for >30 min)
        'Survived': [0, 1] * 20
    }
    
    df = pd.DataFrame(data)
    
    config = AnalysisConfig()
    state = AnalysisState()
    target_col = 'Survived'
    
    try:
        print("1. Validating...")
        df_clean = validate_dataset(df, target_col, config, state)
        
        print(f"Columns after validation: {df_clean.columns.tolist()}")
        
        # Check if we have features?
        # My validator currently passes df_clean with only 'Survived'
        
        print("2. Preprocessing...")
        preprocessor, X, y = build_preprocessor(df_clean, target_col, state)
        print(f"X shape: {X.shape}")
        
        print("3. Training...")
        # This is where it likely crashes with ValueError
        run_experiment(X, y, preprocessor, config, state)
        
        print("FAIL: Pipeline ran with 0 features!")
        
    except InsufficientDataError as e:
        print(f"PASS: Caught expected InsufficientDataError: {e}")
    except ValueError as e:
        print(f"FAIL: Caught ValueError (System Failure): {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_no_features()
