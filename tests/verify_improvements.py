
import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import clean_dataset
from src.data_validator import validate_dataset 
from src.config import AnalysisConfig
from src.state import AnalysisState
# We can't easily import validate_inference_data since it's in app.py and app.py has streamlit calls at top level.
# We will verify that manually or via a mocked app.

def test_semantic_leakage():
    print("\n--- Testing Semantic Leakage ---")
    df = pd.DataFrame({
        'Revenue': [100, 200, 300],
        'Cost': [50, 100, 150], # Should NOT be dropped if we use exact matching "term == col_lower"
        'Marketing_Cost': [10, 20, 30], # Should DEFINITELY NOT be dropped
        'Total_Revenue': [100, 200, 300], # Should be dropped if strict match allows exact equality
        'Profit': [50, 100, 150] # Target is Revenue, so Profit is allowed? Wait.
    })
    
    config = AnalysisConfig()
    state = AnalysisState()
    
    # LEAKAGE_MAPPING has 'revenue': ['profit', 'cost', 'margin', 'tax', 'total']
    # If target is 'Revenue', then 'cost' is suspicious.
    
    # Current logic: only drop if col_lower == term.
    # term='cost'. col='Cost' -> match. Dropped.
    # term='cost'. col='Marketing_Cost' -> mismatch. Kept.
    
    try:
        df_valid = validate_dataset(df, 'Revenue', config, state)
        print("Columns remaining:", df_valid.columns.tolist())
        
        # 'Cost' matches 'cost'. Should be dropped.
        if 'Cost' in df_valid.columns:
            print("❌ 'Cost' was NOT dropped (Expected drop due to exact match with 'cost').")
        else:
            print("✅ 'Cost' was correctly dropped.")
            
        # 'Marketing_Cost' should be kept.
        if 'Marketing_Cost' in df_valid.columns:
             print("✅ 'Marketing_Cost' was correctly PRESERVED.")
        else:
             print("❌ 'Marketing_Cost' was incorrectly dropped.")
             
    except Exception as e:
        print(f"Validation failed: {e}")

def test_multiclass_coefficients():
    print("\n--- Testing Multi-class Coefficients ---")
    # Simulate a multi-class logistic regression
    X = np.random.rand(100, 5)
    y = np.random.choice([0, 1, 2], 100) # 3 classes
    
    model = LogisticRegression()
    model.fit(X, y)
    
    coefs = model.coef_
    print(f"Coef shape: {coefs.shape}") # Should be (3, 5)
    
    if len(coefs.shape) > 1:
        importances = np.abs(coefs).mean(axis=0)
        print("Importances shape:", importances.shape)
        assert importances.shape == (5,), "Importance shape mismatch"
        print("✅ Multi-class coefficient averaging works.")
    else:
        print("❌ Model did not produce multi-class coefficients.")

if __name__ == "__main__":
    test_semantic_leakage()
    test_multiclass_coefficients()
