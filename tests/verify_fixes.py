import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import shutil
import joblib
from src.utils import save_model_package
from src.model_trainer import run_experiment, predict_with_confidence
from src.config import AnalysisConfig
from src.state import AnalysisState
from src.preprocessing import build_preprocessor, engineer_features

def test_fixes():
    print("Testing fixes...")
    
    # Setup dummy data with enough samples for CV
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target_reg': np.random.rand(100)
    })
    
    # Test Regression (SHAP Logic + predict_with_confidence + save_model_package)
    print("Test 1: Regression (SHAP Logic + predict_with_confidence + save_model_package)")
    config = AnalysisConfig()
    state = AnalysisState()
    
    # Feature Engineering
    try:
        df_engineered = engineer_features(df, target_col='target_reg', state=state)
        preprocessor, X, y = build_preprocessor(df_engineered, 'target_reg', state)
        
        # Run experiment
        pipeline = run_experiment(X, y, preprocessor, config, state)
        print("✅ run_experiment completed successfully (SHAP check passed if no crash)")
        
        # Test save_model_package
        model_path, meta_path = save_model_package(pipeline, state, "test_model.pkl")
        if os.path.exists(model_path) and os.path.exists(meta_path):
            print("✅ save_model_package worked")
            # cleanup
            os.remove(model_path)
            os.remove(meta_path)
        else:
            print("❌ save_model_package failed to create files")
            
        # Test predict_with_confidence
        preds, lower, upper = predict_with_confidence(pipeline, X)
        if len(preds) == len(X) and len(lower) == len(X) and len(upper) == len(X):
             print("✅ predict_with_confidence returned correct shapes")
        else:
             print("❌ predict_with_confidence returned incorrect shapes")
             
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixes()
