import pandas as pd
import numpy as np
import sys
import os
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import shap

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import detect_imbalance

def test_imbalance_and_shap():
    print("🧪 Testing Phase 3: SMOTE Pipeline & SHAP Extraction...")
    
    # 1. Create a highly imbalanced classification dataset (95:5)
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(106, 3), columns=['Num1', 'Num2', 'Num3'])
    y = pd.Series([0]*100 + [1]*6) # 100 to 6 imbalance (approx 5.6% minority, <30% triggers SMOTE)
    
    # Test Imbalance Detector
    needs_smote = detect_imbalance(y, is_regression=False)
    assert needs_smote is True, "🚨 Failed to detect severe imbalance!"
    print("✅ Imbalance detection passed.")
    
    # 2. Build Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Num1', 'Num2', 'Num3'])
    ])
    
    steps = [('preprocessor', preprocessor)]
    if needs_smote:
        # Dynamic k_neighbors to match production logic
        min_class_samples = y.value_counts().min()
        k_neighbors = min(5, min_class_samples - 1)
        if k_neighbors < 1: k_neighbors = 1
        
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
        print(f"✅ SMOTE appended with k_neighbors={k_neighbors}")
        
    steps.append(('model', RandomForestClassifier(n_estimators=10, random_state=42)))
    pipeline = Pipeline(steps)
    
    # Fit the pipeline
    pipeline.fit(X, y)
    print("✅ Pipeline (with SMOTE) fitted successfully.")
    
    # 3. Extract SHAP
    model = pipeline.named_steps['model']
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    # Handle SHAP output shapes
    if isinstance(shap_values, list):
        importance = np.abs(shap_values[1]).mean(axis=0)
    elif len(np.array(shap_values).shape) == 3:
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        importance = np.abs(shap_values).mean(axis=0)
        
    feature_names = preprocessor.get_feature_names_out()
    shap_dict = dict(zip(feature_names, importance))
    
    assert len(shap_dict) == 3, "🚨 SHAP dictionary length mismatch!"
    print(f"✅ SHAP Values Extracted: {shap_dict}")

if __name__ == "__main__":
    try:
        test_imbalance_and_shap()
        print("\n🎉 Phase 3 Verification Passed! System is Enterprise-Ready.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise e
