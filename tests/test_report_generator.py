import pytest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state import AnalysisState
from src.report_generator import generate_markdown_report

def test_generate_markdown_report():
    # 1. Instantiate state
    state = AnalysisState()
    
    # 2. Populate with dummy data
    state.dropped_columns.append({"col": "bad_col", "reason": "Too many nulls"})
    state.warnings.append("Dataset is slightly imbalanced")
    state.preprocessing_steps.append("Scaled numeric features")
    state.model_scores = {
        'RandomForest': {'F1-Macro': 0.88},
        'LogisticRegression': {'F1-Macro': 0.75}
    }
    state.selected_model = 'RandomForest'
    
    target_col = 'Sales'
    
    # 3. Call generator
    report = generate_markdown_report(state, target_col)
    
    # 4. Assertions
    assert isinstance(report, str)
    
    # Check keywords/content
    assert "# 📊 Autonomous ML Analysis Report" in report
    assert "## 🎯 Executive Summary" in report
    assert f"Target Column: **{target_col}**" in report
    assert "Best Performing Model: **RandomForest**" in report
    
    assert "## ⚠️ Data Quality Interventions" in report
    assert "bad_col" in report
    assert "Too many nulls" in report
    assert "Dataset is slightly imbalanced" in report
    
    assert "## ⚙️ Preprocessing Strategy" in report
    assert "Scaled numeric features" in report
    
    assert "## 🏆 Model Leaderboard" in report
    assert "| Model | Metric | Score |" in report
    assert "| RandomForest | F1-Macro | 0.8800 |" in report
    assert "| LogisticRegression | F1-Macro | 0.7500 |" in report

def test_generate_markdown_report_empty_state():
    state = AnalysisState()
    target_col = 'Target'
    
    report = generate_markdown_report(state, target_col)
    
    assert "✅ No severe data quality issues detected." in report
    assert "No preprocessing steps recorded." in report
    assert "No models trained." in report
    assert "Best Performing Model: **None selected**" in report

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
