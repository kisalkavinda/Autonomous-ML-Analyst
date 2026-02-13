import sys
import os

# Add project root to sys.path to allow direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state import AnalysisState
import pandas as pd

def generate_markdown_report(state: AnalysisState, target_col: str) -> str:
    """
    Generates a professional Markdown report based on the analysis state.
    """
    
    # Section 1: Executive Summary
    report = "# 📊 Autonomous ML Analysis Report\n\n"
    
    # --- NEW: Dataset Overview ---
    if not state.dataset_stats:
        from src.exceptions import AnalysisError
        raise AnalysisError("AnalysisState missing dataset snapshot")
        
    stats = state.dataset_stats
    report += "## 📈 Dataset Overview\n"
    report += f"Rows: **{stats.get('rows', 'N/A')}** | Columns: **{stats.get('columns', 'N/A')}**\n"
    report += f"- Numeric Features: {stats.get('numeric_features', 0)}\n"
    report += f"- Categorical Features: {stats.get('categorical_features', 0)}\n"
    
    # --- NEW: Target Distribution ---
    if state.target_distribution:
        report += "\n**Target Distribution**:\n"
        for label, pct in state.target_distribution.items():
            report += f"- {label}: {pct}\n"
    report += "\n"

    report += f"## 🎯 Executive Summary\n"
    report += f"Target Column: **{target_col}**\n\n"
    if state.selected_model:
        report += f"Best Performing Model: **{state.selected_model}**\n"
    else:
        report += "Best Performing Model: **None selected**\n"
    report += "\n"
    
    # --- NEW: Validation Notes ---
    if state.validations_passed:
        report += "### ✔ Validation Notes\n"
        for note in state.validations_passed:
            report += f"- {note}\n"
        report += "\n"

    # Section 2: Data Quality Interventions
    report += "## ⚠️ Data Quality Interventions\n"
    has_issues = False
    
    if state.dropped_columns:
        has_issues = True
        for item in state.dropped_columns:
            report += f"- Dropped column **{item['col']}**: {item['reason']}\n"
            
    if state.warnings:
        has_issues = True
        for warning in state.warnings:
            report += f"- Warning: {warning}\n"
            
    if not has_issues:
        report += "✅ No severe data quality issues detected.\n"
    report += "\n"

    # Section 3: Preprocessing Strategy
    report += "## ⚙️ Preprocessing Strategy\n"
    if state.preprocessing_steps:
        for step in state.preprocessing_steps:
            report += f"- {step}\n"
    else:
        report += "No preprocessing steps recorded.\n"
    report += "\n"

    # Section 4: Model Leaderboard
    report += "## 🏆 Model Leaderboard\n"
    if state.model_scores:
        report += "| Model | Metric | Score |\n"
        report += "|---|---|---|\n"
        
        # Sort models based on score if possible?
        # The requirements just say iterate and format.
        # We assume standard dictionary iteration order or could sort.
        # Let's just iterate for now as per requirements.
        
        for model_name, scores in state.model_scores.items():
            # scores is a dict like {'F1-Macro': 0.88}
            for metric, score in scores.items():
                report += f"| {model_name} | {metric} | {score:.4f} |\n"
    else:
        report += "No models trained.\n"
        
    return report

if __name__ == "__main__":
    # Demo execution
    state = AnalysisState()
    state.selected_model = "Demo Model"
    state.warnings.append("This is a demo warning.")
    print(generate_markdown_report(state, "Demo Target"))
