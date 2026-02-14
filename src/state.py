from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class AnalysisState:
    """Audit log and state container for the analysis process."""
    dropped_columns: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    model_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    selected_model: Optional[str] = None
    # --- NEW: Add this line to store the Explainable AI weights ---
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_engineering_metadata: Dict[str, Any] = field(default_factory=dict) # New: Stores decisions for inference consistency
    
    # --- NEW: Reporting Enhancements ---
    dataset_stats: Dict[str, Any] = field(default_factory=dict)
    target_distribution: Dict[str, Any] = field(default_factory=dict)
    validations_passed: List[str] = field(default_factory=list)
