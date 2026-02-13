from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class AnalysisState:
    """Audit log and state container for the analysis process."""
    dropped_columns: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    model_scores: Dict = field(default_factory=dict)
    selected_model: Optional[str] = None
