import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the formulation of the analysis."""
    max_missing_percentage: float = float(os.getenv("MAX_MISSING_PERCENTAGE", 0.40))
    max_cardinality_ratio: float = float(os.getenv("MAX_CARDINALITY_RATIO", 0.90))
    min_samples_absolute: int = int(os.getenv("MIN_SAMPLES_ABSOLUTE", 30))
    min_samples_for_split: int = int(os.getenv("MIN_SAMPLES_FOR_SPLIT", 50))
    max_correlation: float = float(os.getenv("MAX_CORRELATION", 0.95))
