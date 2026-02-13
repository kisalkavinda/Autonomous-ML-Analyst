import pandas as pd
import numpy as np
from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import InsufficientDataError, TargetConstantError, ExcessiveMissingDataError

def validate_dataset(df: pd.DataFrame, target_col: str, config: AnalysisConfig, state: AnalysisState) -> pd.DataFrame:
    """
    Validates and cleans the dataset based on configuration rules.
    Updates the state with dropped columns and reasons.
    """
    # 1. Check minimum samples
    if len(df) < config.min_samples_absolute:
        raise InsufficientDataError(f"Dataset has {len(df)} samples, which is less than the minimum required ({config.min_samples_absolute}).")

    # 2. Check if target is constant
    if df[target_col].nunique() <= 1:
        raise TargetConstantError(f"Target column '{target_col}' is constant (1 or fewer unique values).")

    # 3. Check missing percentage of target
    missing_pct = df[target_col].isnull().mean()
    if missing_pct > config.max_missing_percentage:
        raise ExcessiveMissingDataError(f"Target column '{target_col}' has {missing_pct:.2%} missing values, exceeding limit of {config.max_missing_percentage:.0%}.")

    # 4. Drop columns that are 100% null
    # Identify columns that are all null
    # We use a list to avoid modifying the iterator
    cols_to_drop_null = [col for col in df.columns if df[col].isnull().all()]
    
    if cols_to_drop_null:
        df = df.drop(columns=cols_to_drop_null)
        for col in cols_to_drop_null:
            state.dropped_columns.append({"col": col, "reason": "100% missing"})

    # 5. For object or category dtypes, drop if cardinality > ratio
    # Re-fetch columns as they might have changed
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    cols_to_drop_cardinality = []

    for col in object_cols:
        # We usually shouldn't drop the target column based on feature cardinality rules, 
        # but the prompt didn't strictly exclude it. However, removing target would break things.
        # Assuming standard practice: Don't drop target.
        if col == target_col:
            continue
            
        cardinality_ratio = df[col].nunique() / len(df)
        if cardinality_ratio > config.max_cardinality_ratio:
            cols_to_drop_cardinality.append(col)
            state.dropped_columns.append({"col": col, "reason": f"High cardinality ratio ({cardinality_ratio:.2f})"})

    if cols_to_drop_cardinality:
        df = df.drop(columns=cols_to_drop_cardinality)

    return df
