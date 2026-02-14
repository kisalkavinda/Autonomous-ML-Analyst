import pandas as pd
import numpy as np
from typing import Optional
from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import InsufficientDataError, TargetConstantError, ExcessiveMissingDataError


LEAKAGE_MAPPING = {
    'revenue': ['total_revenue', 'gross_profit', 'net_profit', 'profit_margin'],
    'sales': ['total_sales', 'net_sales', 'revenue'],
    'profit': ['revenue', 'ebitda', 'net_income'],
}

def suggest_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently identifies the most likely target column using a 3-tier strategy.
    
    Strategy 1: Semantic (Keyword Matching) - prioritized by positive words
    Strategy 2: Structural (Position - Last Non-ID/Non-Date Column)
    Strategy 3: Statistical (Mutual Information) - skipped for speed if S1/S2 work
    """
    columns = df.columns.tolist()
    
    # 0. Preparation: Define Keywords
    target_keywords = [
        'target', 'label', 'class', 'outcome', 'y', 
        'survived', 'price', 'revenue', 'sales', 'churn', 
        'diagnosis', 'species', 'salary', 'profit',
        'status', 'grade', 'rating', 'score', 'prediction', 'result',
        'quality', 'admitted', 'default'
    ]
    
    negative_keywords = [
        'id', 'index', 'year', 'date', 'month', 'day', 'time', 'timestamp', 'name', 'sku', 'phone', 'email'
    ]
    
    def is_likely_id_or_date(col_name, series):
        """Check if a column is likely an ID or Date/Time based on name and content."""
        name_lower = col_name.lower()
        
        # 1. Name Check
        if any(kw in name_lower for kw in negative_keywords):
            return True
            
        # 2. ID Check (100% unique strings/ints)
        if series.nunique() == len(series):
            return True
            
        # 3. Constant Check (1 unique value)
        if series.nunique() <= 1:
            return True
            
        return False

    # ---------------------------------------------------------
    # Tier 1: Semantic Analysis (The "Human" Intuition)
    # ---------------------------------------------------------
    for col in columns:
        if col.lower() in target_keywords:
            return col
            
    # ---------------------------------------------------------
    # Tier 2: Structural Analysis (The "CSV" Standard - Smart Backwards Scan)
    # ---------------------------------------------------------
    # Iterate backwards from the last column
    for col in reversed(columns):
        if not is_likely_id_or_date(col, df[col]):
            return col
            
    # If everything looks like an ID/Date (unlikely), fall back to the last column
    return columns[-1]

def validate_dataset(df: pd.DataFrame, target_col: str, config: AnalysisConfig, state: AnalysisState) -> pd.DataFrame:
    """
    Validates and cleans the dataset based on configuration rules.
    Updates the state with dropped columns and reasons.
    """
    # ---------------------------------------------------------
    # NEW: Capture Dataset Snapshot (Before Cleaning)
    # ---------------------------------------------------------
    state.dataset_stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    # Capture Target Distribution (if categorical)
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 20: # Heuristic for categorical
        # Get value counts as percentages
        dist = df[target_col].value_counts(normalize=True).head(10).to_dict()
        # Convert to readable percentage strings
        state.target_distribution = {k: f"{v:.1%}" for k, v in dist.items()}
        
    # ---------------------------------------------------------

    # 1. Check minimum samples
    if len(df) < config.min_samples_absolute:
        raise InsufficientDataError(f"Dataset has {len(df)} samples, which is less than the minimum required ({config.min_samples_absolute}).")
    state.validations_passed.append(f"Dataset passed minimum sample threshold (n={len(df)})")

    # 2. Check if target is constant
    if df[target_col].nunique() <= 1:
        raise TargetConstantError(f"Target column '{target_col}' is constant (1 or fewer unique values).")
    state.validations_passed.append(f"Target variance verified ({df[target_col].nunique()} unique classes/values)")

    # NEW: Coerce dirty numeric targets
    # If target is an object, check if it's actually numbers mixed with text
    if df[target_col].dtype == 'object':
        # Force convert to numeric (turns 'Cyberpunk' into NaN)
        coerced_target = pd.to_numeric(df[target_col], errors='coerce')
        
        # If at least 20% of the column successfully converted to numbers, assume it's a dirty numeric column
        if coerced_target.notna().mean() > 0.20:
            df[target_col] = coerced_target
            state.warnings.append(
                f"Target column '{target_col}' contained mixed text/numbers. Text values were coerced to NaN and dropped."
            )

    # 3. Check missing percentage of target
    missing_pct = df[target_col].isnull().mean()
    if missing_pct > config.max_missing_percentage:
        raise ExcessiveMissingDataError(f"Target column '{target_col}' has {missing_pct:.2%} missing values, exceeding limit of {config.max_missing_percentage:.0%}.")
    state.validations_passed.append("No excessive missing values detected")

    # NEW: Drop rows where target is NaN (including coerced ones)
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        state.warnings.append(f"Dropped {dropped_count} rows with missing target values in '{target_col}'.")

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

    # ---------------------------------------------------------
    # NEW: Phase 5.5 - Semantic Leakage Detection
    # ---------------------------------------------------------
    # Check for features that are contextually "too close" to the target.
    # e.g., If target is 'Revenue', we shouldn't train on 'Profit' or 'Cost'.
    # This must run BEFORE ID checks because typical financial columns might be 100% unique floats.
    
    target_lower = target_col.lower()
    
    # scan for exact target match in the mapping keys
    suspicious_terms = []
    # If target contains key (e.g. target='Total Revenue' contains 'revenue'), add suspicious terms
    for key, terms in LEAKAGE_MAPPING.items():
        if key in target_lower:
            suspicious_terms.extend(terms)
            
    if suspicious_terms:
        # We have a relevant leakage context
        cols_to_drop_semantic = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            col_lower = col.lower()
            col_clean = col_lower.replace('_', '').replace(' ', '')
            
            for term in suspicious_terms:
                term_clean = term.replace('_', '').replace(' ', '')
                
                # STRICTER CHECK: Only drop if it's a derived/parent metric
                # e.g. if target='revenue', drop 'total_revenue' (match) but NOT 'marketing_cost' (no match in new mapping)
                if term_clean == col_clean:
                    cols_to_drop_semantic.append(col)
                    state.dropped_columns.append({
                        "col": col,
                        "reason": f"Semantic Leakage (Derived Metric: '{term}')"
                    })
                    state.warnings.append(
                        f"🧠 Semantic Drop: '{col}' removed as it appears to be a derived form of {target_col}."
                    )
                    break

        
        if cols_to_drop_semantic:
            df = df.drop(columns=cols_to_drop_semantic)
            
    # 6. Check High-Cardinality Numeric Columns (Numeric IDs like PassengerId)
    # Re-fetch numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cols_to_drop_numeric_id = []

    for col in numeric_cols:
        if col == target_col:
            continue
        
        # If a numeric column is exactly 100% unique, it is almost certainly an ID or Index
        if df[col].nunique() == len(df):
            cols_to_drop_numeric_id.append(col)
            state.dropped_columns.append({
                "col": col, 
                "reason": "Numeric identifier dropped (100% unique)"
            })
            
    if cols_to_drop_numeric_id:
        df = df.drop(columns=cols_to_drop_numeric_id)


    # NEW: Phase 9 - Leakage Detection
    # ---------------------------------------------------------
    # Check for features that are simply the target in disguise.
    # We calculate the correlation of all numeric features against the target.
    # If correlation is > config.max_correlation (e.g., 0.95), we drop it.
    
    # Only if target is numeric
    if pd.api.types.is_numeric_dtype(df[target_col]):
        numeric_features = df.select_dtypes(include=['number']).columns
        cols_to_drop_leakage = []
        
        for col in numeric_features:
            if col == target_col:
                continue
                
            # Calculate absolute correlation
            corr = df[col].corr(df[target_col])
            
            if abs(corr) > config.max_correlation:
                cols_to_drop_leakage.append(col)
                state.dropped_columns.append({
                    "col": col, 
                    "reason": f"Possible Leakage: Correlation {corr:.4f} > {config.max_correlation}"
                })
                state.warnings.append(
                    f"⚠️ DROPPED '{col}' due to possible data leakage (Correlation: {corr:.4f})."
                )
        
        if cols_to_drop_leakage:
            df = df.drop(columns=cols_to_drop_leakage)

    # 7. Check if any features remain
    if len(df.columns) <= 1:
        # Only target column remains
        raise InsufficientDataError("All features were dropped due to data quality issues (high cardinality, nulls, etc.). No data left to train on.")

    return df
