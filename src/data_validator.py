import pandas as pd
import numpy as np
from typing import Optional
from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import InsufficientDataError, TargetConstantError, ExcessiveMissingDataError


LEAKAGE_MAPPING = {
    'revenue': ['profit', 'cost', 'margin', 'tax', 'total'],
    'sales': ['profit', 'cost', 'margin', 'revenue'],
    'profit': ['revenue', 'sales', 'cost', 'tax'],
    'price': ['cost', 'discount'], 
    'total': ['subtotal', 'tax']
}

def suggest_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently identifies the most likely target column using a 3-tier strategy.
    
    Strategy 1: Semantic (Keyword Matching)
    Strategy 2: Structural (Position - Last Column)
    Strategy 3: Statistical (Mutual Information)
    """
    columns = df.columns.tolist()
    
    # ---------------------------------------------------------
    # Tier 1: Semantic Analysis (The "Human" Intuition)
    # ---------------------------------------------------------
    target_keywords = [
        'target', 'label', 'class', 'outcome', 'y', 
        'survived', 'price', 'revenue', 'sales', 'churn', 
        'diagnosis', 'species', 'salary', 'profit'
    ]
    
    for col in columns:
        if col.lower() in target_keywords:
            return col
            
    # ---------------------------------------------------------
    # Tier 2: Structural Analysis (The "CSV" Standard)
    # ---------------------------------------------------------
    # In 90% of datasets (Kaggle/Scikit-Learn), the target is the last column.
    last_col = columns[-1]
    
    # Check if the last column is a candidate (not an ID)
    if df[last_col].nunique() < len(df) * 0.9: 
        return last_col

    # ---------------------------------------------------------
    # Tier 3: Statistical Analysis (Mutual Information)
    # ---------------------------------------------------------
    # If the last column is an ID (100% unique), we need math.
    # We calculate the "Mutual Information" score for each column 
    # against all others. The target is usually the dependent variable,
    # so it shares information with the features.
    
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import LabelEncoder
        
        # Take a sample for speed (max 1000 rows)
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        # Simple encoding for the math to work
        block = sample.copy()
        for col in block.columns:
            if block[col].dtype == 'object':
                block[col] = LabelEncoder().fit_transform(block[col].astype(str))
        
        # Calculate dependency scores
        # We crudely sum the correlation of each column against all others
        scores = {}
        for candidate in block.columns:
            # Skip high cardinality ID columns
            if block[candidate].nunique() > len(block) * 0.9:
                continue
                
            # Quick correlation sum
            # (In a real scenario, we'd do full MI, but correlation is a fast proxy)
            corr_sum = block.corrwith(block[candidate]).abs().sum()
            scores[candidate] = corr_sum
            
        # Return the column with the highest accumulated correlation/dependency
        if scores:
            return max(scores, key=scores.get)
            
    except Exception:
        pass # Fallback if math fails

    return columns[-1]

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
    for key, terms in LEAKAGE_MAPPING.items():
        if key in target_lower:
            suspicious_terms.extend(terms)
            
    if suspicious_terms:
        # We have a relevant leakage context (e.g. target is 'revenue')
        cols_to_drop_semantic = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            col_lower = col.lower()
            
            # Substring matching: Check if any prohibited term exists in the column name
            # e.g. "profit" in "Gross_Profit" -> True
            for term in suspicious_terms:
                if term in col_lower:
                    cols_to_drop_semantic.append(col)
                    state.dropped_columns.append({
                        "col": col,
                        "reason": f"Semantic Leakage (Suspicious Feature Name: '{term}')"
                    })
                    state.warnings.append(
                        f"🧠 Semantic Drop: '{col}' removed due to suspicious context (Term: '{term}')."
                    )
                    break # Stop checking other terms for this column
        
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
