import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.state import AnalysisState
from src.exceptions import InsufficientDataError
from typing import Tuple

def detect_imbalance(y: pd.Series, is_regression: bool) -> bool:
    """
    Detects if a classification dataset suffers from severe class imbalance.
    Returns True if the minority class makes up less than 30% of the data.
    """
    if is_regression:
        return False # SMOTE does not apply to continuous regression targets
        
    class_counts = y.value_counts(normalize=True)
    
    # Check if the smallest class is less than 30% of the dataset
    if not class_counts.empty and class_counts.min() < 0.3:
        return True
        
    return False

def engineer_features(df: pd.DataFrame, target_col: str = None, state: AnalysisState = None) -> pd.DataFrame:
    """
    Automatically creates derived mathematical features.
    Designed to be safely run in both Lab (Training) and Factory (Inference).
    Uses state.feature_engineering_metadata to ensure consistency between Train and Test.
    """
    df_eng = df.copy()
    new_features = []
    
    if df_eng.empty:
        return df_eng
    
    # Identify numeric columns (excluding the target if it exists in this dataset)
    numeric_cols = df_eng.select_dtypes(include=['number']).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    if not numeric_cols:
        return df_eng # No numeric features to engineer

    # --- DETERMINISTIC MODE (INFERENCE) ---
    # If metadata exists, we follow the instructions exactly, ignoring data distribution
    if state and state.feature_engineering_metadata:
        meta = state.feature_engineering_metadata
        
        # 1. Log Transforms
        for col in meta.get('log_transform', []):
            if col in df_eng.columns:
                df_eng[f'{col}_log'] = np.log1p(df_eng[col])
                
        # 2. Squared Transforms
        for col in meta.get('squared_transform', []):
            if col in df_eng.columns:
                df_eng[f'{col}_squared'] = df_eng[col] ** 2
                
        # 3. Interactions
        for col1, col2 in meta.get('interactions', []):
            if col1 in df_eng.columns and col2 in df_eng.columns:
                df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]
                
        # 4. Age Groups
        for col in meta.get('age_groups', []):
            if col in df_eng.columns:
                try:
                    df_eng[f'{col}_group'] = pd.cut(
                        df_eng[col], 
                        bins=[0, 18, 35, 50, 65, 100],
                        labels=['young', 'adult', 'middle', 'senior', 'elderly']
                    ).astype(str)
                except: pass
                
        # 5. Ratios
        for col1, col2 in meta.get('ratios', []):
            if col1 in df_eng.columns and col2 in df_eng.columns:
                 safe_denominator = df_eng[col2].replace(0, np.nan)
                 df_eng[f'{col1}_per_{col2}'] = df_eng[col1] / safe_denominator
                 
        return df_eng

    # --- DISCOVERY MODE (TRAINING) ---
    # We decide what features to create and record the decisions
    
    metadata = {
        'log_transform': [],
        'squared_transform': [],
        'interactions': [],
        'age_groups': [],
        'ratios': []
    }

    # 1. Log Transforms for Highly Skewed Data (Normalizes long tails)
    for col in numeric_cols:
        if df_eng[col].dropna().min() >= 0: # Log only works on positive numbers (and 0 via log1p)
            # Check skewness (only if we have enough variance)
            if df_eng[col].nunique() > 10:
                try:
                    skewness = abs(skew(df_eng[col].dropna()))
                    if skewness > 1.0: # Highly skewed
                        df_eng[f'{col}_log'] = np.log1p(df_eng[col])
                        new_features.append(f'{col}_log')
                        metadata['log_transform'].append(col)
                except:
                    pass

    # 2. Polynomial Features (Squares) for the first 3 numeric columns
    # Helps tree models understand non-linear/exponential relationships
    for col in numeric_cols[:5]: # Expanded to top 5 based on feedback
        df_eng[f'{col}_squared'] = df_eng[col] ** 2
        new_features.append(f'{col}_squared')
        metadata['squared_transform'].append(col)

    # 3. Interaction Terms (Multiplying the top 2 numeric features)
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]
        new_features.append(f'{col1}_x_{col2}')
        metadata['interactions'].append((col1, col2))

    # 4. Binning for Age-like columns
    # Heuristic: Column name contains 'age' and values are within human lifespan
    for col in numeric_cols:
        if 'age' in col.lower() and df_eng[col].max() < 120 and df_eng[col].min() >= 0:
            # Create age groups
            try:
                # Cast to string to ensure it's treated as categorical by our pipeline selector
                df_eng[f'{col}_group'] = pd.cut(
                    df_eng[col], 
                    bins=[0, 18, 35, 50, 65, 100],
                    labels=['young', 'adult', 'middle', 'senior', 'elderly']
                ).astype(str)
                new_features.append(f'{col}_group')
                metadata['age_groups'].append(col)
            except:
                pass

    # 5. Ratio features for financial data
    financial_keywords = ['revenue', 'cost', 'price', 'sales', 'profit', 'income', 'expense']
    financial_cols = [col for col in numeric_cols 
                     if any(kw in col.lower() for kw in financial_keywords)]
    
    if len(financial_cols) >= 2:
        # Create ratios between financial columns (e.g. Profit / Revenue)
        # Limit to first few to avoid explosion
        for i, col1 in enumerate(financial_cols[:3]):
            for col2 in financial_cols[i+1:4]:
                metadata['ratios'].append((col1, col2))
                # Avoid division by zero
                safe_denominator = df_eng[col2].replace(0, np.nan)
                ratio_name = f'{col1}_per_{col2}'
                df_eng[ratio_name] = df_eng[col1] / safe_denominator
                # Fill NaNs created by division by zero with 0 or mean? 
                # Let's leave as NaN and let Imputer handle it
                new_features.append(ratio_name)

    if state:
        if new_features:
            state.preprocessing_steps.append(
                f"🔬 Generated {len(new_features)} engineered features: "
                f"{', '.join(new_features[:5])}{'...' if len(new_features) > 5 else ''}"
            )
        # SAVE THE DECISIONS
        state.feature_engineering_metadata = metadata

    return df_eng

def build_preprocessor(df: pd.DataFrame, target_col: str, state: AnalysisState) -> Tuple[ColumnTransformer, pd.DataFrame, pd.Series]:
    """
    Constructs a preprocessing pipeline dynamically based on validation results.
    Separates features and target, detects data types, and builds transformers.
    Updates state with detailed preprocessing steps.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.shape[1] == 0:
        raise InsufficientDataError(
            "No features available for training. All columns were either "
            "the target or removed during data validation."
        )

    # Dynamic feature detection
    # Note: 'int64' and 'float64' covers most standard pandas objects. 
    # Using 'number' might be safer but user specified explicit types or logic similar to it.
    # User said: select_dtypes(include=['int64', 'float64'])
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # User said: select_dtypes(include=['object', 'category', 'string'])
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Numeric Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Logging to state
    if numeric_features:
        state.preprocessing_steps.append(
            f"Applied median imputation and standard scaling to {len(numeric_features)} numeric features."
        )
    
    if categorical_features:
        state.preprocessing_steps.append(
            f"Applied most_frequent imputation and one-hot encoding to {len(categorical_features)} categorical features."
        )

    return preprocessor, X, y
