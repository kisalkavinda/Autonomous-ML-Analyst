import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.state import AnalysisState
from typing import Tuple

def engineer_features(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Automatically creates derived mathematical features.
    Designed to be safely run in both Lab (Training) and Factory (Inference).
    """
    df_eng = df.copy()
    
    # Identify numeric columns (excluding the target if it exists in this dataset)
    numeric_cols = df_eng.select_dtypes(include=['number']).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    if not numeric_cols:
        return df_eng # No numeric features to engineer

    # 1. Log Transforms for Highly Skewed Data (Normalizes long tails)
    for col in numeric_cols:
        if df_eng[col].dropna().min() >= 0: # Log only works on positive numbers (and 0 via log1p)
            # Check skewness (only if we have enough variance)
            if df_eng[col].nunique() > 10:
                try:
                    skewness = abs(skew(df_eng[col].dropna()))
                    if skewness > 1.0: # Highly skewed
                        df_eng[f'{col}_log'] = np.log1p(df_eng[col])
                except:
                    pass

    # 2. Polynomial Features (Squares) for the first 3 numeric columns
    # Helps tree models understand non-linear/exponential relationships
    for col in numeric_cols[:3]: 
        df_eng[f'{col}_squared'] = df_eng[col] ** 2

    # 3. Interaction Terms (Multiplying the top 2 numeric features)
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]

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
