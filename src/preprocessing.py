import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.state import AnalysisState
from typing import Tuple

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
