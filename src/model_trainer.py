import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score, make_scorer
from src.config import AnalysisConfig
from src.state import AnalysisState

def run_experiment(X: pd.DataFrame, y: pd.Series, preprocessor, config: AnalysisConfig, state: AnalysisState) -> Pipeline:
    """
    Runs an ML experiment to select the best model.
    Detects task type (Regression/Classification).
    Evaluates candidates using CV or Train/Test split.
    Updates state with scores and selected model.
    Returns the best model fitted on the entire dataset.
    """
    
    # 1. Task Detection
    # If float -> Regression
    # If integer but many unique values -> Regression (heuristic > 20)
    is_regression = False
    if pd.api.types.is_float_dtype(y):
        is_regression = True
    elif y.nunique() > 20:
        is_regression = True
    
    # 2. Candidate Models
    if is_regression:
        models = {
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression()
        }
        metric_name = 'MAE'
        # For cross_val_score, we need a scorer. neg_mean_absolute_error is standard (higher is better, so it's negative MAE)
        # We will convert back to positive MAE for logging.
        scoring = 'neg_mean_absolute_error'
    else:
        models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }
        metric_name = 'F1-Macro'
        scoring = 'f1_macro'

    best_score = -float('inf') if not is_regression else float('inf')
    best_model_name = None
    best_pipeline = None

    # 3. Evaluation Strategy
    use_cv = len(X) < config.min_samples_for_split
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        current_score = 0
        
        if use_cv:
            # 5-fold CV
            # scores are usually "higher is better" in sklearn cross_val_score
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
            current_score = np.mean(cv_scores)
            
            # If regression, convert neg MAE to positive MAE for logging (and lower is better comparison)
            if is_regression:
                logged_score = -current_score 
            else:
                logged_score = current_score
        else:
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            if is_regression:
                logged_score = mean_absolute_error(y_test, y_pred)
                # For comparison logic internally, ensure we track "best". 
                # If we decide "best_score" variable tracks the optimization metric (higher better vs lower better)
            else:
                logged_score = f1_score(y_test, y_pred, average='macro')

        # Update State
        if name not in state.model_scores:
            state.model_scores[name] = {}
        state.model_scores[name][metric_name] = logged_score
        
        # Determine if this is the new best
        if is_regression:
            # Lower MAE is better
            if logged_score < best_score:
                best_score = logged_score
                best_model_name = name
                best_pipeline = pipeline
        else:
            # Higher F1 is better
            if logged_score > best_score:
                best_score = logged_score
                best_model_name = name
                best_pipeline = pipeline

    # 4. State Mutation
    state.selected_model = best_model_name

    # 5. Final Fit
    # Fit the winning pipeline on the entire dataset
    # Note: best_pipeline might be already fitted if we used train_test_split, but fitting on full X,y is requested.
    # For CV loop, pipeline wasn't fitted on full data yet.
    # Re-create the best pipeline from scratch to ensure clean state or just call fit?
    # Calling fit on existing pipeline object is fine.
    
    # We need to retrieve the best model instance again to ensure a fresh fit?
    # Actually, reusing the pipeline object and calling fit(X, y) will re-train it.
    # However, inside the loop 'pipeline' variable changes. We saved 'best_pipeline'.
    
    # NOTE: In the loop, `pipeline` is created fresh. `best_pipeline` holds reference to one of them.
    best_pipeline.fit(X, y)
    
    return best_pipeline
