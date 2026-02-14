import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
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
    # Using RandomizedSearchCV for Autotuning
    
    for name, model in models.items():
        # Define hyperparameter grid
        if 'RandomForest' in name:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 20, None],
                'model__min_samples_split': [2, 5, 10]
            }
        elif 'GradientBoosting' in name:
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 10]
            }
        elif 'LogisticRegression' in name or 'LinearRegression' in name:
            param_grid = {} # Linear models have fewer knobs, key one is regularization (C, alpha) but keeping simple for now.

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # 🚀 The Hyper-Tuner: Tests 5 random combinations of parameters
        # n_iter=5 to keep it fast for demo, can be increased.
        if param_grid:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=5,       
                cv=3,            # 3-fold cross validation
                scoring=scoring,
                random_state=42,
                n_jobs=-1        
            )
            search.fit(X, y)
            best_tuned_pipeline = search.best_estimator_
            # best_score_ is mean cross-validated score of the best_estimator
            score = search.best_score_
        else:
            # Fallback for models without params defined
            cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring=scoring)
            score = np.mean(cv_scores)
            best_tuned_pipeline = pipeline
            best_tuned_pipeline.fit(X, y)

        
        # Adjust score sign for regression (neg_mae -> mae)
        if is_regression:
            logged_score = -score
        else:
            logged_score = score
            
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
                best_pipeline = best_tuned_pipeline
        else:
            # Higher F1 is better
            if logged_score > best_score:
                best_score = logged_score
                best_model_name = name
                best_pipeline = best_tuned_pipeline

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
    # However,    # 5. Final Fit
    # Fit the best model on the training data
    # Note: best_pipeline is already fitted from the search process.
    # However, fit(X, y) might be safer to ensure it sees ALL data if CV was used?
    # RandomizedSearchCV keeps the best estimator fitted on the whole X,y passed to fit() if refit=True (default).
    # So we don't strictly need to refit. But let's leave it if users want to be sure.
    # best_pipeline.fit(X, y) 
    pass
    
    # ==========================================
    # 🧠 XAI: FEATURE IMPORTANCE EXTRACTION
    # ==========================================
    try:
        # import numpy as np # Removed local import to avoid shadowing
        pass
        
        # 1. Isolate the two steps of your pipeline: Preprocessor and Model
        preprocessor_step = best_pipeline.named_steps['preprocessor']
        model_step = best_pipeline.steps[-1][1] 
        
        # 2. Extract feature names post-transformation 
        # (This handles the fact that One-Hot Encoding turns 1 column into many)
        feature_names = preprocessor_step.get_feature_names_out()
        
        # 3. Extract raw weights based on the type of algorithm
        importances = None
        if hasattr(model_step, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting) use feature_importances_
            importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'):
            # Linear models (Logistic/Linear Regression) use coefficients.
            # We use absolute value (np.abs) because a massive negative coefficient 
            # is just as important as a positive one!
            coefs = model_step.coef_
            if len(coefs.shape) > 1:  # Fix: Average across all classes for multi-class
                importances = np.abs(coefs).mean(axis=0)  
            else:
                importances = np.abs(coefs)
        
        # 4. Map the names to the weights, sort them, and save the top 10
        if importances is not None and len(feature_names) == len(importances):
            # Zip the names and weights together, and sort them highest to lowest
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            
            # Clean up the names (Scikit-learn adds ugly prefixes like 'cat__' or 'num__')
            clean_feat_imp = {k.split('__')[-1]: float(v) for k, v in feat_imp}
            
            # Save it to the system's memory!
            state.feature_importance = clean_feat_imp
            
    except Exception as e:
        # If extraction fails, log it silently. Never crash the main pipeline!
        state.warnings.append(f"Could not extract feature importance: {str(e)}")

    return best_pipeline
