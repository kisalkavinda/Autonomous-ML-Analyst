from src.preprocessing import detect_imbalance
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, VotingRegressor, VotingClassifier
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
        scoring = 'neg_mean_absolute_error'
        
        param_distributions = {
            'RandomForestRegressor': {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'GradientBoostingRegressor': {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7, 9],
                'model__subsample': [0.8, 0.9, 1.0]
            },
            'LinearRegression': {}
        }
    else:
        models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }
        metric_name = 'F1-Macro'
        scoring = 'f1_macro'
        
        param_distributions = {
            'RandomForestClassifier': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'model__C': [0.1, 1.0, 10.0]
            }
        }

    best_score = -float('inf') if not is_regression else float('inf')
    best_model_name = None
    best_pipeline = None
    
    # Store all trained models for ensemble
    trained_models = []

    # 3. Evaluation Strategy
    # Using RandomizedSearchCV for Autotuning
    
    # 🚀 Check for Imbalance
    needs_smote = detect_imbalance(y, is_regression)

    for name, model in models.items():
        # Define hyperparameter grid
        param_grid = param_distributions.get(name, {})

        # 🚀 Conditionally build pipeline steps
        steps = [('preprocessor', preprocessor)]
        
        # Helper to set class_weight if applicable
        def set_class_weight(mdl, weight):
            if hasattr(mdl, 'class_weight'):
                mdl.set_params(class_weight=weight)
        
        if needs_smote:
            # Dynamic k_neighbors to prevent crashes on small datasets
            min_class_samples = y.value_counts().min()
            
            # Strategy:
            # 1. If very small (< 50 samples in minority), SMOTE is risky. Use class_weight='balanced'.
            # 2. If larger, use SMOTE.
            if min_class_samples < 50:
                 # Too few samples for SMOTE, use class weights
                 set_class_weight(model, 'balanced')
                 # Also remove 'smote' if it was added? We haven't added it yet.
                 # Logging? We might want to log this decision.
            else:
                # k_neighbors = min(5, max(1, min_class_samples - 2))
                k_neighbors = min(5, max(1, min_class_samples - 2))
                steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
                
        steps.append(('model', model))
        
        pipeline = Pipeline(steps)
        
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
            try:
                search.fit(X, y)
                best_tuned_pipeline = search.best_estimator_
                # best_score_ is mean cross-validated score of the best_estimator
                score = search.best_score_
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
                continue
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
        
        # Store for ensemble
        trained_models.append((name, best_tuned_pipeline, score))
        
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



    # 4. Ensemble Creation (Top 3 Models)
    # Sort by score (descending because both neg_mae and f1 are higher=better)
    trained_models.sort(key=lambda x: x[2], reverse=True)
    
    print(f"DEBUG: Trained models count: {len(trained_models)}")
    
    if len(trained_models) >= 2:
        top_3 = trained_models[:3]
        print(f"DEBUG: Top 3 models: {[m[0] for m in top_3]}")
        estimators = [(name, pipeline.named_steps['model']) for name, pipeline, score in top_3]
        
        # Note: We need to handle the preprocessor. 
        # VotingRegressor expects estimators. If we wrap them in pipelines, it works? 
        # Yes, VotingEstimator can take Pipelines.
        # But here we have the same preprocessor for all.
        # Efficient way: Voting on the MODELS, inside a single Pipeline.
        # BUT models might need different SMOTE/Preprocessing parameters if we tuned those? 
        # We didn't tune preprocessor. So we can share it.
        
        try:
            if is_regression:
                ensemble = VotingRegressor(estimators=estimators)
            else:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                
            # Create pipeline for ensemble
            ensemble_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            # Add SMOTE if needed (must be consistent)
            if needs_smote:
                # Re-use the smote step from the best pipeline? 
                # Or just create new one with same logic.
                min_class_samples = y.value_counts().min()
                if min_class_samples >= 6:
                    k_neighbors = min(5, max(1, min_class_samples - 2))
                    ensemble_pipeline.steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
            
            ensemble_pipeline.steps.append(('model', ensemble))
            
            # Evaluate Ensemble
            # Note: VotingClassifier needs to be fitted. 
            # We can't use the already-fitted estimators directly in sklearn < 1.0 easily in a Pipeline without re-fitting.
            # So we will let cross_val_score fit it again.
            
            cv_scores = cross_val_score(ensemble_pipeline, X, y, cv=3, scoring=scoring)
            ensemble_score = np.mean(cv_scores)
            
            # Log score
            # Check if it beats the best single model
            # For regression, we store NEGATIVE MAE in model_scores for consistency in "higher is better" sorting?
            # Wait, earlier we did: state.model_scores[name][metric_name] = -score (if is_reg func)
            # So logged_score for reg matches "lower is better" if we look at absolute value? 
            # Actually, `best_score` was tracked as the raw score (negative MAE) or positive?
            # Let's look at loop: if is_regression: logged_score = -score. if logged_score < best_score: best_score = logged_score.
            # So best_score is POSITIVE MAE (lower is better).
            
            # Here ensemble_score is from cross_val_score(scoring='neg_mean_absolute_error').
            # So ensemble_score is typically negative (e.g. -0.2).
            
            if is_regression:
                 # logged_ae = -ensemble_score => Positive MAE
                 ensemble_mae = -ensemble_score
                 state.model_scores["Ensemble (Top 3)"] = {metric_name: ensemble_mae}
                 
                 if ensemble_mae < best_score:
                     best_score = ensemble_mae
                     best_model_name = "Ensemble (Top 3)"
                     best_pipeline = ensemble_pipeline
                     state.preprocessing_steps.append("🏆 Ensemble (Top 3) outperformed single models.")
            else:
                 state.model_scores["Ensemble (Top 3)"] = {metric_name: ensemble_score}
                 if ensemble_score > best_score:
                     best_score = ensemble_score
                     best_model_name = "Ensemble (Top 3)"
                     best_pipeline = ensemble_pipeline
                     state.preprocessing_steps.append("🏆 Ensemble (Top 3) outperformed single models.")
                
        except Exception as e:
            print(f"Ensemble failed: {e}")

    # 5. State Mutation
    state.selected_model = best_model_name

    # 5. Final Fit
    # Fit the winning pipeline on the entire dataset
    # Note: best_pipeline might be already fitted if we used train_test_split, but fitting on full X,y is requested.
    # For CV loop, pipeline wasn't fitted on full data yet.
    # Re-create the best pipeline from scratch to ensure clean state or just call fit?
    # Calling fit on existing pipeline object is fine.
    
    # We need to retrieve the best model instance again to ensure a fresh fit?
    # Actually, reusing the pipeline object and calling fit(X, y) will re-train it.
    # 5. Final Fit
    # Fit the best model on the full training data to ensure it's ready for inference/SHAP
    best_pipeline.fit(X, y) 
    
    # ==========================================
    # 🧠 XAI: SHAP & FEATURE IMPORTANCE
    # ==========================================
    try:
        winning_model = best_pipeline.named_steps['model']
        
        # SHAP relies on the transformed data matrix (after preprocessing)
        # Note: We transform X using the preprocessor step.
        X_transformed = best_pipeline.named_steps['preprocessor'].transform(X)
        
        # Get feature names from preprocessor (works cleanly in scikit-learn >= 1.2)
        try:
            feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
        except:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
            
        if hasattr(winning_model, 'estimators_') and not isinstance(winning_model, (VotingRegressor, VotingClassifier)): # Tree-based models only (Random Forest, GBM)
            # Use SHAP
            explainer = shap.TreeExplainer(winning_model)
            # Sampling for speed if dataset is huge? For now full data.
            # TreeExplainer is fast.
            shap_values = explainer.shap_values(X_transformed)
            
            # Handle SHAP shape differences between Binary/Multiclass/Regression
            if isinstance(shap_values, list): # Older SHAP Binary/Multiclass or current Classifier
                # Average across ALL classes dimensions (0 is typically class index in list)
                # shap_values is a list of arrays [ (samples, features), (samples, features) ... ]
                # We want mean absolute value across all classes and samples
                
                # Stack to (classes, samples, features)
                shap_array = np.array(shap_values)
                # Mean across classes (axis 0) then samples (axis 1)
                importance = np.abs(shap_array).mean(axis=(0, 1))
                
            elif len(np.array(shap_values).shape) == 3: # Newer SHAP Multiclass (samples, features, classes)
                # Mean across samples (0) and classes (2)
                importance = np.abs(shap_values).mean(axis=(0, 2))
            else: # Regression (samples, features)
                importance = np.abs(shap_values).mean(axis=0)
                
            # Map values to feature names and sort
            shap_dict = dict(zip(feature_names, importance))
            # Save top 15 features
            state.feature_importance = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)[:10])
            
        else:
            # Fallback to Coefficient/Feature Importance if SHAP not applicable (Linear Models)
            importances = None
            if hasattr(winning_model, 'feature_importances_'):
                 importances = winning_model.feature_importances_
            elif hasattr(winning_model, 'coef_'):
                coefs = winning_model.coef_
                if len(coefs.shape) > 1:
                    importances = np.abs(coefs).mean(axis=0)  
                else:
                    importances = np.abs(coefs)
            
            if importances is not None and len(feature_names) == len(importances):
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                state.feature_importance = {k.split('__')[-1]: float(v) for k, v in feat_imp}

    except Exception as e:
        state.warnings.append(f"Could not extract feature importance/SHAP: {str(e)}")

    return best_pipeline

def predict_with_confidence(pipeline, X, confidence=0.95):
    """
    Predicts with confidence intervals for regression models.
    Returns (predictions, lower_bound, upper_bound).
    For non-probabilistic models, returns predictions for all.
    """
    preds = pipeline.predict(X)
    return preds, preds, preds
