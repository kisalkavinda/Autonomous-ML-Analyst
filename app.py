import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import AnalysisError, InsufficientDataError, TargetConstantError, ExcessiveMissingDataError
from src.data_validator import validate_dataset, suggest_target_column
from src.preprocessing import build_preprocessor, engineer_features
from src.model_trainer import run_experiment
from src.report_generator import generate_markdown_report
from src.utils import clean_dataset


# Load environment variables
load_dotenv()

def validate_inference_data(inf_df, feature_columns, training_stats):
    """
    Validate inference data matches training expectations.
    Checks: Missing columns, Data Types, Nulls, Value Ranges.
    """
    issues = []
    
    # 1. Check for missing columns
    # (Already handled by alignment, but good to double check)
    missing_features = set(feature_columns) - set(inf_df.columns)
    if missing_features:
        issues.append(f"❌ Missing required columns: {missing_features}")
    
    # 2. Check Data Types
    if 'dtypes' in training_stats:
        for col in feature_columns:
            if col not in inf_df.columns: continue
            
            train_dtype = training_stats['dtypes'].get(col)
            # Simple check: numeric vs object
            if train_dtype and pd.api.types.is_numeric_dtype(train_dtype):
                if not pd.api.types.is_numeric_dtype(inf_df[col]):
                    issues.append(f"⚠️ Type Mismatch: '{col}' should be numeric but is {inf_df[col].dtype}")

    # 3. Check for Nulls
    # Filter to only required columns
    inf_subset = inf_df[[c for c in feature_columns if c in inf_df.columns]]
    null_cols = inf_subset.isnull().any()
    if null_cols.any():
        issues.append(f"⚠️ Null values detected in: {list(null_cols[null_cols].index)}")
        
    # 4. Check Value Ranges (Drift Detection)
    if 'ranges' in training_stats:
        for col in feature_columns:
            if col in training_stats['ranges'] and col in inf_df.columns:
                train_min, train_max = training_stats['ranges'][col]
                if pd.api.types.is_numeric_dtype(inf_df[col]):
                     inf_min, inf_max = inf_df[col].min(), inf_df[col].max()
                     
                     # Simple heuristics for "wildly out of range"
                     # e.g. if new min is < 50% of old min (if positive) or > 200% of old max
                     # This is just a warning, not a blocker
                     if (train_min > 0 and inf_min < train_min * 0.5) or (inf_max > train_max * 2):
                         issues.append(
                            f"⚠️ Data Drift Warnings: '{col}' has unusual values "
                            f"(Train: {train_min:.2f}-{train_max:.2f}, Inf: {inf_min:.2f}-{inf_max:.2f})"
                        )
    
    return issues

# Page config
st.set_page_config(page_title="Autonomous ML Analyst", layout="wide", page_icon="🤖")

# Custom CSS for polished look
st.markdown("""
<style>
    /* Button Styling with Neon Glow */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.2em;
        font-weight: 600;
        background-color: transparent;
        border: 1px solid #3b82f6;
        color: #f8fafc;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.1);
    }
    
    .stButton>button:hover {
        background-color: #3b82f6;
        color: #ffffff;
        border: 1px solid #60a5fa;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }

    /* Container Borders & Shadows */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-color: #1e293b !important;
        border-radius: 10px !important;
        background-color: #0f172a !important;
        box-shadow: 0 4px 15px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* Input Boxes (File Uploader, Select Box) */
    .stSelectbox>div>div, .stFileUploader>div>div {
        background-color: #1e293b !important;
        border-radius: 6px;
    }

    /* Custom Gradient Main Header */
    .main-header {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.title("Control Panel")
    st.info("💡 **Tip:** Use the 'Lab' tab to train your model, then switch to the 'Factory' section below (once unlocked) to generate predictions.")
    
    st.divider()
    st.markdown("### System Status")
    if st.session_state.get("model_ready", False):
        st.success("✅ Model Ready")
        st.write(f"**Target:** `{st.session_state.get('target_column', 'N/A')}`")
        st.write(f"**Model:** `{st.session_state.get('trained_pipeline').steps[-1][1].__class__.__name__}`")
    else:
        st.warning("⚠️ No Model Trained")

# Main Header
st.markdown('<div class="main-header">🤖 Autonomous ML Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI-Powered Data Scientist</div>', unsafe_allow_html=True)

# ==========================================
# 🔬 LAB (Test/Train Mode)
# ==========================================
st.markdown("---")
st.header("🔬 Lab — Model Training")

# Tabs for better organization
tab_data, tab_train, tab_analysis = st.tabs(["📂 Data Upload", "⚙️ Training", "📊 Analysis Report"])

with tab_data:
    # File uploader for Training
    train_file = st.file_uploader("Upload your training dataset (CSV)", type=["csv"], key="train_uploader")

    df = None
    target_col = None

    if train_file is not None:
        try:
            # Add a toggle for headerless CSVs
            has_header = st.checkbox("My dataset has a header row", value=True)
            
            if has_header:
                raw_df = pd.read_csv(train_file)
            else:
                raw_df = pd.read_csv(train_file, header=None)
                raw_df.columns = [f"Feature_{i}" for i in range(raw_df.shape[1])]
            
            # 🧹 CLEANING INTERCEPTOR
            df = clean_dataset(raw_df)

            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            
            with st.expander("👀 View Raw Data"):
                st.dataframe(df.head(10), width="stretch")
            
            # Smart Defaulting
            suggested_target = suggest_target_column(df)
            default_index = list(df.columns).index(suggested_target) if suggested_target in df.columns else len(df.columns) - 1
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🎯 Target Variable")
                st.info("What do you want to predict?")
            with col2:
                target_col = st.selectbox("Select Target Column", df.columns, index=default_index)

        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_train:
    if df is not None:
        st.markdown("### 🚀 Launch Training")
        st.write("Click the button below to start the autonomous analysis pipeline. This will validate data, preprocess features, and train multiple models.")
        
        if st.button("Run Autonomous Analysis", type="primary"):
            # Configuration and State
            config = AnalysisConfig()
            state = AnalysisState()
            
            status_container = st.status("Running Autonomous Pipeline...", expanded=True)
            
            try:
                # 1. Validation
                status_container.write("🕵️ Validating dataset quality...")
                df_clean = validate_dataset(df, target_col, config, state)
                
                # 2. Preprocessing
                status_container.write("🔧 Building preprocessing engine...")
                
                # 🚀 INJECT FEATURE ENGINEERING HERE
                df_engineered = engineer_features(df_clean, target_col=target_col, state=state)
                
                preprocessor, X, y = build_preprocessor(df_engineered, target_col, state)
                
                # 3. Model Training
                status_container.write("🤖 Training candidate models (RandomForest, GradientBoosting, etc.)...")
                best_model = run_experiment(X, y, preprocessor, config, state)
                
                status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                # Store results in session state for the Analysis tab
                st.session_state["analysis_report"] = generate_markdown_report(state, target_col)
                st.session_state["analysis_state"] = state
                
                # NEW: Save training stats for factory mode
                # We need to capture the dtypes and ranges from the TRAINING data (X)
                training_stats = {
                    "dtypes": X.dtypes.to_dict(),
                    "ranges": {
                        col: (X[col].min(), X[col].max())
                        for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
                    }
                }
                st.session_state["training_stats"] = training_stats
                
                # ==========================================
                # 💾 FREEZE PIPELINE (PERSISTENCE)
                # ==========================================
                from src.utils import save_model_package
                try:
                    model_path, meta_path = save_model_package(best_model, state)
                    st.success(f"💾 Model Saved! ({model_path})")
                    # Ideally provide download button here
                    with open(model_path, "rb") as f:
                         st.download_button("📥 Download Model (.pkl)", f, file_name=os.path.basename(model_path))
                except Exception as e:
                    st.warning(f"⚠️ Could not save model to disk: {e}")
                
                 
                # Store results in session state
                st.session_state["trained_pipeline"] = best_model
                st.session_state["feature_columns"] = list(X.columns) # Use X columns (feature engineered)
                st.session_state["target_column"] = target_col
                st.session_state["model_ready"] = True
                st.session_state["model_score"] = state.model_scores[state.selected_model]
                # ==========================================
                if best_model:
                    st.session_state["trained_pipeline"] = best_model
                    st.session_state["feature_columns"] = list(X.columns)
                    st.session_state["feature_dtypes"] = X.dtypes.to_dict()
                    st.session_state["target_column"] = target_col
                    st.session_state["model_ready"] = True
                    st.balloons()
                    
            except InsufficientDataError as e:
                status_container.update(label="❌ Analysis Failed", state="error")
                st.error(f"🛑 Data Error: {e}")
            except TargetConstantError as e:
                status_container.update(label="❌ Analysis Failed", state="error")
                st.error(f"🛑 Target Error: {e}")
            except ExcessiveMissingDataError as e:
                status_container.update(label="❌ Analysis Failed", state="error")
                st.error(f"🛑 Quality Error: {e}")
            except AnalysisError as e:
                status_container.update(label="⚠️ Analysis Interrupted", state="error")
                st.warning(f"⚠️ Analysis Interrupted: {e}")
            except Exception as e:
                print(f"DEBUG: Unexpected error: {e}")
                status_container.update(label="🚨 System Error", state="error")
                st.error("🚨 An unexpected system failure occurred. Please check your dataset formatting.")
    else:
        st.info("👈 Please upload a dataset in the 'Data Upload' tab first.")

with tab_analysis:
    if "analysis_report" in st.session_state:
        st.markdown(st.session_state["analysis_report"])
        
        state = st.session_state["analysis_state"]
        
        # ==========================================
        # 📊 RENDER EXPLAINABLE AI CHART
        # ==========================================
        if state.feature_importance:
            st.divider()
            st.subheader("📊 Model Interpretability: Top Feature Drivers")
            st.write("This chart shows the absolute mathematical weight the winning model assigned to the top variables.")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            features = list(state.feature_importance.keys())
            scores = list(state.feature_importance.values())
            
            sns.barplot(x=scores, y=features, hue=scores, ax=ax, palette="mako", legend=False)
            ax.set_xlabel("Relative Importance")
            ax.set_ylabel("Feature Name")
            ax.set_title(f"Key Drivers for {state.selected_model}")
            sns.despine()
            st.pyplot(fig)
    else:
        st.info("Run the training pipeline to generate the analysis report.")

# ==========================================
# 🏭 FACTORY (Inference Mode)
# ==========================================
if st.session_state.get("model_ready", False):
    st.markdown("---")
    st.header("🏭 Worker Mode — Autonomous Inference")
    st.markdown("Use your trained model to generate predictions on new data.")

    with st.container(border=True):
        col_upload, col_action = st.columns([2, 1])
        
        with col_upload:
            inference_file = st.file_uploader(
                "📂 Upload new data for prediction (Test Set)",
                type=["csv"],
                key="inference_uploader"
            )

        if inference_file:
            try:
                raw_inf_df = pd.read_csv(inference_file)
                
                # 🧹 CLEANING INTERCEPTOR (Factory Mode)
                inf_df = clean_dataset(raw_inf_df)

                # 🚀 INJECT FEATURE ENGINEERING HERE
                inf_df = engineer_features(inf_df)

                # Feature Alignment
                required_features = st.session_state["feature_columns"]
                # Validate Schema
                training_stats = st.session_state.get("training_stats", {})
                validation_issues = validate_inference_data(inf_df, required_features, training_stats)
                
                if validation_issues:
                   for issue in validation_issues:
                       if "❌" in issue:
                           st.error(issue)
                       else:
                           st.warning(issue)
                
                # Proceed only if no fatal errors (missing features)
                if not any("❌" in issue for issue in validation_issues):
                    inf_df_aligned = inf_df[required_features]
                    pipeline = st.session_state["trained_pipeline"]
                    
                    # Detect if regression (target stored in state?)
                    # We can infere from pipeline model type
                    from src.model_trainer import predict_with_confidence
                    
                    # Simple check: is classifier?
                    is_classifier = hasattr(pipeline, 'classes_') or hasattr(pipeline.named_steps['model'], 'classes_')
                    
                    result_df = inf_df.copy()
                    target_col = st.session_state.get("target_column", "Target")
                    
                    if not is_classifier:
                        # Regression -> Try Confidence Intervals
                        preds, lower, upper = predict_with_confidence(pipeline, inf_df_aligned)
                        result_df[f"Predicted_{target_col}"] = preds
                        
                        # Only add intervals if they are different (meaning calculation succeeded)
                        if not np.array_equal(lower, preds):
                            result_df[f"{target_col}_Lower_95"] = lower
                            result_df[f"{target_col}_Upper_95"] = upper
                    else:
                        # Classification
                        predictions = pipeline.predict(inf_df_aligned)
                        result_df[f"Predicted_{target_col}"] = predictions
                        
                        # Add Probability if available
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                probs = pipeline.predict_proba(inf_df_aligned)
                                # Take max prob
                                result_df["Confidence_Score"] = probs.max(axis=1)
                            except:
                                pass

                    with col_action:
                        st.markdown("### ✅ Ready!")
                        st.metric("Rows Processed", len(result_df))
                        
                        csv_output = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Predictions",
                            csv_output,
                            "predictions.csv",
                            "text/csv",
                            type="primary"
                        )
                    
                    st.subheader("Results Preview")
                    st.dataframe(result_df.head(), width="stretch")

            except Exception as e:
                st.error(f"⚠️ Inference failed: {str(e)}")

else:
    # Placeholder to keep layout consistent but indicate locked state
    st.markdown("---")
    st.markdown("### 🔒 Factory Mode Locked")
    st.info("Train a model in the Lab above to unlock the Inference Factory.")


