import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import AnalysisError, InsufficientDataError, TargetConstantError, ExcessiveMissingDataError
from src.data_validator import validate_dataset, suggest_target_column
from src.preprocessing import build_preprocessor
from src.model_trainer import run_experiment
from src.report_generator import generate_markdown_report
from src.utils import clean_chocolate_data

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Autonomous ML Analyst", layout="wide", page_icon="🤖")

# Custom CSS for polished look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
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
            df = clean_chocolate_data(raw_df)

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
                preprocessor, X, y = build_preprocessor(df_clean, target_col, state)
                
                # 3. Model Training
                status_container.write("🤖 Training candidate models (RandomForest, GradientBoosting, etc.)...")
                best_model = run_experiment(X, y, preprocessor, config, state)
                
                status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                # Store results in session state for the Analysis tab
                st.session_state["analysis_report"] = generate_markdown_report(state, target_col)
                st.session_state["analysis_state"] = state
                
                # ==========================================
                # 💾 FREEZE PIPELINE
                # ==========================================
                if best_model:
                    st.session_state["trained_pipeline"] = best_model
                    st.session_state["feature_columns"] = list(X.columns)
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
                inf_df = clean_chocolate_data(raw_inf_df)

                # Feature Alignment
                required_features = st.session_state["feature_columns"]
                missing_features = set(required_features) - set(inf_df.columns)

                if missing_features:
                    st.error(f"❌ Missing required features: {missing_features}")
                else:
                    inf_df_aligned = inf_df[required_features]
                    pipeline = st.session_state["trained_pipeline"]
                    predictions = pipeline.predict(inf_df_aligned)

                    result_df = inf_df.copy()
                    result_df["Predicted_" + st.session_state["target_column"]] = predictions
                    
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


