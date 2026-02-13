import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.config import AnalysisConfig
from src.state import AnalysisState
from src.exceptions import AnalysisError, InsufficientDataError, TargetConstantError, ExcessiveMissingDataError
from src.data_validator import validate_dataset
from src.preprocessing import build_preprocessor
from src.model_trainer import run_experiment
from src.report_generator import generate_markdown_report

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Autonomous ML Analyst", layout="wide")

# Title and description
st.title("🤖 Autonomous ML Analyst")
st.markdown("""
This system automatically analyzes your dataset, cleans it, trains multiple models, 
and selects the best one for your target variable.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Target selection
        target_col = st.selectbox("Select Target Column", df.columns)
        
        if st.button("Run Autonomous Analysis"):
            # Configuration and State
            config = AnalysisConfig()
            state = AnalysisState()
            
            try:
                # 1. Validation
                with st.spinner("Validating data..."):
                    df_clean = validate_dataset(df, target_col, config, state)
                
                # 2. Preprocessing
                with st.spinner("Building preprocessing engine..."):
                    preprocessor, X, y = build_preprocessor(df_clean, target_col, state)
                
                # 3. Model Training
                with st.spinner("Training candidate models..."):
                    best_model = run_experiment(X, y, preprocessor, config, state)
                
                # 4. Reporting
                with st.spinner("Generating report..."):
                    report = generate_markdown_report(state, target_col)
                    st.markdown(report)
                    
            except InsufficientDataError as e:
                st.error(f"🛑 Data Error: {e}")
            
            except TargetConstantError as e:
                st.error(f"🛑 Target Error: {e}")
                
            except ExcessiveMissingDataError as e:
                st.error(f"🛑 Quality Error: {e}")
                
            except AnalysisError as e:
                st.warning(f"⚠️ Analysis Interrupted: {e}")
                
            except Exception as e:
                # Log the actual error for debugging if needed, but show generic message to user
                print(f"DEBUG: Unexpected error: {e}")
                st.error("🚨 An unexpected system failure occurred. Please check your dataset formatting.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
