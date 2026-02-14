import pandas as pd
import numpy as np





def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detects and cleans common data issues (Currencies & Dates)."""
    df_clean = df.copy()
    
    # 1. Auto-detect and clean currency columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        sample = df_clean[col].dropna().astype(str).head(100)
        # Look for common currency symbols
        if sample.str.contains(r'[$£€¥]', regex=True).any():
            df_clean[col] = (df_clean[col].astype(str)
                           .str.replace(r'[$£€¥,]', '', regex=True)
                           .replace('', np.nan))
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 2. Auto-detect and parse date columns
    for col in df_clean.columns:
        # Only attempt on likely date columns to save compute
        if 'date' in str(col).lower() or 'time' in str(col).lower():
            for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']:
                try:
                    parsed = pd.to_datetime(df_clean[col], format=fmt, errors='coerce')
                    if parsed.notna().mean() > 0.8:  # 80% success rate threshold
                        df_clean[f'{col}_Month'] = parsed.dt.month
                        df_clean[f'{col}_Quarter'] = parsed.dt.quarter
                        df_clean[f'{col}_Year'] = parsed.dt.year
                        df_clean = df_clean.drop(columns=[col])
                        break # Found the right format, move to next column
                except:
                    continue
                    
    return df_clean

import joblib
import os
import json
from datetime import datetime

def save_model_package(pipeline, state, filepath="best_model.pkl"):
    joblib.dump(pipeline, filepath)
    meta_path = filepath.replace(".pkl", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": state.selected_model,
            "timestamp": datetime.now().isoformat()
        }, f)
    return filepath, meta_path
