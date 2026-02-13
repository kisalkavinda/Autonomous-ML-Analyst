import pandas as pd

def clean_chocolate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes the Chocolate Sales dataset before it hits the ML Pipeline.
    Fixes string-formatted currency and explodes Date into temporal features.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe with numeric Amount (dropped) and decomposed Date.
    """
    df_clean = df.copy() # Prevent pandas SettingWithCopy warnings

    # 1. Fix the 'Amount' Column (if it exists)
    if 'Amount' in df_clean.columns:
        # Strip the '$' and ',' characters, then cast to a mathematical float
        # We process it first to ensure we can check it, but then we DROP it to prevent leakage.
        try:
            df_clean['Amount'] = df_clean['Amount'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
        except Exception:
            pass # If it fails, maybe it's already numeric or garbage, but we are dropping it anyway.
        
        # 🚨 CRITICAL DATA LEAKAGE PREVENTION 🚨
        # We drop 'Amount' because it likely contains the answer (Revenue) which is directly 
        # correlated to 'Boxes Shipped' * Price. Predicting Volume from Revenue is cheating.
        df_clean = df_clean.drop(columns=['Amount'])

    # 2. Fix the 'Date' Column (if it exists)
    if 'Date' in df_clean.columns:
        # Convert from String to actual Datetime object (matching the DD/MM/YYYY format)
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Extract meaningful numeric features that Machine Learning models can understand
        df_clean['Month'] = df_clean['Date'].dt.month
        df_clean['Quarter'] = df_clean['Date'].dt.quarter
        df_clean['Year'] = df_clean['Date'].dt.year
        
        # Drop the original String 'Date' column to prevent the massive one-hot encoding bug
        df_clean = df_clean.drop(columns=['Date'])

    return df_clean
