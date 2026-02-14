
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import clean_dataset

def test_clean_dataset():
    # Test Data
    data = {
        'Price': ['$1,000.00', '€200.50', '¥3000', '$50.00', '$10.00', '$5.00'],
        'Date': ['01/01/2023', '15/05/2023', '31/12/2023', 'invalid', '01/01/2024', '02/02/2024'],
        'Other': ['A', 'B', 'C', 'D', 'E', 'F']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Assertions
    assert pd.api.types.is_numeric_dtype(cleaned_df['Price']), "Price should be numeric"
    assert cleaned_df['Price'].iloc[0] == 1000.0, "Price cleaning failed"
    
    assert 'Date_Month' in cleaned_df.columns, "Date_Month should exist"
    assert 'Date_Year' in cleaned_df.columns, "Date_Year should exist"
    assert 'Date' not in cleaned_df.columns, "Original Date column should be dropped"
    
    print("\n✅ clean_dataset verification passed!")

if __name__ == "__main__":
    test_clean_dataset()
