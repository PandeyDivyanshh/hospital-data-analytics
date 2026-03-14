"""
Data Cleaning Module for Hospital Patient Data
"""
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the hospital patient dataset.
    
    Steps:
    1. Remove duplicates.
    2. Convert date columns to datetime.
    3. Handle missing values.
    
    Args:
        df: Raw pandas DataFrame.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    # 1. Remove duplicates
    df_cleaned = df.drop_duplicates().copy()
    
    # 2. Convert date columns
    df_cleaned['Admission Date'] = pd.to_datetime(df_cleaned['Admission Date'])
    df_cleaned['Discharge Date'] = pd.to_datetime(df_cleaned['Discharge Date'])
    
    # Calculate average hospital stay
    df_cleaned['Hospital Stay Days'] = (df_cleaned['Discharge Date'] - df_cleaned['Admission Date']).dt.days
    
    # 3. Handle missing values
    # Fill missing Age with median
    if 'Age' in df_cleaned.columns:
        median_age = df_cleaned['Age'].median()
        df_cleaned['Age'] = df_cleaned['Age'].fillna(median_age)
        
    # Fill missing Disease with 'Unknown'
    if 'Disease' in df_cleaned.columns:
        df_cleaned['Disease'] = df_cleaned['Disease'].fillna('Unknown')
        
    # Drop remaining rows with crucial missing values if any
    df_cleaned = df_cleaned.dropna(subset=['Patient ID', 'Admission Date'])
    
    return df_cleaned

if __name__ == '__main__':
    # Test the module
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data/hospital_patient_data.csv')
    df = pd.read_csv(data_path)
    df_clean = clean_data(df)
    print("Data cleaned successfully.")
    print(df_clean.info())
