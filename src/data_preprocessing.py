import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Load the e-commerce dataset"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='latin-1')
    else:
        df = pd.read_excel(filepath)
    return df

def clean_data(df):
    """Clean the dataset"""
    # Initial shape
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove negative quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert CustomerID to string
    df['CustomerID'] = df['CustomerID'].astype(str).str.split('.').str[0]

    print(f"Final data shape: {df.shape}")
    return df

def create_rfm_features(df):
    """Create RFM features"""
    # Set reference date
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Reference date: {reference_date}")

    # Calculate RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    print(rfm.head())
    print(rfm.describe())
    print(rfm.info())
    
    
    # Remove outliers using IQR
    for col in rfm.columns:
        Q1 = rfm[col].quantile(0.25)
        Q3 = rfm[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        initial_count = len(rfm)
        rfm = rfm[(rfm[col] >= lower_bound) & (rfm[col] <= upper_bound)]
        removed = initial_count - len(rfm)
        if removed > 0:
            print(f"Removed {removed} outliers from {col}")

    print(f"Final RFM shape: {rfm.shape}")
    return rfm, reference_date