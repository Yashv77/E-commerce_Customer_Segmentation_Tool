import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_features(rfm_df):
    """Standardize RFM features"""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    return rfm_scaled, scaler

def apply_pca(scaled_features):
    """Apply PCA for visualization"""
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_variance}")
    return pca_features, pca

def create_rfm_scores(rfm_df):
    """Create RFM scores"""
    rfm_scores = rfm_df.copy()

    # Create quintile-based scores
    rfm_scores['R_Score'] = pd.qcut(rfm_scores['Recency'], 5, labels=[5,4,3,2,1])
    rfm_scores['F_Score'] = pd.qcut(rfm_scores['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm_scores['M_Score'] = pd.qcut(rfm_scores['Monetary'], 5, labels=[1,2,3,4,5])

    # Convert to int
    for col in ['R_Score', 'F_Score', 'M_Score']:
        rfm_scores[col] = rfm_scores[col].astype(int)

    # Combined score
    rfm_scores['RFM_Score'] = (
        rfm_scores['R_Score'].astype(str) + 
        rfm_scores['F_Score'].astype(str) + 
        rfm_scores['M_Score'].astype(str)
    )

    return rfm_scores