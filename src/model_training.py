import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

def find_optimal_clusters(scaled_features, max_k=10):
    """Find optimal number of clusters"""
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_k + 1, len(scaled_features) // 2))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

    # Find optimal k using silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]

    return optimal_k, list(K_range), inertias, silhouette_scores

def train_model(scaled_features, n_clusters):
    """Train K-Means model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    print(f"Trained model with {n_clusters} clusters")
    return clusters, kmeans

def get_segment_profiles(rfm_df, clusters):
    """Create segment profiles"""
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = clusters

    # Calculate statistics
    segment_profiles = rfm_with_clusters.groupby('Cluster').agg({
        'Recency': ['mean', 'std', 'min', 'max'],
        'Frequency': ['mean', 'std', 'min', 'max'],
        'Monetary': ['mean', 'std', 'min', 'max']
    }).round(2)

    # Count customers per segment
    segment_counts = rfm_with_clusters['Cluster'].value_counts().sort_index()

    # Assign names
    segment_names = assign_segment_names(segment_profiles)

    return segment_profiles, segment_names, segment_counts

def assign_segment_names(profiles):
    """Assign meaningful names to segments, ensuring uniqueness."""
    segment_names = {}
    used_names = set()
    for i in profiles.index:
        r_mean = profiles.loc[i, ('Recency', 'mean')]
        f_mean = profiles.loc[i, ('Frequency', 'mean')]
        m_mean = profiles.loc[i, ('Monetary', 'mean')]
        r_score = 1 / (1 + r_mean / 100)  # Lower recency is better
        f_score = f_mean / profiles[('Frequency', 'mean')].max()
        m_score = m_mean / profiles[('Monetary', 'mean')].max()
        if r_score > 0.7 and f_score > 0.7 and m_score > 0.7:
            name = "Champions"
        elif r_score > 0.5 and f_score > 0.5 and m_score > 0.5:
            name = "Loyal Customers"
        elif m_score > 0.7:
            name = "Big Spenders"
        elif r_score > 0.7 and f_score < 0.3:
            name = "New Customers"
        elif r_score < 0.3 and f_score > 0.5:
            name = "At Risk"
        elif r_score < 0.3 and f_score < 0.3:
            name = "Lost Customers"
        else:
            name = "Promising"
        # Ensure uniqueness
        if name in used_names:
            name = f"{name} {i}"
        segment_names[i] = name
        used_names.add(name)
    return segment_names

def save_model(model_pipeline, filepath):
    """Save the complete model pipeline"""
    joblib.dump(model_pipeline, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load the model pipeline"""
    return joblib.load(filepath)
