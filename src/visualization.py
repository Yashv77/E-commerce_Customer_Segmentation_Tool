
## 6. Visualization (`src/visualization.py`)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

COLORS = px.colors.qualitative.Set3

def plot_elbow_curve(K_range, inertias, silhouette_scores):
    """Plot elbow curve and silhouette scores"""
    fig = go.Figure()
    # Elbow curve
    fig.add_trace(go.Scatter(
        x=K_range, y=inertias,
        mode='lines+markers',
        name='Inertia',
        yaxis='y'
    ))
    # Silhouette scores
    fig.add_trace(go.Scatter(
        x=K_range, y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        yaxis='y2'
    ))
    fig.update_layout(
        title='Elbow Method & Silhouette Analysis',
        xaxis_title='Number of Clusters',
        yaxis=dict(title='Inertia', side='left'),
        yaxis2=dict(title='Silhouette Score', side='right', overlaying='y'),
        hovermode='x unified'
    )
    return fig

def plot_clusters_pca(pca_features, clusters, segment_names):
    """Plot clusters using PCA"""
    df_plot = pd.DataFrame({
        'PC1': pca_features[:, 0],
        'PC2': pca_features[:, 1],
        'Cluster': clusters
    })
    # Map cluster numbers to names
    df_plot['Segment'] = df_plot['Cluster'].map(segment_names)
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', 
        color='Segment',
        title='Customer Segments (PCA Visualization)',
        color_discrete_sequence=COLORS
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component'
    )
    return fig

def plot_segment_distribution(segment_counts, segment_names):
    """Plot segment distribution"""
    df_plot = pd.DataFrame({
        'Segment': [segment_names[i] for i in segment_counts.index],
        'Count': segment_counts.values
    })
    fig = px.pie(
        df_plot, values='Count', names='Segment',
        title='Customer Segment Distribution',
        color_discrete_sequence=COLORS
    )
    return fig

def plot_rfm_analysis(rfm_df, clusters, segment_names):
    """Plot RFM analysis"""
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = clusters
    rfm_with_clusters['Segment'] = rfm_with_clusters['Cluster'].map(segment_names)
    # Create 3D scatter plot
    fig = px.scatter_3d(
        rfm_with_clusters, 
        x='Recency', y='Frequency', z='Monetary',
        color='Segment',
        title='3D RFM Analysis by Segment',
        color_discrete_sequence=COLORS
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    return fig

def plot_segment_profiles(segment_profiles, segment_names):
    """Plot segment profiles as radar chart"""
    categories = ['Recency', 'Frequency', 'Monetary']
    fig = go.Figure()
    for cluster_id in segment_profiles.index:
        values = []
        for cat in categories:
            # Normalize values for better visualization
            mean_val = segment_profiles.loc[cluster_id, (cat, 'mean')]
            max_val = segment_profiles[(cat, 'mean')].max()
            if cat == 'Recency':
                # Invert recency (lower is better)
                normalized = 1 - (mean_val / max_val)
            else:
                normalized = mean_val / max_val
            values.append(normalized)
        values.append(values[0])  # Close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=segment_names[cluster_id]
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Segment Profiles (Normalized)"
    )
    return fig