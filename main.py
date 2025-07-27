import os
import pandas as pd
import joblib
from src.data_preprocessing import load_data, clean_data, create_rfm_features
from src.feature_engineering import scale_features, apply_pca, create_rfm_scores
from src.model_training import find_optimal_clusters, train_model, get_segment_profiles, save_model
from src.visualization import plot_elbow_curve, plot_clusters_pca, plot_segment_distribution, plot_rfm_analysis, plot_segment_profiles
from config import RAW_DATA_DIR, MODELS_DIR

def run_pipeline():
    """Run the complete customer segmentation pipeline (functional version)"""

    # 1. Load and preprocess data
    print("1. Loading data...")
    data_path = os.path.join(RAW_DATA_DIR, 'Online Retail.xlsx')
    df = load_data(data_path)

    print("2. Cleaning data...")
    df_clean = clean_data(df)

    print("3. Creating RFM features...")
    rfm_df, reference_date = create_rfm_features(df_clean)

    # 2. Feature engineering
    print("4. Scaling features...")
    rfm_scaled, scaler = scale_features(rfm_df)

    print("5. Applying PCA...")
    pca_features, pca = apply_pca(rfm_scaled)

    print("6. Creating RFM scores...")
    rfm_scores = create_rfm_scores(rfm_df)

    # 3. Model training
    print("7. Finding optimal clusters...")
    optimal_k, K_range, inertias, silhouette_scores = find_optimal_clusters(rfm_scaled, max_k=8)

    print(f"8. Training model with {optimal_k} clusters...")
    clusters, kmeans = train_model(rfm_scaled, n_clusters=optimal_k)

    print("9. Creating segment profiles...")
    segment_profiles, segment_names, segment_counts = get_segment_profiles(rfm_df, clusters)

    # 4. Save model and components
    print("10. Saving model pipeline...")
    model_pipeline = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'segment_names': segment_names,
        'reference_date': reference_date
    }
    model_path = os.path.join(MODELS_DIR, 'customer_segmentation_pipeline.pkl')
    save_model(model_pipeline, model_path)

    # 5. Generate visualizations
    print("11. Creating visualizations...")

    # Elbow curve
    elbow_fig = plot_elbow_curve(K_range, inertias, silhouette_scores)

    # PCA clusters
    pca_fig = plot_clusters_pca(pca_features, clusters, segment_names)

    # Segment distribution
    dist_fig = plot_segment_distribution(segment_counts, segment_names)

    # RFM 3D plot
    rfm_fig = plot_rfm_analysis(rfm_df, clusters, segment_names)

    # Segment profiles
    profile_fig = plot_segment_profiles(segment_profiles, segment_names)
        
    # Save visualizations to HTML files
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    vis_results= {
        'elbow': elbow_fig,
        'pca': pca_fig,
        'distribution': dist_fig,
        'rfm_3d': rfm_fig,
        'profiles': profile_fig 
    }
    for name, fig in vis_results.items():
        fig.write_html(os.path.join(vis_dir, f"{name}.html"))
    print(f"Visualizations saved to ./{vis_dir}/ as HTML files.")

    print("\nPipeline completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Number of segments: {optimal_k}")
    print("\nSegment distribution:")
    for cluster_id, count in segment_counts.items():
        print(f"  {segment_names[cluster_id]}: {count} customers")

    return {
        'rfm_df': rfm_df,
        'clusters': clusters,
        'segment_profiles': segment_profiles,
        'segment_names': segment_names,
        'visualizations': {
            'elbow': elbow_fig,
            'pca': pca_fig,
            'distribution': dist_fig,
            'rfm_3d': rfm_fig,
            'profiles': profile_fig
        }
    }

if __name__ == "__main__":
    results = run_pipeline()
