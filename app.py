import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import sys

# --- MODIFICATION: Import the visualization script ---
from src import visualization

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR

# Page config
st.set_page_config(
    page_title="Customer Segmentation Tool",
    page_icon="🎯",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    """Load the pre-trained model pipeline from disk."""
    model_path = os.path.join(MODELS_DIR, 'customer_segmentation_pipeline.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# Main app
def main():
    """Main function to run the Streamlit application."""
    st.title("🎯 E-commerce Customer Segmentation Tool")
    st.markdown("### Analyze customer segments and predict behavior using RFM analysis")
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About the Project")
        st.info(
            "This app segments e-commerce customers using RFM analysis "
            "and a K-Means clustering model. It provides visualizations and "
            "actionable recommendations for each customer segment."
        )
        st.title("🛠️ Tech Stack")
        st.markdown(
            "- Python\n"
            "- Scikit-learn\n"
            "- Pandas\n"
            "- Streamlit\n"
            "- Plotly\n"
            "- Joblib"
        )
        st.markdown("---")
        st.markdown("Developed by [Yash Vardhan](https://yash-vardhan-portfolio-website.netlify.app/)")

    # Load model pipeline
    pipeline = load_model()
    if pipeline is None:
        st.error("Model not found! Please run the training script (e.g., main.py) first to generate the model file.")
        return
        
    segment_names = pipeline['segment_names']

    # --- MODIFICATION: Replaced sidebar and radio buttons with tabs ---
    tab1, tab2, tab3 = st.tabs(["Overview", "Segment Analysis", "New Customer Prediction"])

    with tab1:
        show_overview(pipeline, segment_names)

    with tab2:
        # Pass the entire pipeline to the analysis function
        show_segment_analysis(pipeline)

    with tab3:
        predict_new_customer(pipeline)

def show_overview(pipeline, segment_names):
    """Display the model overview and segment descriptions."""
    st.header("Customer Segmentation Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", pipeline['kmeans'].n_clusters)
    with col2:
        st.metric("Model Type", "K-Means Clustering")
    with col3:
        st.metric("Features Used", "RFM (Recency, Frequency, Monetary)")

    st.markdown("---")

    # Segment descriptions
    st.subheader("Segment Descriptions")
    segment_descriptions = {
        "Champions": "Your best customers. They buy recently, frequently, and spend the most.",
        "Loyal Customers": "Consistent and reliable customers with high frequency and good monetary value.",
        "Big Spenders": "Customers who spend a lot per transaction but may not be very frequent.",
        "New Customers": "Recently acquired customers with high potential for growth.",
        "Promising": "Newer customers with potential, showing good recency or frequency.",
        "At Risk": "Good customers who haven't purchased in a while and need re-engagement.",
        "Lost Customers": "Customers who have churned and have not engaged for a long time."
    }

    for segment_id, segment_name in sorted(segment_names.items()):
        description = segment_descriptions.get(segment_name, "No description available.")
        st.info(f"**{segment_name}**: {description}")

# --- MODIFICATION: Rewritten function to use visualizations from visualization.py ---
def show_segment_analysis(pipeline):
    """
    Shows segment analysis dashboard using visualizations from the visualization script.
    Relies on 'segment_profiles' and 'segment_counts' being present in the pipeline.
    """
    st.header("Segment Analysis Dashboard")

    # Check for required data in the pipeline
    if 'segment_profiles' not in pipeline or 'segment_counts' not in pipeline:
        st.error(
            "Analysis data ('segment_profiles' or 'segment_counts') not found in the model pipeline. "
            "Please ensure these are saved in the .pkl file during the training process."
        )
        return

    segment_profiles = pipeline['segment_profiles']
    segment_counts = pipeline['segment_counts']
    segment_names = pipeline['segment_names']

    st.info("This dashboard visualizes the characteristics of each customer segment based on the trained model.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Segment Distribution")
        fig_dist = visualization.plot_segment_distribution(segment_counts, segment_names)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader("Segment Profiles (Radar Chart)")
        fig_profiles = visualization.plot_segment_profiles(segment_profiles, segment_names)
        st.plotly_chart(fig_profiles, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Segment Metrics")
    st.dataframe(segment_profiles, use_container_width=True)


def predict_new_customer(pipeline):
    """UI for predicting the segment of a new or existing customer."""
    st.header("New Customer Segmentation")
    st.markdown("Enter customer RFM data to predict their segment and get recommendations.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        with st.form("prediction_form"):
            last_purchase_days = st.number_input(
                "Days since last purchase (Recency)", min_value=0, value=30,
                help="How many days ago was the customer's last purchase?"
            )
            purchase_frequency = st.number_input(
                "Number of purchases (Frequency)", min_value=1, value=10,
                help="How many purchases has the customer made in total?"
            )
            total_spent = st.number_input(
                "Total amount spent ($) (Monetary)", min_value=0.0, value=500.0, step=10.0,
                help="What is the total monetary value of all purchases?"
            )
            submitted = st.form_submit_button("Predict Segment", type="primary")

    if submitted:
        # Prepare data for prediction
        customer_data = pd.DataFrame({
            'Recency': [last_purchase_days],
            'Frequency': [purchase_frequency],
            'Monetary': [total_spent]
        })

        # Scale and predict
        customer_scaled = pipeline['scaler'].transform(customer_data)
        cluster = pipeline['kmeans'].predict(customer_scaled)[0]
        segment_name = pipeline['segment_names'][cluster]

        # Display result in the second column
        with col2:
            st.subheader("Prediction Result")
            st.success(f"Predicted Segment: **{segment_name}**")

            # Display segment characteristics
            st.markdown("##### Segment Characteristics")
            if segment_name == "Champions":
                st.info("✅ This customer is one of your best! They buy frequently, recently, and spend a lot.")
            elif segment_name == "Loyal Customers":
                st.info("✅ A reliable customer with consistent purchasing behavior.")
            elif segment_name == "Big Spenders":
                st.info("💰 This customer makes large purchases but may need encouragement to buy more frequently.")
            elif segment_name == "Promising":
                st.info("📈 This customer shows potential for growth with the right engagement strategy.")
            elif segment_name == "At Risk":
                st.warning("⚠️ This customer used to be engaged but hasn't purchased recently. Consider re-engagement.")
            elif segment_name == "Lost Customers":
                st.error("❌ This customer hasn't engaged in a long time and may need special attention.")
            
            # Display recommendations
            st.markdown("##### Recommended Actions")
            if segment_name == "Champions":
                st.markdown("- 🎁 Offer exclusive rewards and early access\n- 🌟 Include in VIP programs")
            elif segment_name == "At Risk":
                st.markdown("- 🔔 Send re-engagement campaigns\n- 🎯 Offer special comeback discounts")
            elif segment_name == "Big Spenders":
                st.markdown("- 💰 Offer bulk purchase discounts\n- 🚀 Promote premium products")
            elif segment_name == "New Customers":
                st.markdown("- 👋 Send welcome series emails\n- 🎁 Offer first-time buyer discounts")
            elif segment_name == "Lost Customers":
                st.markdown("- 💸 Launch win-back campaigns with significant discounts\n- 📧 Send 'We miss you' emails")
            # --- MODIFICATION: Added recommendations for 'Promising' segment ---
            elif segment_name == "Promising":
                st.markdown("- 🎯 Nurture with targeted offers to encourage a second purchase.\n- 📚 Introduce them to your loyalty program to build retention.")
    else:
        with col2:
            st.info("Enter customer data on the left and click 'Predict Segment' to see the results here.")


if __name__ == "__main__":
    main()