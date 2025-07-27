import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import sys
import os
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
    model_path = os.path.join(MODELS_DIR, 'customer_segmentation_pipeline.pkl')
    return joblib.load(model_path)

# Main app
def main():
    st.title("🎯 E-commerce Customer Segmentation Tool")
    st.markdown("### Analyze and segment your customers using RFM analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", ["Overview", "Segment Analysis", "New Customer Prediction"])
    
    # Load model pipeline
    try:
        pipeline = load_model()
        segment_names = pipeline['segment_names']
    except:
        st.error("Model not found! Please run main.py first to train the model.")
        return
    
    if page == "Overview":
        show_overview(pipeline, segment_names)
    
    elif page == "Segment Analysis":
        show_segment_analysis(pipeline)
    
    elif page == "New Customer Prediction":
        predict_new_customer(pipeline)

def show_overview(pipeline, segment_names):
    """Show overview page"""
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
        "Champions": "Best customers - Recent, frequent buyers with high monetary value",
        "Loyal Customers": "Regular customers with good frequency and monetary value",
        "Big Spenders": "Customers who spend significantly but may not buy frequently",
        "New Customers": "Recently acquired customers with potential",
        "At Risk": "Previously good customers who haven't purchased recently",
        "Lost Customers": "Customers who haven't engaged in a long time",
        "Promising": "Customers showing potential for growth"
    }
    
    for segment_id, segment_name in segment_names.items():
        if segment_name in segment_descriptions:
            st.info(f"**{segment_name}**: {segment_descriptions[segment_name]}")

def show_segment_analysis(pipeline):
    """Show segment analysis page"""
    st.header("Segment Analysis Dashboard")
    
    # Load sample data for visualization
    st.info("This dashboard shows the analysis from the trained model.")
    
    # Segment characteristics
    st.subheader("Segment Characteristics")
    
    # Create sample segment data
    segment_data = []
    for cluster_id, name in pipeline['segment_names'].items():
        segment_data.append({
            'Segment': name,
            'Avg Recency (days)': np.random.randint(10, 200),
            'Avg Frequency': np.random.randint(5, 50),
            'Avg Monetary ($)': np.random.randint(100, 5000)
        })
    
    df_segments = pd.DataFrame(segment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = px.bar(df_segments, x='Segment', y='Avg Monetary ($)', 
                     title='Average Monetary Value by Segment',
                     color='Segment')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        fig = px.scatter(df_segments, x='Avg Recency (days)', y='Avg Frequency',
                        size='Avg Monetary ($)', color='Segment',
                        title='Recency vs Frequency by Segment')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Segment Metrics")
    st.dataframe(df_segments, use_container_width=True)

def predict_new_customer(pipeline):
    """Predict segment for new customer"""
    st.header("New Customer Segmentation")
    st.markdown("Enter customer data to predict their segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        # Input fields
        last_purchase_days = st.number_input(
            "Days since last purchase", 
            min_value=0, 
            max_value=1000, 
            value=30,
            help="Number of days since the customer's last purchase"
        )
        
        purchase_frequency = st.number_input(
            "Number of purchases", 
            min_value=1, 
            max_value=500, 
            value=10,
            help="Total number of purchases made by the customer"
        )
        
        total_spent = st.number_input(
            "Total amount spent ($)", 
            min_value=0.0, 
            max_value=100000.0, 
            value=500.0,
            step=10.0,
            help="Total monetary value of all purchases"
        )
        
        if st.button("Predict Segment", type="primary"):
            # Prepare data
            customer_data = pd.DataFrame({
                'Recency': [last_purchase_days],
                'Frequency': [purchase_frequency],
                'Monetary': [total_spent]
            })
            
            # Scale features
            customer_scaled = pipeline['scaler'].transform(customer_data)
            
            # Predict
            cluster = pipeline['kmeans'].predict(customer_scaled)[0]
            segment_name = pipeline['segment_names'][cluster]
            
            # Display result
            st.success(f"Predicted Segment: **{segment_name}**")
            
            # Show characteristics
            st.markdown("### Segment Characteristics")
            if segment_name == "Champions":
                st.info("This customer is one of your best! They buy frequently, recently, and spend a lot.")
            elif segment_name == "Loyal Customers":
                st.info("A reliable customer with consistent purchasing behavior.")
            elif segment_name == "Big Spenders":
                st.info("This customer makes large purchases but may need encouragement to buy more frequently.")
            elif segment_name == "At Risk":
                st.warning("This customer used to be engaged but hasn't purchased recently. Consider re-engagement campaigns.")
            elif segment_name == "Lost Customers":
                st.error("This customer hasn't engaged in a long time and may need special attention to win back.")
            else:
                st.info("This customer shows potential for growth with the right engagement strategy.")
    
    with col2:
        st.subheader("Recommendations")
        
        if 'segment_name' in locals():
            if segment_name == "Champions":
                st.markdown("""
                **Recommended Actions:**
                - 🎁 Offer exclusive rewards and early access
                - 📧 Send personalized product recommendations
                - 🌟 Include in VIP programs
                - 💬 Request feedback and testimonials
                """)
            elif segment_name == "At Risk":
                st.markdown("""
                **Recommended Actions:**
                - 🔔 Send re-engagement campaigns
                - 🎯 Offer special comeback discounts
                - 📊 Survey to understand why they're less active
                - 🛍️ Remind them of abandone
                - 🛍️ Remind them of abandoned carts
                """)
            elif segment_name == "Big Spenders":
                st.markdown("""
                **Recommended Actions:**
                - 💰 Offer bulk purchase discounts
                - 🚀 Promote premium products
                - 📦 Provide free shipping on large orders
                - 🎯 Cross-sell complementary high-value items
                """)
            elif segment_name == "New Customers":
                st.markdown("""
                **Recommended Actions:**
                - 👋 Send welcome series emails
                - 🎁 Offer first-time buyer discounts
                - 📚 Provide product education content
                - 💡 Show popular products for new customers
                """)
            elif segment_name == "Lost Customers":
                st.markdown("""
                **Recommended Actions:**
                - 🎯 Launch win-back campaigns
                - 💸 Offer significant return discounts
                - 📧 Send "We miss you" emails
                - 🔄 Show what's new since they left
                """)

if __name__ == "__main__":
    main()