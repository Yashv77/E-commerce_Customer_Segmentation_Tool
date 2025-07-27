
# E-commerce Customer Segmentation Tool

This project is a machine learning-powered tool for segmenting e-commerce customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. It features an interactive Streamlit dashboard for exploring customer segments, visualizing data, and predicting the segment of new customers.

Streamlit Website: https://e-commerce-customer-segmentation-tool.streamlit.app/

---

## 🚀 Key Features

- **RFM Analysis**: Quantifies customer behavior using Recency, Frequency, and Monetary value.
- **K-Means Clustering**: Groups customers into meaningful segments with automatic optimal cluster selection.
- **Interactive Dashboard**: Built with Streamlit for easy exploration and visualization.
- **Customer Segment Prediction**: Predicts the segment for new or existing customers based on RFM input.
- **Comprehensive Visualizations**: Uses Plotly for dynamic charts (distribution, radar, PCA, etc.).
- **Actionable Recommendations**: Provides tailored marketing actions for each segment.

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit app
├── main.py                     # Model training and pipeline creation
├── config.py                   # Configuration (paths, constants)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   └── raw/
│       └── Online Retail.xlsx  # Raw e-commerce data
├── models/
│   └── customer_segmentation_pipeline.pkl  # Trained model pipeline
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning and RFM calculation
│   ├── feature_engineering.py  # Feature creation and scaling
│   ├── model_training.py       # Model training logic
│   └── visualization.py        # Plotly visualization functions
├── visualizations/             # Pre-generated HTML visualizations
│   ├── distribution.html
│   ├── elbow.html
│   ├── pca.html
│   ├── profiles.html
│   └── rfm_3d.html
└── __pycache__/
```

---

## 🛠️ How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YashV77/E-commerce_Customer_Segmentation_Tool.git
   cd E-commerce_Customer_Segmentation_Tool
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data:**
   - Place your raw e-commerce data (e.g., `Online Retail.xlsx`) in the `data/raw/` directory.

4. **Train the model pipeline:**
   ```bash
   python main.py
   ```
   This will generate `customer_segmentation_pipeline.pkl` in the `models/` directory.

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 💡 Using the Application

Once the app is running, you can:

- **Overview Tab**: View the segmentation approach, number of segments, and segment descriptions.
- **Segment Analysis Tab**: Explore segment distributions, radar charts, and detailed metrics for each segment.
- **New Customer Prediction Tab**: Enter RFM values for a new or existing customer to predict their segment and receive actionable recommendations.

### Example Use Cases

- Identify your best customers and tailor loyalty programs.
- Detect at-risk or lost customers and plan re-engagement campaigns.
- Segment new customers and design onboarding offers.

---


