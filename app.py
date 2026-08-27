"""
app.py
California Housing Price Predictor - Interactive Web Application
Built with Streamlit and Scikit-Learn
Maincrafts Technology - AIML Task 1
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and metrics
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "linear_regression_model.pkl")
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
    
    bundle = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return bundle, metrics

try:
    bundle, metrics = load_model()
    model = bundle['model']
    feature_names = bundle['feature_names']
except Exception as e:
    st.error(f"Error loading model: {e}. Please run `python train_and_export.py` first.")
    st.stop()

# Header Section
st.title("🏠 California Housing Price Predictor")
st.markdown("""
**AIML Task 1: Linear Regression Model** | Built with `scikit-learn` & `Streamlit`  
Predict median district home values across California based on demographic and geospatial characteristics.
""")

# Sidebar - User Inputs
st.sidebar.header("⚙️ Housing District Parameters")
st.sidebar.markdown("Adjust the sliders below to estimate the district median house price:")

med_inc = st.sidebar.slider(
    "Median Income (MedInc) [$10k]",
    min_value=0.5, max_value=15.0, value=3.87, step=0.05,
    help="Median income in block group (e.g., 3.87 corresponds to $38,700/year)"
)

house_age = st.sidebar.slider(
    "Median House Age (HouseAge) [Years]",
    min_value=1.0, max_value=52.0, value=28.0, step=1.0,
    help="Median age of buildings in the district"
)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    ave_rooms = st.number_input(
        "Avg Rooms",
        min_value=1.0, max_value=20.0, value=5.43, step=0.1,
        help="Average number of rooms per household"
    )
with col_sb2:
    ave_bedrms = st.number_input(
        "Avg Bedrooms",
        min_value=0.5, max_value=10.0, value=1.10, step=0.05,
        help="Average number of bedrooms per household"
    )

population = st.sidebar.slider(
    "District Population",
    min_value=10, max_value=35000, value=1425, step=50,
    help="Total population count within the block group"
)

ave_occup = st.sidebar.slider(
    "Avg Occupants per Household (AveOccup)",
    min_value=1.0, max_value=15.0, value=3.07, step=0.1,
    help="Average number of people living in each home"
)

st.sidebar.subheader("📍 Geospatial Coordinates")
latitude = st.sidebar.slider(
    "Latitude (°N)",
    min_value=32.5, max_value=42.0, value=37.88, step=0.01,
    help="California latitude (e.g. SF: ~37.8, LA: ~34.05, San Diego: ~32.7)"
)

longitude = st.sidebar.slider(
    "Longitude (°W)",
    min_value=-124.35, max_value=-114.31, value=-122.23, step=0.01,
    help="California longitude (e.g. SF: ~ -122.4, LA: ~ -118.25)"
)

# Prepare feature dictionary and DataFrame
input_dict = {
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude
}
input_df = pd.DataFrame([input_dict])

# Compute Prediction
pred_value_100k = model.predict(input_df)[0]
# Prevent negative values for realism
pred_display_100k = max(0.0, pred_value_100k)
pred_usd = pred_display_100k * 100000.0

rmse_val = metrics.get('RMSE', 0.7456)
lower_bound_usd = max(0.0, (pred_display_100k - rmse_val) * 100000.0)
upper_bound_usd = (pred_display_100k + rmse_val) * 100000.0

# Top metrics cards layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted Median House Value",
        value=f"${pred_usd:,.0f}",
        delta=f"{pred_display_100k:.3f} ($100k)"
    )

with col2:
    st.metric(
        label="Typical Error Margin (±1 RMSE)",
        value=f"± ${rmse_val * 100000:,.0f}",
        help="Root Mean Squared Error of the model"
    )

with col3:
    st.metric(
        label="Model R² Accuracy (Test Set)",
        value=f"{metrics.get('R2_Test', 0.5758)*100:.1f}%",
        help="Proportion of variance explained by the Linear Regression model"
    )

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Geographic Location & Prediction", "📊 Model Diagnostics & Visuals", "📋 Input Summary & Coefficients"])

with tab1:
    col_map, col_details = st.columns([1.5, 1])
    
    with col_map:
        st.subheader("Selected California Location")
        map_df = pd.DataFrame({
            'lat': [latitude],
            'lon': [longitude]
        })
        st.map(map_df, zoom=7)
    
    with col_details:
        st.subheader("Prediction Breakdown")
        st.info(f"""
        **Estimated Price:** ${pred_usd:,.2f}  
        **Estimated Range (±1 RMSE):**  
        ${lower_bound_usd:,.0f} — ${upper_bound_usd:,.0f}
        
        **Location:** Lat: `{latitude:.2f}`, Lon: `{longitude:.2f}`  
        **Household Income:** `${med_inc * 10000:,.0f} / year`  
        **Rooms / Bedrooms:** `{ave_rooms:.1f} rooms` (`{ave_bedrms:.1f} bedrms`)  
        **Average Occupancy:** `{ave_occup:.1f} persons/home`
        """)
        
        # Proximity note
        if latitude > 37.0 and longitude < -121.5:
            st.success("📍 Bay Area / Northern Coastal Zone (High valuation corridor)")
        elif latitude < 35.0 and longitude < -117.5:
            st.success("📍 Greater Los Angeles / Southern Coast (High valuation corridor)")
        else:
            st.warning("📍 Inland / Central Valley / Rural California Region")

with tab2:
    st.subheader("Model Diagnostic Charts")
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.image(os.path.join(assets_dir, "actual_vs_predicted.png"), caption="Actual vs Predicted House Values", use_container_width=True)
    with c_img2:
        st.image(os.path.join(assets_dir, "feature_importance.png"), caption="Standardized Feature Coefficients (Impact)", use_container_width=True)

    c_img3, c_img4 = st.columns(2)
    with c_img3:
        st.image(os.path.join(assets_dir, "correlation_heatmap.png"), caption="Feature Correlation Matrix", use_container_width=True)
    with c_img4:
        st.image(os.path.join(assets_dir, "residuals_distribution.png"), caption="Error Distribution & Residuals Plot", use_container_width=True)

with tab3:
    st.subheader("Input Feature Summary")
    st.dataframe(input_df.T.rename(columns={0: "Input Value"}), use_container_width=True)
    
    st.subheader("Model Linear Equation Parameters")
    raw_coefs = metrics.get('Raw_Coefficients', {})
    coef_df = pd.DataFrame({
        "Feature": list(raw_coefs.keys()),
        "Raw Coefficient": list(raw_coefs.values()),
        "Standardized Impact": [metrics.get('Std_Coefficients', {}).get(k, 0.0) for k in raw_coefs.keys()]
    })
    st.dataframe(coef_df, use_container_width=True)
    st.caption(f"Intercept (theta_0): {metrics.get('Raw_Intercept', -37.023):.4f}")

# Footer
st.markdown("---")
st.caption("Maincrafts Technology • AI & Machine Learning Task 1: Linear Regression California Housing Predictor")
