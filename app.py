"""
app.py
California Housing Price Predictor - Interactive Web Application
AIML Task 1 & Task 2: Multi-Model Benchmark & Prediction System
Maincrafts Technology - AI & Machine Learning Internship
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="California Housing Price Predictor (Task 1 & 2)",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models and Metrics
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    assets_dir = os.path.join(base_dir, "assets")

    # Load Task 2 all models bundle
    task2_bundle_path = os.path.join(models_dir, "task2_all_models.pkl")
    task2_meta_path = os.path.join(models_dir, "task2_comparison_metrics.json")
    
    task2_bundle = joblib.load(task2_bundle_path)
    with open(task2_meta_path, "r") as f:
        task2_meta = json.load(f)

    return task2_bundle, task2_meta, assets_dir

try:
    bundle, meta, assets_dir = load_artifacts()
    all_models = bundle["models"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    results = meta["models"]
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `python task2_train_and_compare.py` first.")
    st.stop()

# Header Section
st.title("🏠 California Housing Price Predictor")
st.markdown("""
**AIML Tasks 1 & 2: Feature Scaling, Multi-Model Optimization & Comparison** | Maincrafts Technology  
Predict median district home values across California using multiple trained algorithms with `scikit-learn` & `Streamlit`.
""")

# Sidebar - Model Selection & User Inputs
st.sidebar.header("🤖 Algorithm Selection")
model_choice = st.sidebar.selectbox(
    "Active Prediction Model",
    options=list(all_models.keys()),
    index=2, # Default to Decision Tree (Best Core Model)
    help="Select which algorithm to use for primary prediction"
)

st.sidebar.header("⚙️ Housing District Parameters")
med_inc = st.sidebar.slider(
    "Median Income (MedInc) [$10k]",
    min_value=0.5, max_value=15.0, value=3.87, step=0.05,
    help="Median income in block group (e.g. 3.87 = $38,700/year)"
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
    help="Average number of residents per home"
)

st.sidebar.subheader("📍 Geospatial Coordinates")
latitude = st.sidebar.slider(
    "Latitude (°N)",
    min_value=32.5, max_value=42.0, value=37.88, step=0.01,
    help="California latitude (SF: ~37.88, LA: ~34.05)"
)

longitude = st.sidebar.slider(
    "Longitude (°W)",
    min_value=-124.35, max_value=-114.31, value=-122.23, step=0.01,
    help="California longitude (SF: ~ -122.23, LA: ~ -118.25)"
)

# Prepare input DataFrame
raw_input = pd.DataFrame([{
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude
}])

# Scale features
scaled_input = scaler.transform(raw_input)

# Make prediction with selected model
active_model = all_models[model_choice]
pred_value_100k = active_model.predict(scaled_input)[0]
pred_display_100k = max(0.0, pred_value_100k)
pred_usd = pred_display_100k * 100000.0

active_rmse = results[model_choice]["Test_RMSE"]
active_r2 = results[model_choice]["Test_R2"]

# Predict with all models for live comparison
all_predictions = {}
for name, m in all_models.items():
    val = max(0.0, m.predict(scaled_input)[0])
    all_predictions[name] = val

# Top KPI Metric Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"Predicted Value ({model_choice})",
        value=f"${pred_usd:,.0f}",
        delta=f"{pred_display_100k:.3f} ($100k)"
    )

with col2:
    st.metric(
        label="Test RMSE (Error Margin)",
        value=f"± ${active_rmse * 100000:,.0f}",
        help="Root Mean Squared Error on unseen test data"
    )

with col3:
    st.metric(
        label="Model Test R² Score",
        value=f"{active_r2*100:.2f}%",
        help="Proportion of target variance explained by this model"
    )

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geographic Location & Prediction",
    "⚖️ Live Multi-Model Comparison",
    "📊 Model Performance & Scorecards",
    "📈 Visual Diagnostics & Feature Importance"
])

with tab1:
    col_map, col_details = st.columns([1.5, 1])
    with col_map:
        st.subheader("Selected California Location")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=7)

    with col_details:
        st.subheader("Prediction Summary")
        st.info(f"""
        **Active Model:** `{model_choice}`  
        **Predicted Price:** **${pred_usd:,.2f}**  
        **Estimated Range (±1 RMSE):**  
        ${max(0, (pred_display_100k - active_rmse) * 100000):,.0f} — ${(pred_display_100k + active_rmse) * 100000:,.0f}
        
        **Coordinates:** Lat `{latitude:.2f}`, Lon `{longitude:.2f}`  
        **Household Income:** `${med_inc * 10000:,.0f}/year`  
        **Rooms / Bedrooms:** `{ave_rooms:.1f} rooms` / `{ave_bedrms:.1f} bedrms`  
        **Avg Occupancy:** `{ave_occup:.1f} persons/household`
        """)

        if latitude > 37.0 and longitude < -121.5:
            st.success("📍 Bay Area / Northern Coastal Zone (High valuation corridor)")
        elif latitude < 35.0 and longitude < -117.5:
            st.success("📍 Greater Los Angeles / Southern Coast (High valuation corridor)")
        else:
            st.warning("📍 Inland / Central Valley / Rural California Region")

with tab2:
    st.subheader("⚡ Real-Time Multi-Model Prediction Comparison")
    st.markdown("Comparing predictions across all trained algorithms for your currently selected parameters:")

    pred_comparison_df = pd.DataFrame({
        "Algorithm": list(all_predictions.keys()),
        "Predicted Price ($100k)": [round(v, 4) for v in all_predictions.values()],
        "Estimated Market Value ($USD)": [f"${v * 100000:,.2f}" for v in all_predictions.values()],
        "Model Test R²": [f"{results[m]['Test_R2']*100:.2f}%" for m in all_predictions.keys()],
        "Test RMSE": [f"${results[m]['Test_RMSE_USD']:,.0f}" for m in all_predictions.keys()]
    })
    st.dataframe(pred_comparison_df, use_container_width=True)

    # Comparison Bar Chart of Current Predictions
    st.bar_chart(pd.DataFrame({
        "Predicted Price ($100k)": all_predictions
    }))

with tab3:
    st.subheader("📊 Structured Model Performance Comparison (Task 2)")
    
    scorecard_df = pd.DataFrame(results).T
    display_scorecard = pd.DataFrame({
        "Algorithm": scorecard_df.index,
        "Test RMSE ($100k)": scorecard_df["Test_RMSE"].round(4),
        "Test USD Error": scorecard_df["Test_RMSE_USD"].apply(lambda v: f"${v:,.2f}"),
        "Test R² Score": scorecard_df["Test_R2"].apply(lambda v: f"{v*100:.2f}%"),
        "Test MAE ($USD)": scorecard_df["Test_MAE_USD"].apply(lambda v: f"${v:,.2f}"),
        "Train R² Score": scorecard_df["Train_R2"].apply(lambda v: f"{v*100:.2f}%"),
        "Overfitting R² Gap": scorecard_df["Overfitting_R2_Gap"].round(4)
    })
    st.dataframe(display_scorecard.set_index("Algorithm"), use_container_width=True)

    st.markdown("""
    #### 💡 Overfitting & Model Selection Insights:
    * **Decision Tree (depth=5)** outperforms Linear Regression by cutting RMSE to **$72,423** while maintaining an almost identical train-test gap ($\Delta R^2 = 0.0379$).
    * **Ridge Regression** provides L2 weight shrinkage to stabilize collinear features, but yields similar performance to Linear Regression due to dataset sample size ($N = 20,640$).
    * **Random Forest (Benchmark)** establishes the high-water mark at **$R^2 = 77.38\%$** ($\text{RMSE} = \$54,450$).
    """)

with tab4:
    st.subheader("📈 Diagnostic Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(assets_dir, "task2_model_comparison_bar.png"), caption="RMSE & R² Comparison across Models", use_container_width=True)
    with c2:
        st.image(os.path.join(assets_dir, "task2_actual_vs_predicted.png"), caption="Multi-Model Actual vs Predicted Scatter Plots", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.image(os.path.join(assets_dir, "task2_decision_tree_feature_importance.png"), caption="Decision Tree Feature Importance", use_container_width=True)
    with c4:
        st.image(os.path.join(assets_dir, "task2_residuals_comparison.png"), caption="Residual Error Distributions", use_container_width=True)

# Footer
st.markdown("---")
st.caption("Maincrafts Technology • AI & Machine Learning Internship — Task 1 & Task 2 Project Suite")
