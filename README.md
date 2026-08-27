# California Housing Price Predictor (Linear Regression)
### Artificial Intelligence & Machine Learning — Task 1 (Maincrafts Technology)

An end-to-end Machine Learning project demonstrating data loading, exploratory data analysis (EDA), preprocessing, model training, metric evaluation, residual diagnostics, report compilation, and an interactive prediction web UI.

---

## 📌 Project Overview
The objective is to train and evaluate a **Multiple Linear Regression model** on the California Housing dataset to predict median district housing values (`MedHouseVal`). 

- **Target Variable**: `MedHouseVal` (Median house value in $100,000s; capped at 5.0 = $500k)
- **Features**: 8 continuous demographic and geospatial attributes (20,640 census block records)
- **Train/Test Split**: 80% train (16,512 rows), 20% test (4,128 rows), `random_state=42`

---

## 📊 Model Performance Scorecard

| Metric | Test Set Value | USD Equivalent | Meaning |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **0.5332** | **$53,320.01** | Average absolute prediction error |
| **Mean Squared Error (MSE)** | **0.5559** | — | Residual variance |
| **Root Mean Squared Error (RMSE)**| **0.7456** | **$74,558.14** | Standard error penalizing large deviations |
| **$R^2$ Score (Test Set)** | **0.5758** | **57.58%** | Proportion of variance explained by the model |
| **$R^2$ Score (Train Set)** | **0.6126** | **61.26%** | Consistent with test set (low overfitting) |

---

## 📁 Repository Structure

```
house_price_predictor/
│
├── task1_ml_linear_regression.ipynb   # Fully executed Jupyter Notebook with markdown, code & charts
├── train_and_export.py                # Standalone training script, metric evaluator & asset generator
├── generate_report.py                 # ReportLab script producing multi-page PDF summary report
├── app.py                             # Interactive Streamlit web application with California map
├── build_notebook.py                  # Programmatic generator & executor for the notebook
├── requirements.txt                   # Environment dependencies
├── README.md                          # Project documentation
│
├── models/
│   ├── linear_regression_model.pkl    # Serialized model & pipeline bundle (joblib)
│   └── metrics.json                   # Exported evaluation metrics & coefficients
│
├── reports/
│   └── California_Housing_Linear_Regression_Report.pdf # 3-page technical and executive PDF report
│
└── assets/
    ├── correlation_heatmap.png        # Correlation matrix visualization
    ├── distributions.png              # Target & feature distribution plots
    ├── geo_distribution.png           # California geospatial price cluster map
    ├── actual_vs_predicted.png        # Actual vs Predicted scatter plot with identity line
    ├── residuals_distribution.png     # Residual error distribution & homoscedasticity check
    └── feature_importance.png         # Standardized coefficient impact bar chart
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Model and Generate Assets
Run the end-to-end training pipeline:
```bash
python train_and_export.py
```
This will:
- Ingest California Housing dataset
- Generate all high-resolution figures in `assets/`
- Train baseline Linear Regression and standardized pipeline
- Evaluate MAE, RMSE, and $R^2$
- Serialize artifacts to `models/linear_regression_model.pkl` and `models/metrics.json`

### 3. Generate the PDF Report
To compile the 3-page professional PDF summary report:
```bash
python generate_report.py
```
The resulting PDF is located at:  
`reports/California_Housing_Linear_Regression_Report.pdf`

### 4. Launch the Interactive Web Application
Start the Streamlit application for interactive predictions:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### 5. View / Run the Jupyter Notebook
Open `task1_ml_linear_regression.ipynb` in Jupyter Notebook or VS Code / IDE:
```bash
jupyter notebook task1_ml_linear_regression.ipynb
```
*(All cells are pre-executed and contain all rendered tables and charts.)*

---

## 🔍 Key Findings & Diagnostic Insights

1. **Dominant Regressor (`MedInc`)**: Median income has a standardized coefficient of **+0.854** and a Pearson correlation of **+0.69**, making it the single most influential predictor of house prices.
2. **Geospatial Concentration**: Coastal block groups (San Francisco Bay Area and Greater Los Angeles) command higher prices than inland districts.
3. **Multicollinearity (`AveRooms` vs. `AveBedrms`)**: Strong collinearity ($r = 0.85$) produces negative coefficient sign on `AveRooms` when combined with `AveBedrms` in standard unregularized OLS.
4. **Target Truncation ($500k Cap)**: Approximately 4.67% of block groups are capped at 5.0 ($500,000), causing the linear model to underpredict top-tier luxury properties.

---

## 🔮 Recommended Next Steps
- **Geospatial Feature Engineering**: Add distance metrics to the Pacific coastline and major urban tech hubs (San Francisco, Silicon Valley, Los Angeles).
- **Regularization**: Implement **Ridge** or **ElasticNet** regression to penalize collinear room counts and stabilize coefficients.
- **Non-Linear Ensembles**: Transition to tree-based models like **Random Forest**, **LightGBM**, or **XGBoost**, which typically exceed $R^2 > 0.82$ by capturing localized spatial thresholds.