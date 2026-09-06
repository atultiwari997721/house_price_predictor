# California Housing Price Predictor (Tasks 1 & 2)
### Artificial Intelligence & Machine Learning Internship — Maincrafts Technology

An industry-aligned Machine Learning repository covering the full ML lifecycle: data loading, exploratory data analysis (EDA), feature engineering, scaling, multi-algorithm training, structured performance comparison, residual diagnostics, professional PDF reports, and an interactive prediction web UI.

---

## 📌 Project Overview & Task Structure

This repository contains both **Task 1** and **Task 2** of the Maincrafts Technology AI/ML Internship:

### 🔹 Task 1: Baseline Linear Regression Model
* **Dataset**: California Housing Dataset (20,640 records, 8 features)
* **Goal**: Build and evaluate a baseline Multiple Linear Regression model
* **Deliverables**: [`task1_ml_linear_regression.ipynb`](./task1_ml_linear_regression.ipynb), [`reports/California_Housing_Linear_Regression_Report.pdf`](./reports/California_Housing_Linear_Regression_Report.pdf), [`models/linear_regression_model.pkl`](./models/linear_regression_model.pkl).

### 🔹 Task 2: Feature Engineering, Model Optimization & Performance Comparison
* **Goal**: Apply `StandardScaler` preprocessing, train multiple algorithms (`Linear Regression`, `Ridge Regression`, `Decision Tree Regressor`, and `Random Forest`), perform structured comparison on test/train data, analyze overfitting, and justify model selection.
* **Deliverables**: [`AI_ML_Task2_Model_Comparison.ipynb`](./AI_ML_Task2_Model_Comparison.ipynb), [`reports/Task2_Model_Optimization_Report.pdf`](./reports/Task2_Model_Optimization_Report.pdf), [`models/task2_best_model.pkl`](./models/task2_best_model.pkl), [`app.py`](./app.py).

---

## 📊 Task 2 Model Performance Scorecard

All models were evaluated on the standardized dataset using an 80/20 train/test split (`random_state=42`):

| Algorithm | Test RMSE ($100k) | Test Error (USD) | Test $R^2$ Score | Test MAE (USD) | Train $R^2$ | Overfitting Gap ($\Delta R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.7456 | $74,558 | 0.5758 (57.58%) | $53,320 | 0.6126 | +0.0368 |
| **Ridge Regression ($\alpha=1.0$)** | 0.7456 | $74,555 | 0.5758 (57.58%) | $53,319 | 0.6126 | +0.0367 |
| 🏆 **Decision Tree (`max_depth=5`)** | **0.7242** | **$72,423** | **0.5997 (59.97%)** | **$52,226** | **0.6377** | **+0.0379** |
| 🌲 **Random Forest (Benchmark)** | **0.5445** | **$54,450** | **0.7738 (77.38%)** | **$36,634** | **0.8719** | **+0.0982** |

---

## 🏆 Key Findings & Model Selection Justification

1. **Why Feature Scaling was Essential**:
   Features in the dataset have vastly different numerical ranges (e.g. `Population` $\approx 1,425$, `AveBedrms` $\approx 1.1$). Standardizing features via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) ensured uniform learning dynamics and fair penalty attribution in regularized models.

2. **Why Decision Tree Outperformed Linear & Ridge Models**:
   - Housing valuations exhibit strong **non-linear geospatial boundaries** (e.g. coastal Bay Area and Los Angeles premiums) that flat linear planes cannot separate.
   - The Decision Tree captures these step-wise thresholds, reducing test RMSE by **~$2,135** over the linear baseline.
   - Setting `max_depth=5` successfully controlled variance, keeping the train-test $R^2$ divergence minimal ($\Delta R^2 = 0.0379$).

3. **Ensemble Power (Random Forest)**:
   Averaging 100 decorrelated trees lifts $R^2$ to **77.38%** ($\text{RMSE} = \$54,450$), illustrating how bagging overcomes single-tree variance.

---

## 📁 Repository Directory Structure

```
house_price_predictor/
│
├── AI_ML_Task2_Model_Comparison.ipynb # [Task 2] Mandatory Jupyter Notebook (Executed)
├── task1_ml_linear_regression.ipynb   # [Task 1] Baseline Linear Regression Notebook
├── task2_train_and_compare.py         # [Task 2] Scaling, Multi-Model Training & Asset Generation
├── task2_generate_report.py           # [Task 2] ReportLab 2-Page PDF Report Generator
├── task2_build_notebook.py            # [Task 2] Script that compiles & executes Task 2 notebook
├── train_and_export.py                # [Task 1] Training & Asset pipeline
├── generate_report.py                 # [Task 1] PDF Report Generator
├── app.py                             # Interactive Streamlit Web UI (Multi-Model & Map)
├── requirements.txt                   # Project Dependencies
├── README.md                          # Comprehensive Documentation
│
├── models/
│   ├── task2_best_model.pkl           # Best-performing model bundle (Decision Tree)
│   ├── task2_all_models.pkl           # Bundle of all trained models for Streamlit UI
│   ├── task2_comparison_metrics.json  # Multi-model evaluation metrics JSON
│   ├── linear_regression_model.pkl    # Task 1 baseline model
│   └── metrics.json                   # Task 1 metrics JSON
│
├── reports/
│   ├── Task2_Model_Optimization_Report.pdf             # [Task 2] 2-Page Technical PDF Report
│   └── California_Housing_Linear_Regression_Report.pdf # [Task 1] 3-Page Technical PDF Report
│
└── assets/
    ├── task2_model_comparison_bar.png                 # RMSE & R2 Comparison Bar Chart
    ├── task2_actual_vs_predicted.png                  # Multi-panel Actual vs Predicted Plots
    ├── task2_residuals_comparison.png                 # Residual Error Distributions
    ├── task2_decision_tree_feature_importance.png     # Decision Tree Feature Importances
    ├── actual_vs_predicted.png                        # Task 1 Actual vs Predicted Plot
    ├── correlation_heatmap.png                        # Correlation Matrix Heatmap
    ├── distributions.png                              # Salient Feature Distributions
    ├── geo_distribution.png                           # California Geospatial Price Map
    └── feature_importance.png                         # Task 1 Standardized Coefficients
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Task 2 Training & Model Comparison Pipeline
```bash
python task2_train_and_compare.py
```
*Applies `StandardScaler`, fits Linear, Ridge, Decision Tree, and Random Forest models, prints the comparison table, and generates high-res assets in `assets/`.*

### 3. Generate Task 2 PDF Summary Report
```bash
python task2_generate_report.py
```
*Generates the 2-page publication-quality PDF report at [`reports/Task2_Model_Optimization_Report.pdf`](./reports/Task2_Model_Optimization_Report.pdf).*

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```
*Open `http://localhost:8501` to test real-time predictions, toggle between algorithms, view comparative prediction bars, and interact with the California map.*

### 5. Open Jupyter Notebooks
```bash
# Task 2 Model Comparison Notebook
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb

# Task 1 Baseline Notebook
jupyter notebook task1_ml_linear_regression.ipynb
```

---

## 👥 Organization & Internship Attribution
- **Organization**: Maincrafts Technology ([www.maincrafts.com](https://www.maincrafts.com))
- **Program**: Artificial Intelligence & Machine Learning Internship
- **Tasks**: Task 1 (House Price Predictor) & Task 2 (Model Optimization & Comparison)