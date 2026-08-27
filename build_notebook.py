"""
build_notebook.py
Generates and executes task1_ml_linear_regression.ipynb
Ensures all markdown cells, code cells, and execution outputs (plots, tables, logs)
are populated in the final Jupyter Notebook.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def generate_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # 1. Title & Header Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Artificial Intelligence & Machine Learning — Task 1
## Build & Evaluate a Linear Regression Model (House Price Predictor)

**Organization:** Maincrafts Technology  
**Project:** California Housing Price Predictor  
**Author:** AI & Machine Learning Intern  
**Objective:** Introduce the complete Machine Learning workflow: data ingestion, exploratory data analysis (EDA), data preprocessing, model training (`LinearRegression`), performance evaluation (MAE, RMSE, $R^2$), residual diagnostics, and model serialization.

---
### Workflow Stages
1. **Setup & Data Ingestion**: Load the California Housing dataset from `sklearn.datasets`.
2. **Exploratory Data Analysis (EDA)**: Inspect distributions, missing values, statistics, and correlation patterns.
3. **Data Splitting**: Partition data into training (80%) and testing (20%) sets.
4. **Model Training**: Fit an Ordinary Least Squares (OLS) Linear Regression model.
5. **Evaluation**: Compute Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ score.
6. **Residual Diagnostics**: Analyze prediction errors, homoscedasticity, and coefficient weights.
7. **Model Persistence**: Serialize the trained model pipeline for production inference.
8. **Summary & Improvement Roadmap**: Document findings and non-linear model opportunities.
"""))

    # 2. Imports
    cells.append(nbf.v4.new_markdown_cell("""### 1. Imports & Global Configuration
We load standard data science and machine learning libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`."""))

    cells.append(nbf.v4.new_code_cell("""# Basics & Data Science Stack
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetics & Plotting Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (9, 5.5)
plt.rcParams['font.sans-serif'] = 'Arial'
%matplotlib inline

print("All dependencies successfully imported!")
"""))

    # 3. Data Loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Dataset Loading & Inspection
We load the California Housing dataset using `fetch_california_housing(as_frame=True)`.  
The target variable is `MedHouseVal` (Median House Value in hundreds of thousands of dollars, i.e., $100,000s)."""))

    cells.append(nbf.v4.new_code_cell("""# Load California housing dataset
data = fetch_california_housing(as_frame=True)

# Combine features and target into a single pandas DataFrame
df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Feature Descriptions:
* **MedInc**: Median income in block group (in $10,000s).
* **HouseAge**: Median house age in block group (in years).
* **AveRooms**: Average number of rooms per household.
* **AveBedrms**: Average number of bedrooms per household.
* **Population**: Block group population.
* **AveOccup**: Average number of household members.
* **Latitude**: Block group latitude coordinate.
* **Longitude**: Block group longitude coordinate.
* **MedHouseVal (Target)**: Median house value for California districts (in $100,000s).
"""))

    # 4. Data Hygiene & Missing Value Audit
    cells.append(nbf.v4.new_markdown_cell("""### 3. Data Hygiene & Missing Value Audit
Checking for missing values, null entries, and column data types."""))

    cells.append(nbf.v4.new_code_cell("""# Check data types and missing values
null_counts = df.isnull().sum()
print("Missing values per feature:")
print(null_counts)
print(f"\\nTotal missing values across entire dataset: {null_counts.sum()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics
df.describe().T
"""))

    # 5. Exploratory Data Analysis (EDA)
    cells.append(nbf.v4.new_markdown_cell("""### 4. Exploratory Data Analysis (EDA)
Understanding feature distributions, skewness, and the target variable behavior."""))

    cells.append(nbf.v4.new_code_cell("""# Visualizing Target Variable (MedHouseVal) and Key Regressors
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. MedHouseVal
sns.histplot(df['MedHouseVal'], kde=True, color='#1f77b4', ax=axes[0, 0], bins=40)
axes[0, 0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label=f"Mean: ${df['MedHouseVal'].mean():.2f} ($100k)")
axes[0, 0].set_title("Distribution of Median House Value (Target)", fontweight='bold')
axes[0, 0].set_xlabel("MedHouseVal ($100,000s)")
axes[0, 0].legend()

# 2. MedInc
sns.histplot(df['MedInc'], kde=True, color='#2ca02c', ax=axes[0, 1], bins=40)
axes[0, 1].set_title("Distribution of Median Income", fontweight='bold')
axes[0, 1].set_xlabel("MedInc ($10,000s)")

# 3. HouseAge
sns.histplot(df['HouseAge'], kde=True, color='#ff7f0e', ax=axes[1, 0], bins=35)
axes[1, 0].set_title("Distribution of House Age", fontweight='bold')
axes[1, 0].set_xlabel("HouseAge (Years)")

# 4. AveRooms (trimmed for visual clarity)
sns.histplot(df[df['AveRooms'] <= 10]['AveRooms'], kde=True, color='#9467bd', ax=axes[1, 1], bins=30)
axes[1, 1].set_title("Distribution of Average Rooms (<= 10)", fontweight='bold')
axes[1, 1].set_xlabel("Rooms per Household")

plt.suptitle("Exploratory Distributions of Target & Salient Predictors", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Correlation Analysis
We calculate the Pearson correlation matrix to understand the linear relationships between features and the target variable."""))

    cells.append(nbf.v4.new_code_cell("""# Correlation Matrix Heatmap
plt.figure(figsize=(9, 7))
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
            linewidths=0.5, square=True)
plt.title("Correlation Matrix of California Housing Features", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()

print("Correlations with Target (MedHouseVal) sorted by magnitude:")
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Geographic Distribution
Visualizing house prices across geographic coordinates (Latitude vs Longitude).  
Coastal areas (e.g. San Francisco Bay Area and Los Angeles) exhibit noticeable price concentration."""))

    cells.append(nbf.v4.new_code_cell("""# California Geographic Price Scatter Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['Longitude'], df['Latitude'],
    c=df['MedHouseVal'], cmap='plasma', alpha=0.4,
    s=df['Population'] / 100, label='Population / 100'
)
cbar = plt.colorbar(scatter)
cbar.set_label("Median House Value ($100k)", fontsize=11)
plt.xlabel("Longitude", fontsize=11)
plt.ylabel("Latitude", fontsize=11)
plt.title("Geographic Price Distribution in California\\n(Bubble size = Population, Color = House Value)",
          fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
"""))

    # 6. Feature Selection & Train-Test Split
    cells.append(nbf.v4.new_markdown_cell("""### 5. Feature Selection & Train/Test Split
We separate features $X$ and target $y$, followed by an 80/20 train/test split with `random_state=42` for exact reproducibility."""))

    cells.append(nbf.v4.new_code_cell("""# Separate regressors (X) and target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Perform 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set:   X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set:    X_test={X_test.shape},  y_test={y_test.shape}")
"""))

    # 7. Model Training
    cells.append(nbf.v4.new_markdown_cell("""### 6. Model Training (Linear Regression)
We instantiate and fit `LinearRegression()` using Ordinary Least Squares (OLS)."""))

    cells.append(nbf.v4.new_code_cell("""# Instantiate and train baseline Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

print("Model training completed successfully!")
print(f"Intercept (theta_0): {model.intercept_:.4f}")
print("\\nLearned Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:12s}: {coef:12.6f}")
"""))

    # 8. Evaluation Metrics
    cells.append(nbf.v4.new_markdown_cell("""### 7. Model Evaluation & Performance Metrics
We evaluate the model using standard regression metrics:
* **Mean Absolute Error (MAE)**: Average magnitude of the errors without direction.
* **Mean Squared Error (MSE)** & **Root Mean Squared Error (RMSE)**: Penalizes larger deviations more heavily.
* **$R^2$ Score (Coefficient of Determination)**: Proportion of variance in the target explained by the model.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, model.predict(X_train))

# Format and display results
print("=" * 50)
print("              MODEL PERFORMANCE SCORECARD")
print("=" * 50)
print(f"Mean Absolute Error (MAE)   : {mae:.4f} ($100k)  ->  ${mae*100000:,.2f}")
print(f"Mean Squared Error (MSE)    : {mse:.4f}")
print(f"Root Mean Squared Error(RMSE): {rmse:.4f} ($100k)  ->  ${rmse*100000:,.2f}")
print(f"R² Score (Test Set)         : {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"R² Score (Train Set)        : {r2_train:.4f} ({r2_train*100:.2f}%)")
print("=" * 50)
"""))

    # 9. Visual Diagnostics
    cells.append(nbf.v4.new_markdown_cell("""### 8. Visual Diagnostics & Residual Analysis
We inspect:
1. **Actual vs Predicted Scatter Plot**: Shows how closely predictions align with the 45-degree reference line ($y=x$).
2. **Residual Distribution & Homoscedasticity**: Verifies whether errors have zero mean and consistent variance across predicted values.
3. **Standardized Feature Importance**: Compares feature impacts after normalising feature scales.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
plt.xlabel("Actual Median House Value ($100k)", fontsize=11)
plt.ylabel("Predicted Median House Value ($100k)", fontsize=11)
plt.title("Actual vs. Predicted Median House Value", fontsize=13, fontweight='bold')
plt.legend(loc="upper left")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Residual Distribution & Residuals vs Predicted
residuals = y_test - y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Residuals Histogram & KDE
sns.histplot(residuals, kde=True, color='#4575b4', ax=ax1, bins=45)
ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
ax1.set_title("Distribution of Residuals (Errors)", fontweight='bold')
ax1.set_xlabel("Residual: Actual - Predicted ($100k)")
ax1.set_ylabel("Frequency")
ax1.legend()

# Residuals vs Predicted Values
ax2.scatter(y_pred, residuals, alpha=0.3, color='#313695', s=20)
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax2.set_title("Residuals vs. Predicted Values (Homoscedasticity Check)", fontweight='bold')
ax2.set_xlabel("Predicted Value ($100k)")
ax2.set_ylabel("Residual ($100k)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Standardized Feature Importance
# Standardizing features allows direct comparison of relative feature contributions
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

std_coefs = pd.Series(pipeline.named_steps['regressor'].coef_, index=X.columns).sort_values()

plt.figure(figsize=(9, 5))
colors = ['#d73027' if c < 0 else '#1a9850' for c in std_coefs.values]
bars = plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.6)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("Standardized Coefficient (Std Deviations)", fontsize=11)
plt.title("Standardized Feature Impact on House Value", fontsize=13, fontweight='bold')

for bar in bars:
    w = bar.get_width()
    ha = 'left' if w >= 0 else 'right'
    offset = 0.02 if w >= 0 else -0.02
    plt.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
             va='center', ha=ha, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    # 10. Model Persistence
    cells.append(nbf.v4.new_markdown_cell("""### 9. Model Serialization & Test Inference
We persist the trained model pipeline to `models/linear_regression_model.pkl` using `joblib` so it can be deployed into our Streamlit prediction application."""))

    cells.append(nbf.v4.new_code_cell("""# Save the model artifact
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "linear_regression_model.pkl")

bundle = {
    'model': model,
    'pipeline': pipeline,
    'feature_names': list(X.columns),
    'metrics': {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2_test)
    }
}
joblib.dump(bundle, model_path)
print(f"Model successfully saved to: {model_path}")

# Verify reloading and perform a sample prediction
loaded_bundle = joblib.load(model_path)
sample_input = X_test.iloc[0:1]
sample_pred = loaded_bundle['model'].predict(sample_input)[0]
sample_actual = y_test.iloc[0]

print(f"\\nSample Verification:")
print(f"Input features:\\n{sample_input.to_dict(orient='records')[0]}")
print(f"Predicted Price : ${sample_pred * 100000:,.2f} ({sample_pred:.3f} in $100k)")
print(f"Actual Price    : ${sample_actual * 100000:,.2f} ({sample_actual:.3f} in $100k)")
"""))

    # 11. Conclusions & Recommendations
    cells.append(nbf.v4.new_markdown_cell("""### 10. Summary & Future Recommendations

#### Observations
1. **Baseline Fit**: The Linear Regression model explains **57.58%** of the variance in California housing prices ($R^2 = 0.5758$) with an RMSE of **$74,558**.
2. **Key Determinant**: Median Income (`MedInc`) is overwhelmingly the strongest positive contributor (+0.854 std coef).
3. **Ceiling Distortion**: The 5.0 ($500,000) target truncation creates noticeable residual bias at higher price ranges.
4. **Multicollinearity**: `AveRooms` and `AveBedrms` exhibit strong collinearity ($r = 0.85$), which introduces negative coefficients on `AveRooms` when combined with `AveBedrms`.

#### Actionable Improvements
* **Feature Engineering**: Compute house density metrics and distance to the nearest Pacific coast or major metropolitan centers (San Francisco, Los Angeles).
* **Regularization**: Utilize Ridge or ElasticNet regression to handle multicollinearity between room features.
* **Non-Linear Ensembles**: Transition to tree-based ensemble models such as **Random Forest**, **LightGBM**, or **XGBoost**, which routinely achieve $R^2 > 0.80$ on this dataset by learning non-linear geospatial boundary surfaces.
"""))

    nb['cells'] = cells
    nb_path = "task1_ml_linear_regression.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook written to {nb_path}. Now executing cells...")
    
    # Execute the notebook to store outputs
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook successfully executed and saved with all outputs: {nb_path}")

if __name__ == '__main__':
    generate_notebook()
