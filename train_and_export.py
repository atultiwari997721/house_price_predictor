"""
train_and_export.py
California Housing Price Predictor - Training & Asset Generation Pipeline
Task 1: AI & Machine Learning (Maincrafts Technology)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def setup_directories():
    """Ensure output directories exist."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = ['models', 'reports', 'assets']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    return base_dir

def load_data():
    """Load California Housing dataset into a clean pandas DataFrame."""
    print("[1/5] Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = pd.concat([housing.data, housing.target.rename('MedHouseVal')], axis=1)
    print(f"      Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df, housing.feature_names

def perform_eda(df, assets_dir):
    """Generate and save EDA visualization assets."""
    print("[2/5] Performing Exploratory Data Analysis (EDA)...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                linewidths=0.5, square=True)
    plt.title("Correlation Matrix of California Housing Features", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(assets_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"      Saved: {heatmap_path}")

    # 2. Key Distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # MedHouseVal (Target)
    sns.histplot(df['MedHouseVal'], kde=True, color='#2b5c8f', ax=axes[0, 0], bins=40)
    axes[0, 0].set_title("Target: Median House Value ($100k)", fontweight='bold')
    axes[0, 0].set_xlabel("Value ($100,000s)")
    axes[0, 0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label=f'Mean: ${df["MedHouseVal"].mean():.2f}')
    axes[0, 0].legend()

    # MedInc
    sns.histplot(df['MedInc'], kde=True, color='#2e7d32', ax=axes[0, 1], bins=40)
    axes[0, 1].set_title("Feature: Median Income ($10k)", fontweight='bold')
    axes[0, 1].set_xlabel("Income ($10,000s)")

    # HouseAge
    sns.histplot(df['HouseAge'], kde=True, color='#e65100', ax=axes[1, 0], bins=35)
    axes[1, 0].set_title("Feature: Median House Age (Years)", fontweight='bold')
    axes[1, 0].set_xlabel("Age (Years)")

    # AveRooms (clipped to 10 for clean visualization)
    sns.histplot(df[df['AveRooms'] <= 10]['AveRooms'], kde=True, color='#6a1b9a', ax=axes[1, 1], bins=30)
    axes[1, 1].set_title("Feature: Average Rooms (<= 10)", fontweight='bold')
    axes[1, 1].set_xlabel("Rooms per Household")

    plt.suptitle("Distribution of Target and Salient Features", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    dist_path = os.path.join(assets_dir, "distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"      Saved: {dist_path}")

    # 3. Geographic Distribution (Latitude vs Longitude vs Price)
    plt.figure(figsize=(9, 6.5))
    scatter = plt.scatter(
        df['Longitude'], df['Latitude'],
        c=df['MedHouseVal'], cmap='plasma', alpha=0.4,
        s=df['Population'] / 100, label='Population / 100'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Median House Value ($100k)", fontsize=10)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.title("California Geographic Price Distribution\n(High prices concentrated in coastal Bay Area & LA)",
              fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    geo_path = os.path.join(assets_dir, "geo_distribution.png")
    plt.savefig(geo_path, dpi=300)
    plt.close()
    print(f"      Saved: {geo_path}")

def train_and_evaluate(df, feature_names, models_dir, assets_dir):
    """Split data, train baseline Linear Regression, evaluate, and save artifacts."""
    print("[3/5] Splitting data and training Linear Regression model...")
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Unscaled baseline (Starter code version)
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    # 2. Standardized Pipeline (for robust coefficient comparisons)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred_pipeline = pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_baseline)
    mse = mean_squared_error(y_test, y_pred_baseline)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_baseline)

    train_r2 = r2_score(y_train, baseline_model.predict(X_train))

    print(f"\n--- Model Evaluation Results ---")
    print(f"MAE  : {mae:.4f} ($100k) -> ${mae * 100000:,.2f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f} ($100k) -> ${rmse * 100000:,.2f}")
    print(f"R² (Test) : {r2:.4f}")
    print(f"R² (Train): {train_r2:.4f}\n")

    # Save metrics and summary
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Test": float(r2),
        "R2_Train": float(train_r2),
        "Sample_Size_Train": int(len(X_train)),
        "Sample_Size_Test": int(len(X_test)),
        "Features": list(feature_names),
        "Raw_Coefficients": {col: float(coef) for col, coef in zip(feature_names, baseline_model.coef_)},
        "Raw_Intercept": float(baseline_model.intercept_),
        "Std_Coefficients": {col: float(coef) for col, coef in zip(feature_names, pipeline.named_steps['regressor'].coef_)}
    }
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[4/5] Saved metrics to: {metrics_path}")

    # Generate Evaluation plots
    print("      Generating evaluation plots...")
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(7.5, 6))
    plt.scatter(y_test, y_pred_baseline, alpha=0.3, color='#1f77b4', edgecolors='none', s=25)
    min_val = min(min(y_test), min(y_pred_baseline))
    max_val = max(max(y_test), max(y_pred_baseline))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#d62728', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    plt.xlabel("Actual Median House Value ($100k)", fontsize=11)
    plt.ylabel("Predicted Median House Value ($100k)", fontsize=11)
    plt.title("Actual vs. Predicted Values (Linear Regression)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    act_vs_pred_path = os.path.join(assets_dir, "actual_vs_predicted.png")
    plt.savefig(act_vs_pred_path, dpi=300)
    plt.close()

    # Plot 2: Residuals Distribution & Residuals vs Predicted
    residuals = y_test - y_pred_baseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residual Histogram
    sns.histplot(residuals, kde=True, color='#4575b4', ax=ax1, bins=45)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title("Residual Distribution (Errors)", fontweight='bold')
    ax1.set_xlabel("Residual (Actual - Predicted)")
    ax1.set_ylabel("Count")

    # Residuals vs Predicted
    ax2.scatter(y_pred_baseline, residuals, alpha=0.3, color='#313695', s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title("Residuals vs. Predicted Values", fontweight='bold')
    ax2.set_xlabel("Predicted Value ($100k)")
    ax2.set_ylabel("Residual ($100k)")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    res_path = os.path.join(assets_dir, "residuals_distribution.png")
    plt.savefig(res_path, dpi=300)
    plt.close()

    # Plot 3: Standardized Feature Importance / Coefficients
    std_coefs = pd.Series(pipeline.named_steps['regressor'].coef_, index=feature_names).sort_values()
    plt.figure(figsize=(8, 5))
    colors = ['#d73027' if c < 0 else '#1a9850' for c in std_coefs.values]
    bars = plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.6)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("Standardized Coefficient Impact (Std Deviations)", fontsize=10)
    plt.title("Feature Impact on House Value (Standardized Linear Regression)", fontsize=12, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        offset = 0.02 if width >= 0 else -0.02
        plt.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                 va='center', ha=ha, fontsize=9, fontweight='bold')
    plt.tight_layout()
    feat_path = os.path.join(assets_dir, "feature_importance.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()

    # Save trained model and pipeline bundle
    model_bundle = {
        'model': baseline_model,
        'pipeline': pipeline,
        'feature_names': list(feature_names),
        'metrics': metrics,
        'target_name': 'MedHouseVal'
    }
    model_path = os.path.join(models_dir, "linear_regression_model.pkl")
    joblib.dump(model_bundle, model_path)
    print(f"[5/5] Saved model bundle to: {model_path}")
    print("\nTraining and asset generation complete!")

    return metrics

if __name__ == '__main__':
    base_dir = setup_directories()
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    assets_dir = os.path.join(base_dir, 'assets')
    
    df, feature_names = load_data()
    perform_eda(df, assets_dir)
    train_and_evaluate(df, feature_names, models_dir, assets_dir)
