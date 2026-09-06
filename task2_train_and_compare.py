"""
task2_train_and_compare.py
Task 2: Feature Engineering, Model Optimization & Performance Comparison
Maincrafts Technology - AI & Machine Learning Internship

This script implements:
1. Feature Scaling using StandardScaler
2. Training multiple regression algorithms:
   - Linear Regression (Baseline)
   - Ridge Regression (L2 Regularization)
   - Decision Tree Regressor (Non-linear)
   - Random Forest Regressor (Ensemble Benchmark)
3. Quantitative Comparison (RMSE, R2, MAE for Train and Test)
4. Overfitting / Generalization Assessment
5. Asset & Model Serialization
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def setup_dirs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for d in ['models', 'reports', 'assets']:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    return base_dir

def run_pipeline():
    base_dir = setup_dirs()
    assets_dir = os.path.join(base_dir, 'assets')
    models_dir = os.path.join(base_dir, 'models')

    print("=" * 65)
    print("  TASK 2: FEATURE ENGINEERING & MULTI-MODEL COMPARISON PIPELINE")
    print("=" * 65)

    # Step 1 & 2: Load Dataset
    print("\n[Step 1 & 2] Loading California Housing Dataset...")
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)
    feature_names = list(data.feature_names)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 3: Separate Features and Target
    print("\n[Step 3] Separating Features (X) and Target (y)...")
    X = df.drop("HousePrice", axis=1)
    y = df["HousePrice"]

    # Step 4: Feature Scaling (StandardScaler)
    print("\n[Step 4] Applying Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Step 5: Train-Test Split
    print("\n[Step 5] Splitting into Train (80%) and Test (20%) sets (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 6: Define Multiple Regression Models
    print("\n[Step 6] Initializing Regression Models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest (Benchmark)": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }

    # Step 7: Train, Evaluate, and Compare
    print("\n[Step 7] Training and Evaluating Models...")
    results = {}
    fitted_models = {}
    test_predictions = {}
    train_predictions = {}

    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Test set evaluation
        y_test_pred = model.predict(X_test)
        test_predictions[name] = y_test_pred
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Train set evaluation (for overfitting check)
        y_train_pred = model.predict(X_train)
        train_predictions[name] = y_train_pred
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        results[name] = {
            "Test_RMSE": float(test_rmse),
            "Test_RMSE_USD": float(test_rmse * 100000),
            "Test_R2": float(test_r2),
            "Test_MAE": float(test_mae),
            "Test_MAE_USD": float(test_mae * 100000),
            "Train_RMSE": float(train_rmse),
            "Train_R2": float(train_r2),
            "Overfitting_R2_Gap": float(train_r2 - test_r2)
        }

    # Format Results DataFrame
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 80)
    print("                    MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    summary_display = results_df[["Test_RMSE", "Test_R2", "Test_MAE", "Train_R2", "Overfitting_R2_Gap"]].copy()
    summary_display.columns = ["Test RMSE ($100k)", "Test R2", "Test MAE ($100k)", "Train R2", "R2 Gap (Train-Test)"]
    print(summary_display.to_string())
    print("=" * 80)

    # Determine best mandatory model (among Linear, Ridge, Decision Tree)
    mandatory_models = ["Linear Regression", "Ridge Regression", "Decision Tree"]
    best_mandatory_name = min(mandatory_models, key=lambda m: results[m]["Test_RMSE"])
    print(f"\n[BEST CORE MODEL] {best_mandatory_name}")
    print(f"   Test RMSE : {results[best_mandatory_name]['Test_RMSE']:.4f} (${results[best_mandatory_name]['Test_RMSE_USD']:,.2f})")
    print(f"   Test R2   : {results[best_mandatory_name]['Test_R2']:.4f} ({results[best_mandatory_name]['Test_R2']*100:.2f}%)")

    # Step 8: Visual Performance Validation
    print("\n[Step 8] Generating Visual Performance Validation Charts...")

    # 1. Model Comparison Bar Chart (RMSE & R2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    model_names_plot = list(results.keys())
    test_rmses = [results[m]["Test_RMSE"] for m in model_names_plot]
    test_r2s = [results[m]["Test_R2"] for m in model_names_plot]
    bar_colors = ['#4575b4', '#74add1', '#fdae61', '#f46d43']

    # RMSE Bar Chart
    bars1 = ax1.bar(model_names_plot, test_rmses, color=bar_colors, edgecolor='#333333', linewidth=0.8, width=0.55)
    ax1.set_title("Test RMSE Comparison (Lower is Better)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("RMSE ($100k)", fontsize=10)
    ax1.set_ylim(0, max(test_rmses) * 1.25)
    ax1.tick_params(axis='x', rotation=15)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}\n(${yval*100000:,.0f})",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # R2 Bar Chart
    bars2 = ax2.bar(model_names_plot, test_r2s, color=bar_colors, edgecolor='#333333', linewidth=0.8, width=0.55)
    ax2.set_title("Test R2 Score Comparison (Higher is Better)", fontweight='bold', fontsize=12)
    ax2.set_ylabel("R2 Score", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=15)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}\n({yval*100:.1f}%)",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.suptitle("Model Performance Comparison (Task 2)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    comparison_bar_path = os.path.join(assets_dir, "task2_model_comparison_bar.png")
    plt.savefig(comparison_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {comparison_bar_path}")

    # 2. Multi-Panel Actual vs Predicted Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, y_pred) in enumerate(test_predictions.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.3, color=bar_colors[i], s=18, edgecolors='none')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                color='#d73027', linestyle='--', linewidth=1.8, label='Ideal Line (y = x)')
        ax.set_title(f"{name}\n(R2 = {results[name]['Test_R2']:.3f}, RMSE = {results[name]['Test_RMSE']:.3f})",
                     fontweight='bold', fontsize=10)
        ax.set_xlabel("Actual House Price ($100k)", fontsize=9)
        ax.set_ylabel("Predicted House Price ($100k)", fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Actual vs. Predicted House Prices Across Models", fontsize=13, fontweight='bold')
    plt.tight_layout()
    act_vs_pred_path = os.path.join(assets_dir, "task2_actual_vs_predicted.png")
    plt.savefig(act_vs_pred_path, dpi=300)
    plt.close()
    print(f"      Saved: {act_vs_pred_path}")

    # 3. Residual Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, name in enumerate(mandatory_models):
        ax = axes[i]
        res = y_test - test_predictions[name]
        sns.histplot(res, kde=True, color=bar_colors[i], ax=ax, bins=40)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f"{name} Residuals\nMean: {np.mean(res):.3f}, Std: {np.std(res):.3f}", fontweight='bold', fontsize=10)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("Residual (Error) Distributions for Core Models", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    res_path = os.path.join(assets_dir, "task2_residuals_comparison.png")
    plt.savefig(res_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {res_path}")

    # 4. Feature Importance for Decision Tree
    dt_model = fitted_models["Decision Tree"]
    dt_importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values()

    plt.figure(figsize=(8, 4.8))
    bars = plt.barh(dt_importances.index, dt_importances.values, color='#4575b4', height=0.6, edgecolor='#333333')
    plt.title("Decision Tree Feature Importance (Variance Reduction)", fontweight='bold', fontsize=12)
    plt.xlabel("Relative Importance Weight", fontsize=10)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f} ({w*100:.1f}%)",
                 va='center', fontsize=8.5, fontweight='bold')
    plt.xlim(0, max(dt_importances.values) * 1.2)
    plt.tight_layout()
    dt_feat_path = os.path.join(assets_dir, "task2_decision_tree_feature_importance.png")
    plt.savefig(dt_feat_path, dpi=300)
    plt.close()
    print(f"      Saved: {dt_feat_path}")

    # Save Metrics JSON
    comparison_meta = {
        "models": results,
        "best_core_model": best_mandatory_name,
        "feature_names": feature_names,
        "feature_importances": {
            "Decision Tree": {k: float(v) for k, v in dt_importances.items()}
        }
    }
    metrics_json_path = os.path.join(models_dir, "task2_comparison_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(comparison_meta, f, indent=4)
    print(f"\nSaved comparison metrics to: {metrics_json_path}")

    # Save Best Model and Full Bundle
    best_model_bundle = {
        "model_name": best_mandatory_name,
        "model": fitted_models[best_mandatory_name],
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": results[best_mandatory_name]
    }
    best_model_path = os.path.join(models_dir, "task2_best_model.pkl")
    joblib.dump(best_model_bundle, best_model_path)
    print(f"Saved best model bundle to: {best_model_path}")

    all_models_bundle = {
        "models": fitted_models,
        "scaler": scaler,
        "feature_names": feature_names,
        "results": results
    }
    all_models_path = os.path.join(models_dir, "task2_all_models.pkl")
    joblib.dump(all_models_bundle, all_models_path)
    print(f"Saved all models bundle to: {all_models_path}")

    print("\nTask 2 Training & Asset Generation Pipeline Complete!")

if __name__ == '__main__':
    run_pipeline()
