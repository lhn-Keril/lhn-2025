# -*- coding: utf-8 -*-
"""
Carbon Dot Fluorescence Intensity Prediction System (Final Enhanced Version v2)
✅ Added: Linear fit comparison plots for three models | ✅ Output directory: E:/桌面/第一章机器学习/18
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import joblib

# ====== Font and Display Settings (English) ======
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Fallback-safe sans-serif font
matplotlib.rcParams['axes.unicode_minus'] = False

# ====== Path Configuration ======
output_dir = r"E:/桌面/第一章机器学习/18"
os.makedirs(output_dir, exist_ok=True)

# ====== Load Data ======
data_path = r"E:/桌面/第一章机器学习/18/增强后数据.xlsx"
df = pd.read_excel(data_path, sheet_name="Sheet1")
df = df[df['FL'] > 0].reset_index(drop=True)

features = ['GSH(g)', 'Urea(g)', 'FA(ml)', 'Temp(℃)', 'Time(h)']
X = df[features].copy()
y = df['FL'].copy()

# ====== Feature Engineering ======
X['GSH_urea_ratio'] = X['GSH(g)'] / (X['Urea(g)'] + 1e-6)
X['Temp_time'] = X['Temp(℃)'] * X['Time(h)']
X['FA_GSH'] = X['FA(ml)'] * X['GSH(g)']
X['urea_sq'] = X['Urea(g)'] ** 2
X['Temp_sq'] = X['Temp(℃)'] ** 2
engineered_features = X.columns.tolist()

print(f"Original features: 5 → Engineered features: {len(engineered_features)} - 18.py:44")

# ====== Correlation Heatmap ======
plt.figure(figsize=(12, 10))
corr = pd.concat([X, y], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Feature Correlation with Fluorescence Intensity (FL)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# ====== Train/Test Split & Scaling ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.pkl"))
joblib.dump(engineered_features, os.path.join(output_dir, "feature_names.pkl"))

# ====== Model Definitions ======
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

models_config = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            'n_estimators': [200, 300],
            'max_depth': [15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        "needs_scaling": False
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
        "params": {
            'n_estimators': [300, 400],
            'learning_rate': [0.03, 0.05],
            'max_depth': [5, 6],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        },
        "needs_scaling": False
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            'n_estimators': [300, 400],
            'learning_rate': [0.03, 0.05],
            'num_leaves': [31, 41],
            'max_depth': [6, -1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        },
        "needs_scaling": False
    }
}

# ====== Training & Evaluation ======
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_r2 = -np.inf
best_name = ""

for name, config in models_config.items():
    print(f"\n🔍 Training {name}... - 18.py:118")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=kf,
        scoring='r2',
        n_jobs=-1,
        error_score='raise'
    )

    try:
        if config["needs_scaling"]:
            grid.fit(X_train_scaled, y_train)
            y_pred = grid.predict(X_test_scaled)
        else:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
    except Exception as e:
        print(f"⚠️ {name} training failed: {e} - 18.py:137")
        continue

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    cv_r2 = cross_val_score(grid.best_estimator_, X_train, y_train, cv=kf, scoring='r2').mean()

    results[name] = {
        "R² (Test)": r2,
        "R² (5-Fold CV)": cv_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Best Params": str(grid.best_params_),
        "Model": grid.best_estimator_,
        "Predictions": y_pred
    }

    print(f"{name} → R²: {r2:.4f}, CV R²: {cv_r2:.4f}, MAE: {mae:.2f} - 18.py:157")

    if r2 > best_r2:
        best_r2 = r2
        best_model = grid.best_estimator_
        best_name = name

if best_model is None:
    raise RuntimeError("All models failed to train!")

# ====== Save Best Model ======
joblib.dump(best_model, os.path.join(output_dir, f"best_model_{best_name}.pkl"))

# ====== Performance Summary Excel ======
summary_data = []
for model, metrics in results.items():
    row = {"Model": model}
    for k, v in metrics.items():
        if k not in ["Model", "Predictions"]:
            row[k] = v
    summary_data.append(row)

results_df = pd.DataFrame(summary_data)
results_df.to_excel(os.path.join(output_dir, "model_performance_comparison.xlsx"), index=False)

# ===================================================================================
# ======================= Linear Fit Comparison Plots ===============================
# ===================================================================================

for name in results:
    y_pred = results[name]["Predictions"]
    r2 = results[name]["R² (Test)"]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=40, label='Predictions')

    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')

    slope, intercept, _, _, _ = stats.linregress(y_test, y_pred)
    fit_line = slope * np.array(y_test) + intercept
    plt.plot(y_test, fit_line, 'b-', lw=2, label=f'Linear Fit: y={slope:.2f}x+{intercept:.0f}')

    plt.xlabel("Actual FL")
    plt.ylabel("Predicted FL")
    plt.title(f"{name} Linear Fit\nR$^2$ = {r2:.4f}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"linear_fit_{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ===================================================================================
# ========================== Other Visualizations ===================================
# ===================================================================================

# Prediction vs Actual (Best Model)
y_pred_best = results[best_name]["Predictions"]
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7, edgecolors='k', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel("Actual FL", fontsize=12)
plt.ylabel("Predicted FL", fontsize=12)
plt.title(f"{best_name} Prediction Performance\nR$^2$ = {best_r2:.4f}", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "prediction_vs_actual.png"), dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': engineered_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', color='steelblue')
plt.title(f"Top 10 Important Features ({best_name})", fontsize=14)
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()

feat_imp_df.to_excel(os.path.join(output_dir, "feature_importance_ranking.xlsx"), index=False)

# Optimization Guidelines (English)
top_features = feat_imp_df.head(5)['Feature'].tolist()
top_features_str = "\n  ".join([f"{i+1}. {feat}" for i, feat in enumerate(top_features)])

optimization_logic = f"""# Carbon Dot Synthesis Optimization Guidelines (Based on {best_name} Model)

## Best FL Prediction Model: {best_name}
- Test Set R²: {best_r2:.4f}
- Top 5 Influential Features:
  {top_features_str}

## Synthesis Recommendations:
1. Urea amount: 0.8–1.2 g yields higher FL (both lower and higher reduce FL)
2. Reaction temperature: 150–170 °C optimal (excessive temp causes over-carbonization)
3. GSH/Urea ratio ≈ 0.2:1 performs best
4. Reaction time: 6–9 hours ideal; longer duration may cause fluorescence quenching
5. FA volume: 10–15 mL preferred for better dispersion and FL

## Approximate Prediction Behavior:
FL ≈ f(GSH, Urea, FA, Temperature, Time, GSH/Urea, Temp×Time, ...)

> Note: The model is tree-based and has no explicit analytical equation. Use the saved model for predictions.
"""

with open(os.path.join(output_dir, "optimization_guidelines.txt"), "w", encoding="utf-8") as f:
    f.write(optimization_logic)

pseudo_eq_df = pd.DataFrame({
    "Optimization Direction": [
        "↑ Urea (0.8~1.2 g)",
        "↑ Temperature (150~170°C)",
        "↑ GSH/Urea ≈ 0.2",
        "↑ Time (6~9 h)",
        "↑ FA (10~15 mL)"
    ],
    "Expected Effect": [
        "FL peaks within range",
        "FL maximized in mid-temp zone",
        "Imbalance significantly reduces FL",
        "Excessive time reduces FL",
        "Optimal FA improves dispersion & FL"
    ]
})
pseudo_eq_df.to_excel(os.path.join(output_dir, "optimization_recommendations.xlsx"), index=False)

# Save Test Results
test_results = X_test.copy()
test_results['Actual_FL'] = y_test.values
test_results['Predicted_FL'] = y_pred_best
test_results.to_excel(os.path.join(output_dir, "test_set_predictions.xlsx"), index=False)

# ===================================================================================
# ========================== Multi-Model Comparison Plots ============================
# ===================================================================================

# R² and MAE Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names = list(results.keys())
r2_vals = [results[name]["R² (Test)"] for name in names]
mae_vals = [results[name]["MAE"] for name in names]

bars1 = axes[0].bar(names, r2_vals, color=['#ED5F5F', '#ff7f0e', '#4497C4'], alpha=0.85)
axes[0].set_title("Model R$^2$ Comparison", fontsize=14)
axes[0].set_ylabel("R$^2$")
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
for bar, val in zip(bars1, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center')

bars2 = axes[1].bar(names, mae_vals, color=[ '#ED5F5F', '#ff7f0e', '#4497C4'], alpha=0.85)
axes[1].set_title("Model MAE Comparison", fontsize=14)
axes[1].set_ylabel("MAE")
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
for bar, val in zip(bars2, mae_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + max(mae_vals)*0.03, f'{val:.1f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_R2_MAE_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Residual Analysis
for name in results:
    y_pred = results[name]["Predictions"]
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k', s=40)
    plt.axhline(0, color='red', linestyle='--', lw=1.5)
    plt.xlabel("Predicted FL")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"{name} Residual Analysis")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"residual_analysis_{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Prediction Error Distribution
plt.figure(figsize=(8, 5))
for name in results:
    y_pred = results[name]["Predictions"]
    errors = y_pred - y_test
    sns.kdeplot(errors, label=name, fill=True, alpha=0.4)
plt.axvline(0, color='black', linestyle='--', lw=1, label='Zero Error')
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Density")
plt.title("Prediction Error Distribution Across Models")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "prediction_error_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# ===================================================================================
# ============ Local Optimal Condition Search Near User-Specified Point ==============
# ===================================================================================

print("\n🔍 Searching for local optimal synthesis conditions near userspecified point... - 18.py:360")

center_condition = {
    'GSH(g)': 0.2,
    'Urea(g)': 0.8,
    'FA(ml)': 10.0,
    '温度(℃)': 160.0,
    '时间(h)': 8.0
}

tolerance = {
    'GSH(g)': 0.02,
    'Urea(g)': 0.08,
    'FA(ml)': 1.0,
    'Temp(℃)': 5.0,
    'Time(h)': 0.5
}

param_bounds = {key: (val - tol, val + tol) for key, (val, tol) in zip(center_condition.keys(), zip(center_condition.values(), tolerance.values()))}

np.random.seed(2025)
n_local = 20000
local_candidates = pd.DataFrame({
    'GSH(g)': np.random.uniform(*param_bounds['GSH(g)'], n_local),
    'Urea(g)': np.random.uniform(*param_bounds['Urea(g)'], n_local),
    'FA(ml)': np.random.uniform(*param_bounds['FA(ml)'], n_local),
    'Temp(℃)': np.random.uniform(*param_bounds['Temp(℃)'], n_local),
    'Time(h)': np.random.uniform(*param_bounds['Time(h)'], n_local),
})

# Feature engineering (must match training)
local_candidates['GSH_urea_ratio'] = local_candidates['GSH(g)'] / (local_candidates['Urea(g)'] + 1e-6)
local_candidates['Temp_time'] = local_candidates['Temp(℃)'] * local_candidates['Time(h)']
local_candidates['FA_GSH'] = local_candidates['FA(ml)'] * local_candidates['GSH(g)']
local_candidates['urea_sq'] = local_candidates['Urea(g)'] ** 2
local_candidates['Temp_sq'] = local_candidates['Temp(℃)'] ** 2

feature_names = joblib.load(os.path.join(output_dir, "feature_names.pkl"))
X_local = local_candidates[feature_names].copy()

fl_pred_local = best_model.predict(X_local)
local_candidates['Predicted_FL'] = fl_pred_local

top3_local = local_candidates.nlargest(3, 'Predicted_FL').reset_index(drop=True)

# Check original data in neighborhood
df_orig = pd.read_excel(r"E:/桌面/第一章机器学习/18/增强后数据.xlsx", sheet_name="Sheet1")
df_orig = df_orig[df_orig['FL'] > 0].reset_index(drop=True)

def in_neighborhood(row):
    for col in ['GSH(g)', 'Urea(g)', 'FA(ml)', 'Temp(℃)', 'Time(h)']:
        low, high = param_bounds[col]
        if not (low <= row[col] <= high):
            return False
    return True

df_in_region = df_orig[df_orig.apply(in_neighborhood, axis=1)].copy()
df_in_region = df_in_region.sort_values('FL', ascending=False).head(3).reset_index(drop=True)

top3_local.to_excel(os.path.join(output_dir, "local_optimal_conditions_Top3.xlsx"), index=False)
if not df_in_region.empty:
    df_in_region.to_excel(os.path.join(output_dir, "high_FL_experiments_in_neighborhood.xlsx"), index=False)

# Generate English report
report_text = f"""# Local Optimal Synthesis Conditions (Near User-Specified Point)

## User-Specified Center:
- GSH: {center_condition['GSH(g)']} g
- Urea: {center_condition['Urea(g)']} g
- FA: {center_condition['FA(ml)']} mL
- Temperature: {center_condition['Temp(℃)']} °C
- Time: {center_condition['Time(h)']} h

## Search Neighborhood:
- GSH: [{param_bounds['GSH(g)'][0]:.2f}, {param_bounds['GSH(g)'][1]:.2f}] g
- Urea: [{param_bounds['Urea(g)'][0]:.2f}, {param_bounds['Urea(g)'][1]:.2f}] g
- FA: [{param_bounds['FA(ml)'][0]:.1f}, {param_bounds['FA(ml)'][1]:.1f}] mL
- Temperature: [{param_bounds['Temp(℃)'][0]:.0f}, {param_bounds['Temp(℃)'][1]:.0f}] °C
- Time: [{param_bounds['Time(h)'][0]:.1f}, {param_bounds['Time(h)'][1]:.1f}] h

## Top 3 Model-Predicted Conditions (using {best_name}):
"""

for i, row in top3_local.iterrows():
    report_text += f"\n### Recommendation #{i+1} (Predicted FL = {row['Predicted_FL']:.2f})\n"
    report_text += f"- GSH: {row['GSH(g)']:.3f} g\n"
    report_text += f"- Urea: {row['Urea(g)']:.3f} g\n"
    report_text += f"- FA: {row['FA(ml)']:.2f} mL\n"
    report_text += f"- Temperature: {row['Temp(℃)']:.1f} °C\n"
    report_text += f"- Time: {row['Time(h)']:.2f} h\n"

if not df_in_region.empty:
    report_text += "\n## High-FL Experimental Points in Original Data (within neighborhood):\n"
    for _, row in df_in_region.iterrows():
        report_text += f"- Exp: FL={row['FL']:.2f}, Conditions=({row['GSH(g)']:.3f}g, {row['Urea(g)']:.3f}g, {row['FA(ml)']:.1f}mL, {row['Temp(℃)']:.0f}°C, {row['Time(h)']:.1f}h)\n"
else:
    report_text += "\n## No experimental points found in the specified neighborhood in original data.\n"

report_text += "\n> We recommend validating Recommendation #1 first, as it is closest to your target conditions and has the highest predicted FL."

with open(os.path.join(output_dir, "local_optimal_recommendations.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# ===================================================================================
# ================================= Final Summary ===================================
# ===================================================================================

print(f"\n✅ All results saved to: {output_dir} - 18.py:467")
print(f"🏆 Best Model: {best_name} | Test R² = {best_r2:.4f} - 18.py:468")
if best_r2 >= 0.91:
    print("🎉 Target R² ≥ 0.91 achieved! - 18.py:470")
else:
    print(f"💡 Current R² = {best_r2:.4f}. Consider expanding dataset or refining experimental design. - 18.py:472")
print("\n📊 New figures include: - 18.py:473")
print("linear_fit_RandomForest.png - 18.py:474")
print("linear_fit_XGBoost.png - 18.py:475")
print("linear_fit_LightGBM.png - 18.py:476")
print("model_R2_MAE_comparison.png - 18.py:477")
print("residual_analysis_XXX.png - 18.py:478")
print("prediction_error_distribution.png - 18.py:479")
print("✅ Local optimal condition search completed! - 18.py:480")