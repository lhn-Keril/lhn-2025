import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# Set font for English plots (remove Chinese font dependency)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# File path configuration - modify to your "数据扩增3" directory
BASE_DIR = r"E:\桌面\第一章机器学习\数据扩增4"
EXCEL_PATH = os.path.join(BASE_DIR, "4.xlsx")  # Assume Excel file is here
OUTPUT_DIR = BASE_DIR  # Output directory same as input
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Enhanced_Data.xlsx")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_excel_data():
    """
    Load data from Excel file, auto-detect column names.
    """
    try:
        print(f"Loading data from {EXCEL_PATH}...  3.py:33 - 4.py:33")
        df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
        
        print("Excel columns: - 4.py:36", df.columns.tolist())
        print("Data shape: - 4.py:37", df.shape)
        print("\nFirst 5 rows: - 4.py:38")
        print(df.head())
        
        # Auto-detect required columns
        column_mapping = {}
        for col in df.columns:
            col_clean = str(col).lower().replace(' ', '').replace('(', '').replace(')', '')
            if 'gsh' in col_clean:
                column_mapping['GSH'] = col
            elif 'urea' in col_clean:
                column_mapping['Urea'] = col
            elif 'fa' in col_clean or '甲酰胺' in col_clean:
                column_mapping['FA'] = col
            elif 'temp' in col_clean:
                column_mapping['Temp'] = col
            elif 'time' in col_clean:
                column_mapping['Time'] = col
            elif 'fl' in col_clean or '荧光' in col_clean:
                column_mapping['FL'] = col
        
        print("\nDetected column mapping: - 4.py:58", column_mapping)
        
        required_columns = ['GSH', 'Urea', 'FA', 'Temp', 'Time', 'FL']
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            print(f"Missing required columns: {missing_columns} - 4.py:63")
            print("Please check your Excel column names. - 4.py:64")
            return None
        
        features_data = pd.DataFrame({
            'GSH(g)': df[column_mapping['GSH']],
            'Urea(g)': df[column_mapping['Urea']],
            'FA(ml)': df[column_mapping['FA']],
            'Temp(℃)': df[column_mapping['Temp']],
            'Time(h)': df[column_mapping['Time']],
            'FL': df[column_mapping['FL']]
        })
        
        features_data = features_data.dropna()
        for col in features_data.columns:
            features_data[col] = pd.to_numeric(features_data[col], errors='coerce')
        features_data = features_data.dropna()
        
        print(f"\nSuccessfully loaded {len(features_data)} records - 4.py:81")
        print("Processed columns: - 4.py:82", features_data.columns.tolist())
        print("\nFirst 5 processed rows: - 4.py:83")
        print(features_data.head())
        print("\nData types: - 4.py:85")
        print(features_data.dtypes)
        
        return features_data
        
    except Exception as e:
        print(f"Error loading Excel file: {e} - 4.py:91")
        return None

def manual_column_selection():
    """
    Manual column selection if auto-detection fails.
    """
    try:
        print("\nAutodetection failed, trying manual loading... - 4.py:99")
        df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
        
        print("\nAvailable columns: - 4.py:102")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col} - 4.py:104")
        
        if len(df.columns) >= 6:
            features_data = pd.DataFrame({
                'GSH(g)': df.iloc[1:, 1],
                'Urea(g)': df.iloc[1:, 2],
                'FA(ml)': df.iloc[1:, 3],
                'Temp(℃)': df.iloc[1:, 4],
                'Time(h)': df.iloc[1:, 5],
                'FL': df.iloc[1:, 7]
            })
            
            features_data = features_data.dropna()
            for col in features_data.columns:
                features_data[col] = pd.to_numeric(features_data[col], errors='coerce')
            features_data = features_data.dropna()
            print(f"Manual loading succeeded: {len(features_data)} records - 4.py:120")
            return features_data
        else:
            print("Insufficient columns in Excel file - 4.py:123")
            return None
            
    except Exception as e:
        print(f"Manual loading failed: {e} - 4.py:127")
        return None

def custom_smote(X, n_samples=100):
    X_numeric = X.select_dtypes(include=[np.number])
    if len(X_numeric) < 2:
        print("Too few samples for SMOTE - 4.py:133")
        return pd.DataFrame(columns=X.columns)
    
    n_neighbors = min(5, len(X_numeric) - 1)
    if n_neighbors < 1:
        print("Not enough neighbors for SMOTE - 4.py:138")
        return pd.DataFrame(columns=X.columns)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_numeric)
    new_samples = []
    
    for _ in range(n_samples):
        idx = np.random.randint(len(X_numeric))
        sample = X_numeric.iloc[idx]
        distances, indices = nbrs.kneighbors([sample])
        if len(indices[0]) > 1:
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor = X_numeric.iloc[neighbor_idx]
            ratio = np.random.random()
            new_sample = sample + ratio * (neighbor - sample)
            new_samples.append(new_sample)
    
    if new_samples:
        return pd.DataFrame(new_samples, columns=X_numeric.columns)
    else:
        return pd.DataFrame(columns=X.columns)

def physics_constrained_augmentation(original_data, n_samples=50):
    new_samples = []
    for _ in range(n_samples):
        base = original_data.iloc[np.random.randint(len(original_data))].copy()
        base['GSH(g)'] += np.random.normal(0, 0.02)
        base['Urea(g)'] += np.random.normal(0, 0.05)
        base['FA(ml)'] += np.random.normal(0, 0.3)
        base['Temp(℃)'] += np.random.normal(0, 3)
        base['Time(h)'] += np.random.normal(0, 0.3)
        
        temp_effect = (base['Temp(℃)'] - 160) * 8
        time_effect = (base['Time(h)'] - 8) * 30
        urea_effect = (base['Urea(g)'] - 0.8) * 200
        gsh_effect = (base['GSH(g)'] - 0.21) * 100
        base['FL'] += temp_effect + time_effect + urea_effect + gsh_effect
        
        base['GSH(g)'] = max(0.05, min(0.35, base['GSH(g)']))
        base['Urea(g)'] = max(0, min(1.6, base['Urea']))
        base['FA(ml)'] = max(5, min(18, base['FA(ml)']))
        base['Temp(℃)'] = max(120, min(200, base['Temp(℃)']))
        base['Time(h)'] = max(1, min(10, base['Time(h)']))
        base['FL'] = max(0, min(5000, base['FL']))
        new_samples.append(base)
    
    return pd.DataFrame(new_samples)

def gmm_augmentation(data, n_components=3, n_new_samples=100):
    data_numeric = data.select_dtypes(include=[np.number])
    if len(data_numeric) < n_components:
        print("Too few samples for GMM - 4.py:189")
        return pd.DataFrame(columns=data.columns)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)
    n_comp = min(n_components, len(data_numeric) // 3)
    if n_comp < 1:
        n_comp = 1
    
    gmm = GaussianMixture(n_components=n_comp, random_state=42)
    gmm.fit(data_scaled)
    generated_scaled, _ = gmm.sample(n_new_samples)
    generated_data = scaler.inverse_transform(generated_scaled)
    return pd.DataFrame(generated_data, columns=data_numeric.columns)

def random_perturbation_augmentation(original_data, n_samples=50):
    new_samples = []
    for _ in range(n_samples):
        base = original_data.iloc[np.random.randint(len(original_data))].copy()
        noise_scale = 0.05
        for col in base.index:
            if col != 'FL':
                base[col] = base[col] * (1 + np.random.normal(0, noise_scale))
        base['FL'] = base['FL'] * (1 + np.random.normal(0, noise_scale * 0.5))
        new_samples.append(base)
    return pd.DataFrame(new_samples)

def simple_interpolation_augmentation(original_data, n_samples=50):
    new_samples = []
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(len(original_data), 2, replace=False)
        s1, s2 = original_data.iloc[idx1], original_data.iloc[idx2]
        ratio = np.random.random()
        new_samples.append(s1 + ratio * (s2 - s1))
    return pd.DataFrame(new_samples)

def validate_augmented_data(original_data, augmented_data):
    print("\n - 4.py:226" + "="*50)
    print("Data Quality Validation Report - 4.py:227")
    print("= - 4.py:228"*50)
    print(f"Original data size: {len(original_data)} - 4.py:229")
    print(f"Augmented data size: {len(augmented_data)} - 4.py:230")
    if len(original_data) > 0:
        print(f"Augmentation factor: {len(augmented_data)/len(original_data):.2f}x - 4.py:232")
    
    print("\nOriginal data statistics: - 4.py:234")
    print(original_data.describe())
    print("\nAugmented data statistics: - 4.py:236")
    print(augmented_data.describe())

def visualize_comparison(original_data, augmented_data):
    try:
        if len(original_data) == 0 or len(augmented_data) == 0:
            print("Data is empty, skipping visualization - 4.py:242")
            return

        # Use English-friendly font
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Original vs Augmented Data Distribution', fontsize=16, fontweight='bold')

        features = original_data.columns.tolist()
        for i, feature in enumerate(features):
            if i >= 6:
                break
            row, col = i // 3, i % 3
            axes[row, col].hist(original_data[feature], alpha=0.7, label='Original',
                                bins=15, color='blue', edgecolor='black', density=True)
            axes[row, col].hist(augmented_data[feature], alpha=0.7, label='Augmented',
                                bins=15, color='red', edgecolor='black', density=True)
            axes[row, col].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Data_Distribution_Comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Correlation heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(original_data.corr(), annot=True, cmap='coolwarm', ax=ax1, fmt=".2f")
        ax1.set_title('Original Data Correlation', fontweight='bold')
        sns.heatmap(augmented_data.corr(), annot=True, cmap='coolwarm', ax=ax2, fmt=".2f")
        ax2.set_title('Augmented Data Correlation', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Correlation_Comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Visualization error: {e} - 4.py:282")

def main():
    print("Starting data augmentation pipeline... - 4.py:285")
    print(f"Input file: {EXCEL_PATH} - 4.py:286")
    print(f"Output directory: {OUTPUT_DIR} - 4.py:287")
    
    original_data = load_excel_data()
    if original_data is None:
        print("\nTrying manual column selection... - 4.py:291")
        original_data = manual_column_selection()
    
    if original_data is None or len(original_data) == 0:
        print("Failed to load data. Exiting. - 4.py:295")
        return None, None
    
    print(f"\nSuccessfully loaded {len(original_data)} original records - 4.py:298")
    
    max_total_samples = 250
    original_count = len(original_data)
    max_new_samples = max_total_samples - original_count
    
    if max_new_samples <= 0:
        print(f"Original data ({original_count}) already meets/exceeds target ({max_total_samples}). No augmentation needed. - 4.py:305")
        augmented_data = original_data.copy()
    else:
        print(f"Target: generate {max_new_samples} new samples to reach total of {max_total_samples} - 4.py:308")
        
        if original_count < 10:
            method_samples = {
                'simple_interpolation': min(50, max_new_samples),
                'random_perturbation': min(50, max_new_samples)
            }
        else:
            base = min(30, max_new_samples // 3)
            method_samples = {
                'physics_constrained': base,
                'random_perturbation': base,
                'simple_interpolation': base
            }
        
        augmented_dfs = [original_data.copy()]
        total_generated = 0

        # Physics-constrained
        if method_samples.get('physics_constrained', 0) > 0:
            try:
                print("Generating data via physicsconstrained augmentation... - 4.py:329")
                phy_data = physics_constrained_augmentation(original_data, method_samples['physics_constrained'])
                if len(phy_data) > 0:
                    augmented_dfs.append(phy_data)
                    total_generated += len(phy_data)
                    print(f"Generated {len(phy_data)} samples via physicsconstrained method - 4.py:334")
            except Exception as e:
                print(f"Physicsconstrained augmentation failed: {e} - 4.py:336")

        # Random perturbation
        if method_samples.get('random_perturbation', 0) > 0:
            try:
                print("Generating data via random perturbation... - 4.py:341")
                pert_data = random_perturbation_augmentation(original_data, method_samples['random_perturbation'])
                if len(pert_data) > 0:
                    augmented_dfs.append(pert_data)
                    total_generated += len(pert_data)
                    print(f"Generated {len(pert_data)} samples via random perturbation - 4.py:346")
            except Exception as e:
                print(f"Random perturbation failed: {e} - 4.py:348")

        # Simple interpolation
        if method_samples.get('simple_interpolation', 0) > 0:
            try:
                print("Generating data via simple interpolation... - 4.py:353")
                interp_data = simple_interpolation_augmentation(original_data, method_samples['simple_interpolation'])
                if len(interp_data) > 0:
                    augmented_dfs.append(interp_data)
                    total_generated += len(interp_data)
                    print(f"Generated {len(interp_data)} samples via simple interpolation - 4.py:358")
            except Exception as e:
                print(f"Simple interpolation failed: {e} - 4.py:360")

        # Fill remaining if needed
        remaining = max_new_samples - total_generated
        if remaining > 0:
            print(f"Still need {remaining} more samples - 4.py:365")
            if original_count >= 3:
                try:
                    print("Trying SMOTE... - 4.py:368")
                    smote_data = custom_smote(original_data, n_samples=min(remaining, 30))
                    if len(smote_data) > 0:
                        augmented_dfs.append(smote_data)
                        total_generated += len(smote_data)
                        print(f"SMOTE generated {len(smote_data)} samples - 4.py:373")
                except Exception as e:
                    print(f"SMOTE failed: {e} - 4.py:375")
            if total_generated < max_new_samples:
                extra = max_new_samples - total_generated
                if extra > 0:
                    try:
                        print(f"Generating {extra} extra samples via interpolation... - 4.py:380")
                        extra_data = simple_interpolation_augmentation(original_data, n_samples=extra)
                        if len(extra_data) > 0:
                            augmented_dfs.append(extra_data)
                            print(f"Extra interpolation generated {len(extra_data)} samples - 4.py:384")
                    except Exception as e:
                        print(f"Extra interpolation failed: {e} - 4.py:386")

        # Combine and clean
        if len(augmented_dfs) > 1:
            augmented_data = pd.concat(augmented_dfs, ignore_index=True)
            augmented_data = augmented_data.drop_duplicates()
            if len(augmented_data) > max_total_samples:
                print(f"Sampling down from {len(augmented_data)} to {max_total_samples} - 4.py:393")
                augmented_data = augmented_data.sample(n=max_total_samples, random_state=42).reset_index(drop=True)
        else:
            augmented_data = original_data.copy()
            print("Warning: all augmentation methods failed; using original data only - 4.py:397")

    # Final bounds filtering
    augmented_data = augmented_data[
        (augmented_data['FL'] >= 0) & (augmented_data['FL'] <= 5000) &
        (augmented_data['GSH(g)'] >= 0.05) & (augmented_data['GSH(g)'] <= 0.35) &
        (augmented_data['Urea(g)'] >= 0) & (augmented_data['Urea(g)'] <= 1.6) &
        (augmented_data['FA(ml)'] >= 5) & (augmented_data['FA(ml)'] <= 18) &
        (augmented_data['Temp(℃)'] >= 120) & (augmented_data['Temp(℃)'] <= 200) &
        (augmented_data['Time(h)'] >= 1) & (augmented_data['Time(h)'] <= 10)
    ].copy()

    validate_augmented_data(original_data, augmented_data)

    # Save results
    try:
        save_data = augmented_data.copy()
        save_data.columns = ['GSH(g)', 'Urea(g)', 'FA(ml)', 'Temp(℃)', 'Time(h)', 'FL']
        save_data.to_excel(OUTPUT_PATH, index=False)
        print(f"\nEnhanced data saved to: {OUTPUT_PATH} - 4.py:416")

        comparison_path = os.path.join(OUTPUT_DIR, "Data_Comparison_Report.xlsx")
        with pd.ExcelWriter(comparison_path) as writer:
            original_data.to_excel(writer, sheet_name='Original_Data', index=False)
            augmented_data.to_excel(writer, sheet_name='Augmented_Data', index=False)
        print(f"Comparison report saved to: {comparison_path} - 4.py:422")

        csv_path = os.path.join(OUTPUT_DIR, "Augmented_Data.csv")
        save_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV backup saved to: {csv_path} - 4.py:426")

    except Exception as e:
        print(f"Error saving files: {e} - 4.py:429")

    print("\nGenerating visualizations... - 4.py:431")
    visualize_comparison(original_data, augmented_data)

    print("\n - 4.py:434" + "="*50)
    print("Data augmentation completed successfully! - 4.py:435")
    print(f"Original data: {len(original_data)} records - 4.py:436")
    print(f"Augmented data: {len(augmented_data)} records - 4.py:437")
    if len(original_data) > 0:
        print(f"Augmentation factor: {len(augmented_data)/len(original_data):.2f}x - 4.py:439")
    print("= - 4.py:440"*50)

    return original_data, augmented_data

if __name__ == "__main__":
    result = main()
    if result is not None:
        print("Script executed successfully! - 4.py:447")
    else:
        print("Script failed to execute. - 4.py:449")