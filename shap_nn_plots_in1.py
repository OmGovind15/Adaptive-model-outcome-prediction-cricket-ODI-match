# generate_nn_shap_plots_innings1.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION for 1st INNINGS ---
# This should point to the output directory of your nn_ml_1_partnership_new.py script
BASE_ANALYSIS_DIR = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\comprehensive_analysis_innings1"
MODELS_DIR = os.path.join(BASE_ANALYSIS_DIR, "all_models")
SHAP_OUTPUT_DIR = os.path.join(BASE_ANALYSIS_DIR, "shap_analysis_neural_network")

# These should point to the original data folders used for training
TRAIN_FOLDER = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\training_data_innings1_encoded"
TEST_FOLDER = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\encoded_test_inning1"


os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
print(f"SHAP analysis results will be saved in: {SHAP_OUTPUT_DIR}")


# --- REPLICATION OF NECESSARY FUNCTIONS AND CLASSES ---

def load_data_from_folder(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            match_number = file.split('.')[0]
            match_data['match_number'] = match_number
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def feature_engineering(data):
    """
    Replicates the 1st innings feature engineering from the training script.
    """
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['projected_runs'] = data['run_rate'] * 50
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = 300 / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features = ['cumulative_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'projected_runs', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
                'weighted_bowling_average', 'venue_index', 'partnership']
                
    return data, features

class SimpleWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x_numeric):
        return self.network(x_numeric)

class ShapModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x_numeric):
        x_numeric_3d = x_numeric.unsqueeze(1)
        pred = self.model(x_numeric_3d)
        return pred.squeeze(-1)

def main():
    print("Starting SHAP analysis for 1st Innings Neural Network models...")
    
    # 1. Load and prepare data
    print("Loading and preprocessing data...")
    train_data = load_data_from_folder(TRAIN_FOLDER)
    test_data = load_data_from_folder(TEST_FOLDER)
    if train_data.empty or test_data.empty:
        print("Error: Could not load training or test data. Please check FOLDER paths.")
        return
        
    # Get the full list of possible 1st innings features
    train_data, all_possible_features = feature_engineering(train_data)
    test_data, _ = feature_engineering(test_data)
    
    train_data.dropna(subset=['match_result'] + all_possible_features, inplace=True)
    test_data.dropna(subset=['match_result'] + all_possible_features, inplace=True)
    scaler = StandardScaler()
    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()
    train_data_scaled[all_possible_features] = scaler.fit_transform(train_data[all_possible_features])
    test_data_scaled[all_possible_features] = scaler.transform(test_data[all_possible_features])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load all saved Neural Network models
    print("Loading saved Neural Network models...")
    over_models = {}
    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename.startswith("neural_network_over_") and filename.endswith(".pt"):
            try:
                over = int(filename.split('_')[-1].split('.')[0])
                model_path = os.path.join(MODELS_DIR, filename)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model_state_dict = checkpoint['model_state_dict']
                selected_features = checkpoint['selected_features']
                input_dim = checkpoint['input_dim']
                model = SimpleWinPredictor(input_dim=input_dim, hidden_dim=48).to(device)
                model.load_state_dict(model_state_dict)
                model.eval()
                over_models[over] = {'model': model, 'features': selected_features}
            except (ValueError, IndexError, KeyError) as e:
                print(f"Warning: Could not load model from {filename}. Error: {e}")
                continue
    if not over_models:
        print(f"Error: No Neural Network models found in {MODELS_DIR}. Exiting.")
        return
    print(f"Successfully loaded {len(over_models)} NN models for analysis.")

    # 3. Calculate SHAP values for each model
    over_wise_importance = {}
    print("Calculating SHAP values for each over-specific model...")
    for over, model_info in tqdm(sorted(over_models.items()), desc="Processing Overs"):
        model, trained_features = model_info['model'], model_info['features']
        
        train_over_data = train_data_scaled[train_data_scaled['over'] == over]
        test_over_data = test_data_scaled[test_data_scaled['over'] == over]
        if test_over_data.empty or train_over_data.empty or len(test_over_data) < 2:
            continue

        X_test_over = test_over_data[trained_features]
        background_df = train_over_data.sample(min(100, len(train_over_data)), random_state=42)
        X_background_over = background_df[trained_features]
        background_tensor = torch.tensor(X_background_over.values, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_over.values, dtype=torch.float32).to(device)
        shap_wrapper = ShapModelWrapper(model)
        explainer = shap.DeepExplainer(shap_wrapper, background_tensor)
        shap_values_over = explainer.shap_values(test_tensor)
        mean_abs_shap = np.abs(shap_values_over).mean(axis=0)
        
        calculated_shap_values = dict(zip(trained_features, mean_abs_shap))

        full_feature_importance = {feature: 0.0 for feature in all_possible_features}
        full_feature_importance.update(calculated_shap_values)
        over_wise_importance[over] = full_feature_importance

    # 4. Aggregate and Visualize Results
    if not over_wise_importance:
        print("No SHAP values were calculated. Cannot generate plots.")
        return
        
    importance_df = pd.DataFrame(over_wise_importance).T
    importance_df.sort_index(axis=0, inplace=True)
    importance_df.sort_index(axis=1, inplace=True) 
    importance_df = importance_df.astype(float)
    importance_df.index = importance_df.index + 1
    # --- Heatmap Plot ---
    print("Generating over-wise SHAP importance heatmap...")
    # Using compact height: (14, 6)
    plt.figure(figsize=(24, 10)) 
    ax = sns.heatmap( # Capture the axis object
        importance_df.T, cmap="Greens", annot=False, linewidths=.5,
        cbar_kws={'label': 'Mean Absolute SHAP Value (Feature Importance)'},
        vmin=0,       # <-- ADD THIS LINE
        vmax=0.45     # <-- ADD THIS LINE (use 0.45 or your chosen max)
    )
    # Set colorbar label font size
    ax.figure.axes[-1].yaxis.label.set_size(25) 
    ax.figure.axes[-1].tick_params(labelsize=18)
    
    plt.title("1st Innings: NN SHAP Feature Importance (Heatmap)", fontsize=20, fontweight='bold')
    plt.xlabel("Over Number", fontsize=25)
    plt.ylabel("Features", fontsize=25)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    heatmap_path = os.path.join(SHAP_OUTPUT_DIR, "nn_shap_heatmap_over_wise_inn1.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

    # --- Line Plot ---
    smoothed_df = importance_df.rolling(window=3, min_periods=1, center=True).mean()

    print("Generating over-wise SHAP importance line plot...")
    # Using (14, 8) for a good aspect ratio
    plt.figure(figsize=(14, 8)) 
    sns.lineplot(data=smoothed_df, dashes=False, legend="full", palette="tab10")
    
    plt.title("1st Innings: NN SHAP Feature Importance (Smoothed)", fontsize=20, fontweight='bold')
    plt.xlabel("Over Number", fontsize=16)
    plt.ylabel("Mean Absolute SHAP Value (Importance)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=16, fontsize=14)
    plt.tight_layout()
    line_plot_path = os.path.join(SHAP_OUTPUT_DIR, "nn_shap_lineplot_over_wise_inn1.png")
    plt.savefig(line_plot_path, dpi=300)
    plt.close()
    print(f"Line plot saved to: {line_plot_path}")
    
    csv_path = os.path.join(SHAP_OUTPUT_DIR, "nn_shap_values_by_over_inn1.csv")
    importance_df.to_csv(csv_path)
    
    print(f"Aggregated SHAP values saved to: {csv_path}")

    print("\nSHAP analysis complete!")

if __name__ == "__main__":
    main()