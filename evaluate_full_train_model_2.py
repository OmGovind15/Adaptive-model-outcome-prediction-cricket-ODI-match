import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# Directory where your saved unified models are located
MODELS_DIR = "unified_model_analysis_innings2"

# Path to the raw test data folder
TEST_DATA_FOLDER = r"C:\Users\91878\Downloads\dataset_altogether\encoded_test_inning2"

# Directory to save the new analysis results
OUTPUT_DIR = "per_over_evaluation_of_unified_models_in2"


# --- Helper Functions & Classes (Copied from the training script for consistency) ---

def load_data_from_folder(folder_path):
    """Loads and concatenates all CSV files from a folder."""
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

def feature_engineering(data):
    """Re-creates the features exactly as they were during training."""
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['required_run_rate'] = data['trail_runs'] / (49 - data['over']).clip(lower=0.1)
    data['rate_gap'] = data['required_run_rate'] - data['run_rate']
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = (300 * data['rate_gap']) / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features = ['over', 'trail_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'rate_gap', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
                'weighted_bowling_average', 'venue_index', 'partnership']
    return data, features

# Required for loading the PyTorch model
class SimpleWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)


# --- Main Evaluation Logic ---

def evaluate_models_over_by_over():
    """Loads trained models and evaluates them on the test set, over by over."""
    
    # 1. Setup and Data Loading
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading and preparing test data from: {TEST_DATA_FOLDER}")
    test_df_raw = load_data_from_folder(TEST_DATA_FOLDER)

    # --- THIS IS THE MODIFIED PART ---
    # Filter out the 50th over to align with the training data
    print("Excluding 50th over (over == 49) from evaluation for a fair test.")
    test_df_raw = test_df_raw[test_df_raw['over'] < 49].copy()
    # --- END OF MODIFICATION ---

    test_df, all_features = feature_engineering(test_df_raw)
    test_df.dropna(subset=['match_result'] + all_features, inplace=True)
    print("Test data prepared.")

    # 2. Load all trained models
    print(f"\nLoading trained models from: {MODELS_DIR}")
    loaded_models = {}
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.pkl', '.pt'))]

    for file_name in model_files:
        model_name = file_name.split('_unified_model')[0]
        file_path = os.path.join(MODELS_DIR, file_name)
        
        if file_name.endswith('.pkl'):
            # Load Scikit-learn models
            model_data = joblib.load(file_path)
            loaded_models[model_name] = model_data
            print(f"  - Loaded {model_name} (ML Model)")
        
        elif file_name.endswith('.pt'):
            # Load PyTorch model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(file_path, map_location=device, weights_only=False)
            model = SimpleWinPredictor(input_dim=checkpoint['input_dim'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            loaded_models[model_name] = {
                'model': model,
                'selected_features': checkpoint['selected_features'],
                'device': device
            }
            print(f"  - Loaded {model_name} (NN Model)")

    # 3. Perform Per-Over Evaluation
    print("\nStarting per-over evaluation...")
    per_over_accuracies = {}
    overs_to_evaluate = sorted(test_df['over'].unique())

    for model_name, model_info in loaded_models.items():
        per_over_accuracies[model_name] = {}
        scaler = model_info.get('scaler') # Get scaler if it's an ML model

        for over in overs_to_evaluate:
            over_data = test_df[test_df['over'] == over].copy()
            if over_data.empty:
                continue

            y_true = over_data['match_result']
            
            # Prepare features (scaling for ML, selection for all)
            X_over_full = over_data[all_features]
            if scaler: # For sklearn models
                X_over_scaled = pd.DataFrame(scaler.transform(X_over_full), columns=all_features)
                X_test = X_over_scaled[model_info['selected_features']]
                
                # Predict and calculate accuracy
                model = model_info['model']
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_true, y_pred)
                per_over_accuracies[model_name][over] = acc
            
            else: # For PyTorch model (assumes no scaler was saved with it)
                # We need a scaler. Let's load one from an ML model.
                temp_scaler = loaded_models['RandomForest']['scaler']
                X_over_scaled = pd.DataFrame(temp_scaler.transform(X_over_full), columns=all_features)
                X_test_tensor = torch.tensor(X_over_scaled[model_info['selected_features']].values, dtype=torch.float32).to(model_info['device'])
                
                # Predict and calculate accuracy
                model = model_info['model']
                with torch.no_grad():
                    preds = model(X_test_tensor).squeeze()
                    y_pred = (preds > 0.5).long().cpu().numpy()
                acc = accuracy_score(y_true, y_pred)
                per_over_accuracies[model_name][over] = acc

    # 4. Analyze and Report Results
    results_df = pd.DataFrame(per_over_accuracies).sort_index()
    results_df.index.name = 'Over'
    
    # Calculate stability (standard deviation)
    stability = results_df.std().sort_values(ascending=True)

    print("\n--- Per-Over Accuracy Results ---")
    print(results_df)
    results_df.to_excel(os.path.join(OUTPUT_DIR, "per_over_accuracies_of_unified_models.xlsx"))
    
    print("\n--- Model Stability (Standard Deviation of Accuracy Across Overs) ---")
    print("(Lower is better, indicating more consistent performance)")
    print(stability)
    stability.to_csv(os.path.join(OUTPUT_DIR, "model_stability_summary.csv"))

    # 5. Visualize Results
    # Line plot of accuracy vs. over
    results_df.plot(figsize=(16, 9), style='-o', markersize=5)
    plt.title('Unified Model Performance Across Each Over', fontsize=16, fontweight='bold')
    plt.xlabel('Over', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xticks(overs_to_evaluate)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_over.png"), dpi=300)

    # Bar plot of standard deviation
    plt.figure(figsize=(12, 7))
    sns.barplot(x=stability.index, y=stability.values, palette='coolwarm_r')
    plt.title('Model Stability (Standard Deviation of Per-Over Accuracy)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stability_barplot.png"), dpi=300)
    
    print(f"\n✅ Analysis complete. All results and plots saved in '{OUTPUT_DIR}' directory.")
    plt.show()
if __name__ == "__main__":
    evaluate_models_over_by_over()