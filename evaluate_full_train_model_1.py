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
# Directory where your saved 1st INNINGS unified models are located
MODELS_DIR = "unified_model_analysis_innings1"

# Path to the raw 1st INNINGS test data folder
TEST_DATA_FOLDER = r"C:\Users\91878\Downloads\dataset_altogether\encoded_test_inning1"

# Directory to save the new analysis results
OUTPUT_DIR = "per_over_evaluation_of_unified_models_innings1"


# --- Helper Functions & Classes (Copied from the 1st INNINGS training script) ---

def load_data_from_folder(folder_path):
    """Loads and concatenates all CSV files from a folder."""
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

def feature_engineering(data):
    """Re-creates the features exactly as they were during 1st INNINGS training."""
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['projected_runs'] = data['run_rate'] * 50
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = 300 / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features = ['over', 'cumulative_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'projected_runs', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
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
    """Loads trained 1st innings models and evaluates them on the test set, over by over."""
    
    # 1. Setup and Data Loading
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading and preparing 1st innings test data from: {TEST_DATA_FOLDER}")
    test_df_raw = load_data_from_folder(TEST_DATA_FOLDER)
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
            model_data = joblib.load(file_path)
            loaded_models[model_name] = model_data
            print(f"  - Loaded {model_name} (ML Model)")
        
        elif file_name.endswith('.pt'):
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
    print("\nStarting per-over evaluation for Innings 1...")
    per_over_accuracies = {}
    overs_to_evaluate = sorted(test_df['over'].unique())

    for model_name, model_info in loaded_models.items():
        per_over_accuracies[model_name] = {}
        scaler = model_info.get('scaler') 

        for over in overs_to_evaluate:
            over_data = test_df[test_df['over'] == over].copy()
            if over_data.empty:
                continue

            y_true = over_data['match_result']
            
            X_over_full = over_data[all_features]
            if scaler: # For sklearn models
                X_over_scaled = pd.DataFrame(scaler.transform(X_over_full), columns=all_features)
                X_test = X_over_scaled[model_info['selected_features']]
                
                model = model_info['model']
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_true, y_pred)
                per_over_accuracies[model_name][over] = acc
            
            else: # For PyTorch model
                temp_scaler = loaded_models['RandomForest']['scaler']
                X_over_scaled = pd.DataFrame(temp_scaler.transform(X_over_full), columns=all_features)
                X_test_tensor = torch.tensor(X_over_scaled[model_info['selected_features']].values, dtype=torch.float32).to(model_info['device'])
                
                model = model_info['model']
                with torch.no_grad():
                    preds = model(X_test_tensor).squeeze()
                    y_pred = (preds > 0.5).long().cpu().numpy()
                acc = accuracy_score(y_true, y_pred)
                per_over_accuracies[model_name][over] = acc

    # 4. Analyze and Report Results
    results_df = pd.DataFrame(per_over_accuracies).sort_index()
    results_df.index.name = 'Over'
    
    stability = results_df.std().sort_values(ascending=True)

    print("\n--- Per-Over Accuracy Results (Innings 1) ---")
    print(results_df)
    results_df.to_excel(os.path.join(OUTPUT_DIR, "per_over_accuracies_innings1.xlsx"))
    
    # --- THIS IS THE NEW SECTION ---
    # Calculate mean accuracy for each model
    mean_accuracies = results_df.mean().sort_values(ascending=False)

    print("\n--- Mean Test Accuracy Across All Overs (Innings 1) ---")
    print("(Higher is better, indicating better overall performance)")
    print(mean_accuracies)
    mean_accuracies.to_csv(os.path.join(OUTPUT_DIR, "model_mean_accuracy_summary_innings1.csv"))
    # --- END OF NEW SECTION ---
    
    print("\n--- Model Stability (Innings 1) ---")
    print("(Lower is better, indicating more consistent performance)")
    print(stability)
    stability.to_csv(os.path.join(OUTPUT_DIR, "model_stability_summary_innings1.csv"))

    # 5. Visualize Results
    results_df.plot(figsize=(16, 9), style='-o', markersize=5)
    plt.title('Unified Model Performance Across Each Over (Innings 1)', fontsize=16, fontweight='bold')
    plt.xlabel('Over', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xticks(overs_to_evaluate)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_over_innings1.png"), dpi=300)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=stability.index, y=stability.values, palette='coolwarm_r')
    plt.title('Model Stability (Std Dev of Per-Over Accuracy) - Innings 1', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stability_barplot_innings1.png"), dpi=300)
    
    print(f"\n✅ Analysis complete. All results and plots saved in '{OUTPUT_DIR}' directory.")
    plt.show()
if __name__ == "__main__":
    evaluate_models_over_by_over()