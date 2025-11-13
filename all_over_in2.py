import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminanceAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import warnings
import joblib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# --- CONFIGURATION (MODIFIED FOR INNINGS 2) ---
TRAIN_FOLDER = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
OUTPUT_DIR = "unified_model_analysis_innings2" # Dir for the UNIFIED models
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- DATA & FEATURE FUNCTIONS (Copied from your script) ---

def load_data_from_folder(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

# --- MODIFIED FOR INNINGS 2 ---
def feature_engineering(data):
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    # Use clip(lower=0.1) to avoid division by zero if over 49 is present
    data['required_run_rate'] = data['trail_runs'] / (49 - data['over']).clip(lower=0.1) 
    data['rate_gap'] = data['required_run_rate'] - data['run_rate']
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = (300 * data['rate_gap']) / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # IMPORTANT: 'over' is included as a feature
    # These are the Innings 2 features
    features = ['over', 'trail_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'rate_gap', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
                'weighted_bowling_average', 'venue_index', 'partnership']
    return data, features

# --- NEURAL NETWORK CLASSES (Copied from your script) ---

class UnifiedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# --- MAIN TRAINING SCRIPT ---

print("Starting Unified Model Training for Innings 2...")

# 1. Load all data
train_data = load_data_from_folder(TRAIN_FOLDER)

# --- MODIFIED: Filter out 50th over (over == 49) ---
print("Excluding 50th over (over == 49) from training.")
train_data = train_data[train_data['over'] < 49].copy()

train_data, features = feature_engineering(train_data)
train_data.dropna(subset=['match_result'] + features, inplace=True)

print(f"Total training samples: {len(train_data)}")

# 2. Scale data
scaler = StandardScaler()
X_train = train_data[features]
y_train = train_data['match_result']
X_train_scaled = scaler.fit_transform(X_train)

# 3. Define models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'LDA': LDA(),
    'SVM': SVC(probability=True, random_state=42),
    'NaiveBayes': GaussianNB()
}

# 4. Train and Save ML Models
print("\nTraining ML Models...")
for name, model in tqdm(models.items(), desc="Training ML Models"):
    model.fit(X_train_scaled, y_train)
    
    save_path = os.path.join(OUTPUT_DIR, f"{name}_unified_model.pkl")
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'selected_features': features # All features are used
    }, save_path)
    print(f"Saved {name} model to {save_path}")

# 5. Train and Save Neural Network
print("\nTraining Neural Network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NN uses the same scaled data
X_train_nn = X_train_scaled
y_train_nn = y_train.values

train_dataset = UnifiedDataset(X_train_nn, y_train_nn)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

input_dim = len(features)
nn_model = SimpleWinPredictor(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 30 # Reduced epochs as we have more data per epoch

for epoch in range(epochs):
    nn_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        preds = nn_model(batch_X).squeeze()
        loss = F.binary_cross_entropy(preds, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save NN Model
nn_save_path = os.path.join(OUTPUT_DIR, "NeuralNetwork_unified_model.pt")
torch.save({
    'model_state_dict': nn_model.state_dict(),
    'selected_features': features,
    'input_dim': input_dim
    # Note: Save the scaler with an ML model and reuse it during evaluation
}, nn_save_path)
print(f"Saved Neural Network model to {nn_save_path}")

print("\n✅ Unified model training for Innings 2 complete.")