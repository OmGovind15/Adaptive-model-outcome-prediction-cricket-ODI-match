import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import warnings
import joblib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TRAIN_FOLDER = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
OUTPUT_DIR = "unified_model_analysis_innings2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- DATA & FEATURE FUNCTIONS ---

def load_data_from_folder(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

def feature_engineering(data):
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

# --- NEURAL NETWORK CLASSES AND CV FUNCTION ---

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

def train_unified_nn_cv(X_train_data, y_train_data, device, epochs=50, k_folds=5):
    """
    Trains the unified NN using k-fold CV, logic copied from final_in2.py
    """
    print(f"\nTraining Unified Neural Network with {k_folds}-Fold CV...")
    
    dataset = UnifiedDataset(X_train_data, y_train_data)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    dataset_indices = list(range(len(dataset)))
    
    cv_results = {'fold_val_accs': []}
    best_fold_model = None
    best_overall_val_acc = 0

    # --- MODIFICATION: Added TQDM for k-fold loop ---
    for fold_idx, (train_indices, val_indices) in tqdm(enumerate(kf.split(dataset_indices)), desc="NN CV Folds", total=k_folds):
        # print(f"   Training Fold {fold_idx + 1}/{k_folds}") # No longer needed
        
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=256, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=256, shuffle=False)
        
        input_dim = X_train_data.shape[1]
        model = SimpleWinPredictor(input_dim=input_dim, hidden_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)

        patience, epochs_no_improve, best_val_acc = 5, 0, 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                features, labels = batch_X.to(device), batch_y.to(device)
                preds = model(features).squeeze()
                loss = F.binary_cross_entropy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    features, labels = batch_X.to(device), batch_y.to(device)
                    preds = model(features).squeeze()
                    predicted = (preds > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            if val_acc > best_val_acc:
                best_val_acc, epochs_no_improve = val_acc, 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1

            # (Optional: can remove this print to keep the progress bar clean)
            # if epoch % 10 == 0:
            #     print(f"         Epoch {epoch+1}: Val Acc: {val_acc:.4f}")

            if epochs_no_improve == patience:
                # print(f"         -- Early stopping triggered at epoch {epoch+1}! --")
                break
            scheduler.step(val_acc)

        cv_results['fold_val_accs'].append(best_val_acc)
        
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_fold_model = best_model_state

    # Load the best model state from all folds
    if best_fold_model:
        model.load_state_dict(best_fold_model)
        
    avg_cv_metrics = {'mean_val_acc': np.mean(cv_results['fold_val_accs'])}
    print(f"   Neural Network CV Results: Mean Val Acc: {avg_cv_metrics['mean_val_acc']:.4f}")
    
    return model, avg_cv_metrics

# --- MAIN TRAINING SCRIPT ---

print("Starting Unified Model Training for Innings 2...")

# 1. Load all data
train_data = load_data_from_folder(TRAIN_FOLDER)
print("Excluding 50th over (over == 49) from training.")
train_data = train_data[train_data['over'] < 49].copy() # Keep this filter
train_data, features = feature_engineering(train_data)
train_data.dropna(subset=['match_result'] + features, inplace=True)

print(f"Total training samples: {len(train_data)}")

# 2. Scale data
scaler = StandardScaler()
X_train = train_data[features]
y_train = train_data['match_result']
X_train_scaled = scaler.fit_transform(X_train)

# 3. Define models and GridSearchCV (Copied from final_in2.py)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
param_grid_xgb = {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.2]}
param_grid_lr = {'C': [0.1, 1, 10], 'penalty': ['l2']}
param_grid_lda = {'solver': ['svd', 'lsqr']}
param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
param_grid_nb = {}

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'LDA': LDA(),
    'SVM': SVC(probability=True, random_state=42),
    'NaiveBayes': GaussianNB()
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_searches = [
    GridSearchCV(models['RandomForest'], param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0),
    GridSearchCV(models['XGBoost'], param_grid_xgb, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0),
    GridSearchCV(models['LogisticRegression'], param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0),
    GridSearchCV(models['LDA'], param_grid_lda, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0),
    GridSearchCV(models['SVM'], param_grid_svm, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0),
    GridSearchCV(models['NaiveBayes'], param_grid_nb, cv=kf, scoring='accuracy', n_jobs=-1, verbose=0)
]

# 4. Train and Save ML Models (UPGRADED)
print("\nTraining ML Models with GridSearchCV...")
# --- MODIFICATION: Added TQDM for ML model loop ---
for model_name, grid_search in tqdm(zip(models.keys(), grid_searches), desc="Training ML Models", total=len(models)):
    # print(f"\n-- Fitting {model_name} --") # No longer needed
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n  {model_name} Best CV Score: {grid_search.best_score_:.4f}")
    # print(f"  Best Params: {grid_search.best_params_}") # Optional
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_unified_model.pkl")
    joblib.dump({
        'model': grid_search.best_estimator_,
        'scaler': scaler,
        'selected_features': features,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }, save_path)
    # print(f"Saved {model_name} model to {save_path}") # Optional

# 5. Train and Save Neural Network (UPGRADED)
print("\nTraining Neural Network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_nn = X_train_scaled
y_train_nn = y_train.values

# Call the new CV-based training function
nn_model, nn_cv_results = train_unified_nn_cv(
    X_train_nn, y_train_nn, device, epochs=50, k_folds=5
)

# Save NN Model
nn_save_path = os.path.join(OUTPUT_DIR, "NeuralNetwork_unified_model.pt")
torch.save({
    'model_state_dict': nn_model.state_dict(),
    'selected_features': features,
    'input_dim': len(features)
}, nn_save_path)
print(f"Saved Neural Network model to {nn_save_path}")

print("\n✅ Unified model training for Innings 2 complete.")