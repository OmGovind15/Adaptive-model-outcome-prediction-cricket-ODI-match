import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import random
import ast
from collections import Counter, defaultdict
import json

# Neural Network imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def safe_literal_eval(s):
    try:
        if not isinstance(s, str):
            return s
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s

# Define folder paths
train_folder = r"/home/intern/Documents/SWAT-8677/__pycache__/conf/dataset_ckt/dataset_altogether/training_data_innings1_encoded"
test_folder = r"/home/intern/Documents/SWAT-8677/__pycache__/conf/dataset_ckt/dataset_altogether/encoded_test_inning1"

# Enhanced Feature Selection Functions
def select_features_for_model(X, y, model_name, features, k_features=None):
    """
    Enhanced feature selection that works fairly for all models including Naive Bayes
    """
    print(f"    Selecting features for {model_name}...")
    
    if k_features is None:
        k_features = min(max(3, len(features) // 2), len(features))
    
    selected_features = []
    method_used = ""
    
    try:
        if model_name == 'NaiveBayes':
            # For Naive Bayes, use statistical methods
            # Method 1: Mutual Information
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_features)
            selector_mi.fit(X, y)
            selected_features_mi = list(np.array(features)[selector_mi.get_support()])
            
            # Method 2: Chi-squared (for non-negative features)
            try:
                X_non_neg = X - X.min() + 1e-6  # Make all values positive
                selector_chi2 = SelectKBest(score_func=chi2, k=k_features)
                selector_chi2.fit(X_non_neg, y)
                selected_features_chi2 = list(np.array(features)[selector_chi2.get_support()])
            except:
                selected_features_chi2 = selected_features_mi
            
            # Combine both methods (union)
            selected_features = list(set(selected_features_mi + selected_features_chi2))
            if len(selected_features) > k_features:
                # If too many features, use only mutual information
                selected_features = selected_features_mi
            method_used = "Mutual Information + Chi2"
            
        elif model_name in ['RandomForest', 'XGBoost']:
            # Use SelectFromModel for tree-based models
            if model_name == 'RandomForest':
                selector_model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                selector_model = xgb.XGBClassifier(n_estimators=50, random_state=42, 
                                                use_label_encoder=False, eval_metric='logloss')
            
            selector = SelectFromModel(selector_model, max_features=k_features)
            selector.fit(X, y)
            selected_features = list(np.array(features)[selector.get_support()])
            
            if len(selected_features) < 3:  # Fallback to top k features
                importances = selector.estimator_.feature_importances_
                top_indices = np.argsort(importances)[-k_features:]
                selected_features = [features[i] for i in top_indices]
            method_used = "SelectFromModel"
            
        else:
            # For other models, use RFECV with appropriate estimators
            if model_name == 'SVM':
                estimator = SVC(kernel='linear', random_state=42)
            elif model_name == 'LogisticRegression':
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == 'LDA':
                estimator = LDA()
            else:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            
            try:
                rfecv = RFECV(estimator=estimator, step=1, cv=3, scoring="accuracy", 
                            min_features_to_select=3)
                rfecv.fit(X, y)
                selected_features = list(np.array(features)[rfecv.support_])
                method_used = "RFECV"
            except:
                # Fallback to SelectKBest
                selector = SelectKBest(score_func=f_classif, k=k_features)
                selector.fit(X, y)
                selected_features = list(np.array(features)[selector.get_support()])
                method_used = "SelectKBest (fallback)"
    
    except Exception as e:
        print(f"      Feature selection failed for {model_name}: {e}")
        # Ultimate fallback: select top k features using F-score
        try:
            selector = SelectKBest(score_func=f_classif, k=k_features)
            selector.fit(X, y)
            selected_features = list(np.array(features)[selector.get_support()])
            method_used = "F-score (ultimate fallback)"
        except:
            selected_features = features[:k_features]  # Just take first k features
            method_used = "First k features (emergency fallback)"
    
    print(f"      Method used: {method_used}")
    print(f"      Selected {len(selected_features)} features: {selected_features}")
    
    return selected_features, method_used

# Dataset class
class EnhancedOverSpecificDataset(Dataset):
    def __init__(self, data, target_over, selected_features, target_innings=1):
        self.target_over = target_over
        self.target_innings = target_innings
        self.samples = []
        self.selected_features = selected_features
        
        for match_num in data['match_number'].unique():
            match_data = data[data['match_number'] == match_num].reset_index(drop=True)
            over_specific_data = match_data[match_data['over'] == self.target_over]

            if not over_specific_data.empty:
                sample_data = self.process_match_data(over_specific_data)
                self.samples.append((match_num, sample_data))

    def process_match_data(self, df):
        batting_result = 1 - int(df['match_result'].iloc[0])
        x_numeric = torch.tensor(df[self.selected_features].values, dtype=torch.float32)
        return {
            'features': x_numeric,
            'match_result': batting_result,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        match_num, item = self.samples[idx]
        return {
            'match_number': match_num,
            'features': item['features'],
            'match_result': item['match_result'],
        }

def enhanced_pad_collate_fn(batch):
    max_len = max(item['features'].shape[0] for item in batch)

    def pad_seq(seq, pad_val=0):
        pad_size = (0, 0, 0, max_len - seq.shape[0])
        return F.pad(seq, pad_size, value=pad_val)

    features = torch.stack([pad_seq(item['features']) for item in batch])
    match_results = torch.tensor([item['match_result'] for item in batch], dtype=torch.float32)
    
    return {
        'features': features,
        'match_result': match_results,
    }

class SimpleWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
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

def load_data_from_folder(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            match_number = file.split('.')[0]
            match_data['match_number'] = match_number
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

def feature_engineering(data):
    # 1st innings specific feature engineering
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['projected_runs'] = data['run_rate'] * 50
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = 300 / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    
    # Replace inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 1st innings features - no rate_gap or required_run_rate
    features = ['cumulative_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'projected_runs', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
                'weighted_bowling_average', 'venue_index', 'partnership']

    return data, features

def train_neural_network_cv(over, train_data, test_data, device, selected_features, epochs=50, k_folds=5):
    print(f"\nTraining Neural Network for Over {over} with {k_folds}-Fold CV using {len(selected_features)} features.")
    
    dataset = EnhancedOverSpecificDataset(train_data, target_over=over, selected_features=selected_features)
    test_dataset = EnhancedOverSpecificDataset(test_data, target_over=over, selected_features=selected_features)
    
    if len(dataset) == 0:
        print(f"No data found for Over {over}. Skipping...")
        return None, None, None

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=enhanced_pad_collate_fn)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    dataset_indices = list(range(len(dataset)))
    
    cv_results = {'fold_train_accs': [], 'fold_val_accs': [], 'fold_test_accs': [], 'fold_histories': []}
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(dataset_indices)):
        print(f"   Training Fold {fold_idx + 1}/{k_folds}")
        
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=16, shuffle=True, collate_fn=enhanced_pad_collate_fn)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=16, shuffle=False, collate_fn=enhanced_pad_collate_fn)
        
        model = SimpleWinPredictor(input_dim=len(selected_features), hidden_dim=48).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, [weight_decay]=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3, verbose=False)

        patience, epochs_no_improve, best_val_acc = 5, 0, 0
        best_model_state = None
        fold_history = {'train_acc': [], 'val_acc': [], 'test_acc': []}
        
        for epoch in range(epochs):
            model.train()
            train_correct, train_total = 0, 0
            for batch in train_loader:
                features, labels = batch['features'].to(device), batch['match_result'].to(device)
                preds = model(features).squeeze(-1)[:, -1]
                loss = F.binary_cross_entropy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                predicted = (preds > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    features, labels = batch['features'].to(device), batch['match_result'].to(device)
                    preds = model(features).squeeze(-1)[:, -1]
                    predicted = (preds > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            test_correct, test_total = 0, 0
            if test_loader and len(test_loader.dataset) > 0:
                with torch.no_grad():
                    for batch in test_loader:
                        features, labels = batch['features'].to(device), batch['match_result'].to(device)
                        preds = model(features).squeeze(-1)[:, -1]
                        predicted = (preds > 0.5).float()
                        test_correct += (predicted == labels).sum().item()
                        test_total += labels.size(0)

            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            test_acc = test_correct / test_total if test_total > 0 else 0
            
            fold_history['train_acc'].append(train_acc)
            fold_history['val_acc'].append(val_acc)
            fold_history['test_acc'].append(test_acc)
            
            if val_acc > best_val_acc:
                best_val_acc, epochs_no_improve = val_acc, 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                print(f"         Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

            if epochs_no_improve == patience:
                print(f"         -- Early stopping triggered at epoch {epoch+1}! --")
                break
            scheduler.step(val_acc)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        cv_results['fold_train_accs'].append(fold_history['train_acc'][-1] if fold_history['train_acc'] else 0)
        cv_results['fold_val_accs'].append(best_val_acc)
        cv_results['fold_test_accs'].append(fold_history['test_acc'][-1] if fold_history['test_acc'] else 0)
        cv_results['fold_histories'].append(fold_history)
    
    avg_cv_metrics = {
        'mean_train_acc': np.mean(cv_results['fold_train_accs']),
        'std_train_acc': np.std(cv_results['fold_train_accs']),
        'mean_val_acc': np.mean(cv_results['fold_val_accs']),
        'std_val_acc': np.std(cv_results['fold_val_accs']),
        'mean_test_acc': np.mean(cv_results['fold_test_accs']),
        'std_test_acc': np.std(cv_results['fold_test_accs'])
    }
    
    print(f"   Neural Network CV Results: Mean Val Acc: {avg_cv_metrics['mean_val_acc']:.4f} ± {avg_cv_metrics['std_val_acc']:.4f}")
    
    return model, avg_cv_metrics, cv_results

def evaluate_neural_network_detailed(over, model, test_data, device, selected_features):
    dataset = EnhancedOverSpecificDataset(test_data, target_over=over, selected_features=selected_features)
    if len(dataset) == 0:
        return None, None, None

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=enhanced_pad_collate_fn)
    model.eval()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            match_labels = batch['match_result'].to(device)
            match_pred = model(features).squeeze(-1)
            final_predictions = match_pred[:, -1]

            all_predictions.append(final_predictions.cpu())
            all_labels.append(match_labels.cpu())
    
    if not all_predictions:
        return None, None, None

    all_predictions_tensor = torch.cat(all_predictions)
    all_labels_tensor = torch.cat(all_labels)

    predicted_classes = (all_predictions_tensor > 0.5).float()
    correct = (predicted_classes == all_labels_tensor).sum().item()
    total = all_labels_tensor.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, all_predictions_tensor.numpy(), all_labels_tensor.numpy()

def create_comprehensive_plots(results_df, model_usage, cv_histories, output_dir):
    """Create comprehensive plots for analysis"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    all_models = [col.replace('_Test_Accuracy', '') for col in results_df.columns if col.endswith('_Test_Accuracy')]
    
    # 1. Test Accuracies Plot
    plt.figure(figsize=(15, 8))
    for model in all_models:
        if f'{model}_Test_Accuracy' in results_df.columns:
            plt.plot(results_df['Over'], results_df[f'{model}_Test_Accuracy'], 
                    marker='o', label=model, linewidth=2, markersize=6)
    
    plt.plot(results_df['Over'], results_df['Best_Model_Test_Accuracy'], 
             label='Best Model', color='black', linestyle='--', linewidth=3, marker='s')
    
    plt.title('Test Accuracies Across Overs', fontsize=16, fontweight='bold')
    plt.xlabel('Over', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'test_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Validation Accuracies Plot (CV Scores)
    plt.figure(figsize=(15, 8))
    for model in all_models:
        if f'{model}_CV_Score' in results_df.columns:
            plt.plot(results_df['Over'], results_df[f'{model}_CV_Score'], 
                    marker='o', label=model, linewidth=2, markersize=6)
    
    plt.title('Cross-Validation Accuracies Across Overs', fontsize=16, fontweight='bold')
    plt.xlabel('Over', fontsize=12)
    plt.ylabel('CV Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'validation_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Training Accuracies Plot (for Neural Network)
    if 'NeuralNetwork_Train_Accuracy' in results_df.columns:
        plt.figure(figsize=(15, 8))
        plt.plot(results_df['Over'], results_df['NeuralNetwork_Train_Accuracy'], 
                 marker='o', label='Neural Network Training', linewidth=2, markersize=6, color='red')
        plt.plot(results_df['Over'], results_df['NeuralNetwork_CV_Score'], 
                 marker='s', label='Neural Network Validation', linewidth=2, markersize=6, color='blue')
        plt.plot(results_df['Over'], results_df['NeuralNetwork_Test_Accuracy'], 
                 marker='^', label='Neural Network Test', linewidth=2, markersize=6, color='green')
        
        plt.title('Neural Network Training vs Validation vs Test Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Over', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'nn_training_validation_test.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Model Selection Frequency
    model_counts = pd.Series(model_usage).value_counts()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_counts.index, model_counts.values, color=plt.cm.Set3(range(len(model_counts))))
    plt.title('Best Model Selection Frequency', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Times Selected', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_selection_frequency.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Accuracy Distribution Box Plot
    accuracy_data = []
    model_names = []
    
    for model in all_models:
        if f'{model}_Test_Accuracy' in results_df.columns:
            accuracies = results_df[f'{model}_Test_Accuracy'].dropna()
            accuracy_data.extend(accuracies.tolist())
            model_names.extend([model] * len(accuracies))
    
    if accuracy_data:
        plt.figure(figsize=(14, 8))
        df_box = pd.DataFrame({'Model': model_names, 'Test_Accuracy': accuracy_data})
        sns.boxplot(data=df_box, x='Model', y='Test_Accuracy')
        plt.title('Test Accuracy Distribution by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

def save_feature_selection_analysis(feature_selection_log, output_dir):
    """Save detailed feature selection analysis"""
    
    # Create feature selection directory
    fs_dir = os.path.join(output_dir, 'feature_selection_analysis')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. Save detailed feature selection log as JSON
    with open(os.path.join(fs_dir, 'feature_selection_detailed.json'), 'w') as f:
        json.dump(feature_selection_log, f, indent=2)
    
    # 2. Create feature selection summary DataFrame
    fs_summary = []
    for over, models_features in feature_selection_log.items():
        for model, features_info in models_features.items():
            if isinstance(features_info, dict):
                features = features_info.get('features', [])
                method = features_info.get('method', 'Unknown')
            else:
                features = features_info
                method = 'Legacy'
            
            fs_summary.append({
                'Over': over,
                'Model': model,
                'Num_Features_Selected': len(features),
                'Selected_Features': ', '.join(features),
                'Selection_Method': method
            })
    
    fs_df = pd.DataFrame(fs_summary)
    fs_df.to_csv(os.path.join(fs_dir, 'feature_selection_summary.csv'), index=False)
    fs_df.to_excel(os.path.join(fs_dir, 'feature_selection_summary.xlsx'), index=False)
    
    # 3. Feature importance across models and overs
    all_features = set()
    for models_features in feature_selection_log.values():
        for features_info in models_features.values():
            if isinstance(features_info, dict):
                features = features_info.get('features', [])
            else:
                features = features_info
            all_features.update(features)
    
    all_features = sorted(list(all_features))
    
    # Create feature selection matrix
    feature_matrix = []
    for over, models_features in feature_selection_log.items():
        for model, features_info in models_features.items():
            if isinstance(features_info, dict):
                features = features_info.get('features', [])
            else:
                features = features_info
            
            row = {'Over': over, 'Model': model}
            for feature in all_features:
                row[feature] = 1 if feature in features else 0
            feature_matrix.append(row)
    
    feature_matrix_df = pd.DataFrame(feature_matrix)
    feature_matrix_df.to_csv(os.path.join(fs_dir, 'feature_selection_matrix.csv'), index=False)
    
    # 4. Plot feature selection frequency
    feature_counts = defaultdict(int)
    for models_features in feature_selection_log.values():
        for features_info in models_features.values():
            if isinstance(features_info, dict):
                features = features_info.get('features', [])
            else:
                features = features_info
            for feature in features:
                feature_counts[feature] += 1
    
    if feature_counts:
        plt.figure(figsize=(14, 8))
        features_sorted = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*features_sorted)
        
        bars = plt.bar(range(len(features)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        plt.title('Feature Selection Frequency Across All Models and Overs', fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Selection Count', fontsize=12)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, 'feature_selection_frequency.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Create comprehensive output directory structure
    output_dir = "comprehensive_analysis_innings1"
    models_dir = os.path.join(output_dir, "all_models")
    best_models_dir = os.path.join(output_dir, "best_models")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)
    
    print(f"Output directory structure created at: {output_dir}")
    
    # Load and prepare data
    train_data = load_data_from_folder(train_folder)
    test_data = load_data_from_folder(test_folder)

    train_data, features = feature_engineering(train_data)
    test_data, _ = feature_engineering(test_data)
    train_data.dropna(subset=['match_result'] + features, inplace=True)
    test_data.dropna(subset=['match_result'] + features, inplace=True)
    
    # Handle missing values for dismissed_player_index if it exists
    if 'dismissed_player_index' in train_data.columns:
        train_data['dismissed_player_index'].fillna(1605, inplace=True)
    if 'dismissed_player_index' in test_data.columns:
        test_data['dismissed_player_index'].fillna(1605, inplace=True)

    scaler = StandardScaler()
    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()
    train_data_scaled[features] = scaler.fit_transform(train_data[features])
    test_data_scaled[features] = scaler.transform(test_data[features])
    
    # Define parameter grids
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
        GridSearchCV(models['RandomForest'], param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1),
        GridSearchCV(models['XGBoost'], param_grid_xgb, cv=kf, scoring='accuracy', n_jobs=-1),
        GridSearchCV(models['LogisticRegression'], param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1),
        GridSearchCV(models['LDA'], param_grid_lda, cv=kf, scoring='accuracy', n_jobs=-1),
        GridSearchCV(models['SVM'], param_grid_svm, cv=kf, scoring='accuracy', n_jobs=-1),
        GridSearchCV(models['NaiveBayes'], param_grid_nb, cv=kf, scoring='accuracy', n_jobs=-1)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tracking variables
    model_usage = {}
    all_models_keys = list(models.keys()) + ['NeuralNetwork']
    model_accuracies = {model: [] for model in all_models_keys}
    cv_scores = {model_name: [] for model_name in all_models_keys}
    train_accuracies = {model_name: [] for model_name in all_models_keys}
    best_model_accuracies = []
    feature_selection_log = {}
    cv_histories = {}
    all_saved_models = {}
    calibrated_soft_accuracies = {model: [] for model in all_models_keys}
    best_model_soft_accuracies = []
    
    print(f"\n{'='*100}")
    print("STARTING COMPREHENSIVE MODEL TRAINING AND ANALYSIS")
    print(f"{'='*100}")
    
    for over in tqdm(sorted(train_data['over'].unique()), desc="Training models for each over"):
        over = int(over) 

        print(f"\n{'='*100}\nPROCESSING OVER {over}\n{'='*100}")
        
        feature_selection_log[over] = {}
        all_saved_models[over] = {}
        
        train_over_data = train_data_scaled[train_data_scaled['over'] == over]
        X_over = train_over_data[features]
        y_over = train_over_data['match_result']
        
        test_over_data = test_data_scaled[test_data_scaled['over'] == over]
        y_test = test_over_data['match_result']

        if len(X_over) < 20 or len(y_over.unique()) < 2:
            print("Not enough data or only one class present. Skipping.")
            for model_name in all_models_keys:
                cv_scores[model_name].append(0)
                model_accuracies[model_name].append(0)
                train_accuracies[model_name].append(0)
                calibrated_soft_accuracies[model_name].append(None)
            best_model_accuracies.append(0)
            best_model_soft_accuracies.append(None)
            continue
        
        print(f"Training data size: {len(X_over)} samples")
        print(f"Test data size: {len(test_over_data)} samples")
        print(f"Class distribution: {y_over.value_counts().to_dict()}")
        
        # --- NEURAL NETWORK PROCESSING ---
        print(f"\n{'-'*60}\nPROCESSING NEURAL NETWORK\n{'-'*60}")
        nn_selected_features, nn_method = select_features_for_model(X_over, y_over, 'NeuralNetwork', features)
        feature_selection_log[over]['NeuralNetwork'] = {
            'features': nn_selected_features,
            'method': nn_method
        }
        
        nn_model, nn_cv_results, nn_cv_history = train_neural_network_cv(
            over, train_data_scaled, test_data_scaled, device, nn_selected_features, epochs=50, k_folds=5
        )
        
        if nn_cv_results:
            cv_scores['NeuralNetwork'].append(nn_cv_results['mean_val_acc'])
            train_accuracies['NeuralNetwork'].append(nn_cv_results['mean_train_acc'])
            cv_histories[f'NN_Over_{over}'] = nn_cv_history
        else:
            cv_scores['NeuralNetwork'].append(0)
            train_accuracies['NeuralNetwork'].append(0)
        
        # Save Neural Network model
        if nn_model:
            nn_model_path = os.path.join(models_dir, f"neural_network_over_{over}.pt")
            torch.save({
                'model_state_dict': nn_model.state_dict(),
                'selected_features': nn_selected_features,
                'input_dim': len(nn_selected_features),
                'over': over
            }, nn_model_path)
            all_saved_models[over]['NeuralNetwork'] = nn_model_path
            print(f"Saved Neural Network model to: {nn_model_path}")
        
        # --- ML MODELS PROCESSING ---
        print(f"\n{'-'*60}\nPROCESSING ML MODELS\n{'-'*60}")
        
        fitted_grid_searches = {}
        
        for model_name, grid_search in zip(models.keys(), grid_searches):
            print(f"\n-- Processing: {model_name} --")
            try:
                # Feature selection
                model_selected_features, selection_method = select_features_for_model(
                    X_over, y_over, model_name, features
                )
                feature_selection_log[over][model_name] = {
                    'features': model_selected_features,
                    'method': selection_method
                }
                
                X_over_selected = X_over[model_selected_features]
                
                # Train model with selected features
                print(f"    Training {model_name} with {len(model_selected_features)} selected features...")
                grid_search.fit(X_over_selected, y_over)
                
                cv_scores[model_name].append(grid_search.best_score_)
                fitted_grid_searches[model_name] = grid_search
                
                # Calculate training accuracy
                train_pred = grid_search.best_estimator_.predict(X_over_selected)
                train_acc = accuracy_score(y_over, train_pred)
                train_accuracies[model_name].append(train_acc)
                
                # Calculate test accuracy
                X_test_selected = test_over_data[model_selected_features]
                if not X_test_selected.empty:
                    test_pred = grid_search.best_estimator_.predict(X_test_selected)
                    test_acc = accuracy_score(y_test, test_pred)
                    print(f"    {model_name} Results:")
                    print(f"      CV Score: {grid_search.best_score_:.4f}")
                    print(f"      Train Accuracy: {train_acc:.4f}")
                    print(f"      Test Accuracy: {test_acc:.4f}")
                    print(f"      Best Params: {grid_search.best_params_}")
                else:
                    test_acc = 0
                    print(f"    {model_name} - No test data available")
                
                # Save ML model
                ml_model_path = os.path.join(models_dir, f"{model_name.lower()}_over_{over}.pkl")
                joblib.dump({
                    'model': grid_search.best_estimator_,
                    'selected_features': model_selected_features,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'over': over,
                    'scaler': scaler
                }, ml_model_path)
                all_saved_models[over][model_name] = ml_model_path
                print(f"    Saved {model_name} model to: {ml_model_path}")

            except Exception as e:
                print(f"    Error processing {model_name}: {str(e)}")
                cv_scores[model_name].append(0)
                train_accuracies[model_name].append(0)
                fitted_grid_searches[model_name] = None
        
        # --- FINAL EVALUATION AND BEST MODEL SELECTION ---
        print(f"\n{'-'*60}\nFINAL EVALUATION FOR OVER {over}\n{'-'*60}")
        
        # Calculate final test accuracies and soft accuracies
        if nn_model:
            test_acc, nn_probs, nn_labels = evaluate_neural_network_detailed(over, nn_model, test_data_scaled, device, nn_selected_features)
            model_accuracies['NeuralNetwork'].append(test_acc if test_acc is not None else 0)
            
            # Calculate soft accuracy for Neural Network
            soft_accuracy_nn = None
            if nn_probs is not None and nn_labels is not None and len(nn_labels) > 0:
                nn_probs_2d = np.vstack([1 - nn_probs, nn_probs]).T
                y_test_one_hot = np.eye(2)[nn_labels.astype(int)]
                if nn_probs_2d.shape == y_test_one_hot.shape:
                    mean_deviation = np.mean(np.abs(y_test_one_hot - nn_probs_2d))
                    soft_accuracy_nn = 1 - mean_deviation
            calibrated_soft_accuracies['NeuralNetwork'].append(soft_accuracy_nn)
        else:
            model_accuracies['NeuralNetwork'].append(0)
            calibrated_soft_accuracies['NeuralNetwork'].append(None)

        for model_name in models.keys():
            if fitted_grid_searches.get(model_name):
                model_features = feature_selection_log[over][model_name]['features']
                X_test_selected = test_over_data[model_features]
                if not X_test_selected.empty:
                    y_pred = fitted_grid_searches[model_name].best_estimator_.predict(X_test_selected)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_accuracies[model_name].append(accuracy)
                    
                    # Calculate soft accuracy with calibration
                    soft_accuracy_calibrated = None
                    if hasattr(fitted_grid_searches[model_name].best_estimator_, "predict_proba"):
                        try:
                            calibrator = CalibratedClassifierCV(fitted_grid_searches[model_name].best_estimator_, cv='prefit', method='sigmoid')
                            cal_X = X_over[model_features][:min(50, len(X_over))]
                            cal_y = y_over[:min(50, len(y_over))]
                            calibrator.fit(cal_X, cal_y)
                            y_prob_calibrated = calibrator.predict_proba(X_test_selected)
                            
                            unique_labels = np.unique(y_test)
                            if len(unique_labels) == 2:
                                y_test_one_hot = np.eye(2)[y_test.astype(int)]
                                if y_prob_calibrated.shape[1] == 2:
                                    mean_deviation = np.mean(np.abs(y_test_one_hot - y_prob_calibrated))
                                    soft_accuracy_calibrated = 1 - mean_deviation
                            
                            calibrated_soft_accuracies[model_name].append(soft_accuracy_calibrated)
                        except Exception:
                            calibrated_soft_accuracies[model_name].append(None)
                    else:
                        calibrated_soft_accuracies[model_name].append(None)
                else:
                    model_accuracies[model_name].append(0)
                    calibrated_soft_accuracies[model_name].append(None)
            else:
                model_accuracies[model_name].append(0)
                calibrated_soft_accuracies[model_name].append(None)
        
        # Select best model based on CV score
        best_model_name, best_cv_score = '', -1
        current_cv_scores = {}
        
        for model_name in all_models_keys:
            if len(cv_scores[model_name]) > 0:
                current_score = cv_scores[model_name][-1]
                current_cv_scores[model_name] = current_score
                if current_score > best_cv_score:
                    best_cv_score = current_score
                    best_model_name = model_name
        
        model_usage[over] = best_model_name
        best_model_accuracies.append(model_accuracies[best_model_name][-1])
        best_model_soft_accuracies.append(calibrated_soft_accuracies[best_model_name][-1])
        
        print(f"Current Over {over} Results:")
        for model_name, score in current_cv_scores.items():
            test_acc = model_accuracies[model_name][-1]
            train_acc = train_accuracies[model_name][-1]
            soft_acc = calibrated_soft_accuracies[model_name][-1]
            soft_acc_str = f"{soft_acc:.4f}" if soft_acc is not None else "N/A"
            print(f"  {model_name:<15}: CV={score:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}, Soft={soft_acc_str}")
        
        print(f"\nBest Model Selected: {best_model_name} (CV Score: {best_cv_score:.4f})")
        
        # Copy best model to best_models directory
        if best_model_name in all_saved_models[over]:
            source_path = all_saved_models[over][best_model_name]
            if best_model_name == 'NeuralNetwork':
                best_model_path = os.path.join(best_models_dir, f"best_model_over_{over}.pt")
            else:
                best_model_path = os.path.join(best_models_dir, f"best_model_over_{over}.pkl")
            
            import shutil
            shutil.copy2(source_path, best_model_path)
            print(f"Copied best model to: {best_model_path}")

    # --- COMPREHENSIVE RESULTS ANALYSIS ---
    print(f"\n{'='*100}")
    print("CREATING COMPREHENSIVE RESULTS AND ANALYSIS")
    print(f"{'='*100}")
    
    # Create results DataFrame
    results_data = []
    processed_overs = sorted(model_usage.keys())
    
    for i, over in enumerate(processed_overs):
        row = {
            'Over': over,
            'Best_Model_Selected': model_usage[over],
            'Best_Model_Test_Accuracy': best_model_accuracies[i] if i < len(best_model_accuracies) else None,
            'Best_Model_Calibrated_Soft_Accuracy': best_model_soft_accuracies[i] if i < len(best_model_soft_accuracies) else None
        }
        
        for model_name in all_models_keys:
            row[f'{model_name}_CV_Score'] = cv_scores[model_name][i] if i < len(cv_scores[model_name]) else None
            row[f'{model_name}_Test_Accuracy'] = model_accuracies[model_name][i] if i < len(model_accuracies[model_name]) else None
            row[f'{model_name}_Train_Accuracy'] = train_accuracies[model_name][i] if i < len(train_accuracies[model_name]) else None
            row[f'{model_name}_Calibrated_Soft_Accuracy'] = calibrated_soft_accuracies[model_name][i] if i < len(calibrated_soft_accuracies[model_name]) else None

        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    
    # Save results
    results_path = os.path.join(output_dir, 'comprehensive_results.xlsx')
    results_df.to_excel(results_path, index=False)
    results_df.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), index=False)
    print(f"Comprehensive results saved to: {results_path}")
    
    # Create comprehensive plots
    create_comprehensive_plots(results_df, model_usage, cv_histories, output_dir)
    
    # Save feature selection analysis
    save_feature_selection_analysis(feature_selection_log, output_dir)
    
    # Save model information
    model_info = {
        'all_models_paths': all_saved_models,
        'model_usage_per_over': model_usage,
        'feature_selection_log': feature_selection_log,
        'training_info': {
            'total_overs_processed': len(processed_overs),
            'device_used': str(device),
            'features_used': features,
            'models_trained': list(models.keys()) + ['NeuralNetwork']
        }
    }
    
    with open(os.path.join(output_dir, 'model_information.json'), 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    # --- FINAL SUMMARY ---
    print(f"\n{'='*100}")
    print("FINAL COMPREHENSIVE SUMMARY")
    print(f"{'='*100}")
    
    print(f"Total overs processed: {len(processed_overs)}")
    print(f"Models trained per over: {len(all_models_keys)}")
    print(f"Total models saved: {sum(len(models_dict) for models_dict in all_saved_models.values())}")
    
    # Model performance summary
    print(f"\n{'-'*60}")
    print("AVERAGE MODEL PERFORMANCE SUMMARY")
    print(f"{'-'*60}")
    
    for model_name in all_models_keys:
        cv_avg = np.mean([score for score in cv_scores[model_name] if score > 0])
        test_avg = np.mean([score for score in model_accuracies[model_name] if score > 0])
        train_avg = np.mean([score for score in train_accuracies[model_name] if score > 0])
        soft_scores = [score for score in calibrated_soft_accuracies[model_name] if score is not None]
        soft_avg = np.mean(soft_scores) if soft_scores else 0
        
        print(f"{model_name:<15}: CV={cv_avg:.4f}, Train={train_avg:.4f}, Test={test_avg:.4f}, Soft={soft_avg:.4f}")
    
    # Model selection frequency
    print(f"\n{'-'*60}")
    print("MODEL SELECTION FREQUENCY")
    print(f"{'-'*60}")
    
    selection_counts = Counter(model_usage.values())
    for model, count in selection_counts.most_common():
        percentage = (count / len(model_usage)) * 100
        print(f"{model:<15}: {count:2d} times ({percentage:5.1f}%)")
    
    # Feature selection summary
    print(f"\n{'-'*60}")
    print("FEATURE SELECTION SUMMARY")
    print(f"{'-'*60}")
    
    all_selected_features = []
    feature_method_counts = Counter()
    
    for over_data in feature_selection_log.values():
        for model_data in over_data.values():
            if isinstance(model_data, dict):
                features_list = model_data.get('features', [])
                method = model_data.get('method', 'Unknown')
                all_selected_features.extend(features_list)
                feature_method_counts[method] += 1
            else:
                all_selected_features.extend(model_data)
    
    feature_counts = Counter(all_selected_features)
    total_selections = len(all_selected_features)
    
    print("Most frequently selected features:")
    for feature, count in feature_counts.most_common(10):
        percentage = (count / total_selections) * 100
        print(f"  {feature:<25}: {count:3d} times ({percentage:5.1f}%)")
    
    print(f"\nFeature selection methods used:")
    for method, count in feature_method_counts.most_common():
        print(f"  {method:<30}: {count:3d} times")
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"All results saved in: {output_dir}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()