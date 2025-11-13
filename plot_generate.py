import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL MATPLOTLIB CONFIG (Fallback)
# ============================================================
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'figure.figsize': (18, 10),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
})

# ============================================================
# DATA LOADING FUNCTIONS (Unchanged)
# ============================================================

def load_data_from_folder(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            try:
                match_data = pd.read_csv(os.path.join(folder_path, file))
                match_number = file.split('.')[0]
                match_data['match_number'] = match_number
                all_data.append(match_data)
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty file: {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def feature_engineering_in1(data):
    # 1st innings specific feature engineering
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

def feature_engineering_in2(data):
    # 2nd innings specific feature engineering
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['required_run_rate'] = data['trail_runs'] / (49 - data['over']).clip(lower=1)
    data['rate_gap'] = data['required_run_rate'] - data['run_rate']
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = (300 * data['rate_gap']) / ((data['wickets_remaining'] + 1) * (data['momentum_factor'] + 1) * (data['balls_remaining'] + 1)).clip(lower=1)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = ['trail_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'rate_gap', 'pressure_index', 'momentum_factor', 'weighted_batting_average',
                'weighted_bowling_average','venue_index', 'partnership']
    return data, features

# ============================================================
# MODIFIED PLOTTING FUNCTIONS
# ============================================================

def create_correlation_matrix_plot(df, features, output_dir, font_settings, filename):
    """
    Plots the feature correlation matrix using custom font sizes and filename.
    """
    print(f"... regenerating {filename}")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(18, 16))
    fs = font_settings 
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", 
                linewidths=1, vmin=-1, vmax=1, ax=ax,
                annot_kws={'fontweight': 'bold', 'fontsize': fs['annot']},
                cbar_kws={'shrink': 0.8})
    
    ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=25, fontsize=fs['title'])
    ax.tick_params(axis='x', rotation=45, labelsize=fs['ticks'])
    ax.tick_params(axis='y', rotation=0, labelsize=fs['ticks'])
    plt.xticks(ha='right')
    plt.yticks(rotation=0)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(width=2, length=8, labelsize=fs['cbar_ticks'])
    cbar.set_label('Correlation Coefficient', fontweight='bold', labelpad=15, fontsize=fs['cbar_label'])
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Feature correlation matrix saved to: {save_path}")

def create_comprehensive_plots(results_df, model_usage, cv_histories, output_dir, font_settings_dict, filenames_dict):
    """
    Create comprehensive plots using a dictionary of custom font sizes and filenames.
    """
    plt.style.use('default')
    sns.set_palette("husl")
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    all_models = [col.replace('_Test_Accuracy', '') for col in results_df.columns if col.endswith('_Test_Accuracy')]
    
    # ============================================================
    # 1. TEST ACCURACIES PLOT
    # ============================================================
    filename = filenames_dict['test_acc']
    print(f"... regenerating {filename}")
    fs = font_settings_dict['test_acc']
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # This adds +1 to your 'Over' data
    plot_x_axis = results_df['Over'] + 1
    
    for model in all_models:
        if f'{model}_Test_Accuracy' in results_df.columns:
            ax.plot(plot_x_axis, results_df[f'{model}_Test_Accuracy'], marker='o', label=model, linewidth=3.5, markersize=9)
    if 'Best_Model_Test_Accuracy' in results_df.columns:
        ax.plot(plot_x_axis, results_df['Best_Model_Test_Accuracy'], label='Best Model', color='black', linestyle='--', linewidth=4.5, marker='s', markersize=11, markeredgewidth=2.5)
    
    ax.set_ylim(0.6, 1.0)
    
    # --- THIS IS THE NEW FIX ---
    # 1. Set the axis limit to just before 1
    max_x = plot_x_axis.max() 
    ax.set_xlim(left=-1, right=max_x + 1)
    
    # 2. Manually define the ticks to show
    max_x = plot_x_axis.max()
    # Create a list like [1, 10, 20, 30, 40, 50] or [1, 10, 20, 30, 40, 49]
    ticks = [1] + list(range(10, int(max_x), 10)) + [int(max_x)]
    ax.set_xticks(ticks)
    # ---------------------------
    
    ax.set_title('Test Accuracies Across Overs', fontweight='bold', pad=25, fontsize=fs['title'])
    ax.set_xlabel('Over', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.set_ylabel('Test Accuracy', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.legend(frameon=True, shadow=True, loc='best', ncol=2, columnspacing=1.5, framealpha=0.95, fontsize=fs['legend'])
    ax.tick_params(axis='both', which='major', width=2.5, length=10, labelsize=fs['ticks'])
    ax.grid(True, alpha=0.4, linewidth=1.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 2. VALIDATION ACCURACIES (CV SCORES)
    # ============================================================
    filename = filenames_dict['val_acc']
    print(f"... regenerating {filename}")
    fs = font_settings_dict['val_acc']
    fig, ax = plt.subplots(figsize=(18, 10))
    
    for model in all_models:
        if f'{model}_CV_Score' in results_df.columns:
            ax.plot(results_df['Over'], results_df[f'{model}_CV_Score'], marker='o', label=model, linewidth=3.5, markersize=9)
    
    ax.set_title('Cross-Validation Accuracies Across Overs', fontweight='bold', pad=25, fontsize=fs['title'])
    ax.set_xlabel('Over', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.set_ylabel('CV Accuracy', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.legend(frameon=True, shadow=True, loc='best', ncol=2, fontsize=fs['legend'])
    ax.tick_params(axis='both', which='major', width=2.5, length=10, labelsize=fs['ticks'])
    ax.grid(True, alpha=0.4, linewidth=1.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 3. NEURAL NETWORK TRAIN/VAL/TEST
    # ============================================================
    filename = filenames_dict['nn_plot']
    print(f"... regenerating {filename}")
    fs = font_settings_dict['nn_plot']
    if 'NeuralNetwork_Train_Accuracy' in results_df.columns:
        fig, ax = plt.subplots(figsize=(18, 10))
        
        ax.plot(results_df['Over'], results_df['NeuralNetwork_Train_Accuracy'], marker='o', label='Training', linewidth=3.5, markersize=9, color='red')
        ax.plot(results_df['Over'], results_df['NeuralNetwork_CV_Score'], marker='s', label='Validation', linewidth=3.5, markersize=9, color='blue')
        ax.plot(results_df['Over'], results_df['NeuralNetwork_Test_Accuracy'], marker='^', label='Test', linewidth=3.5, markersize=9, color='green')
        
        ax.set_title('Neural Network: Training vs Validation vs Test Accuracy', fontweight='bold', pad=25, fontsize=fs['title'])
        ax.set_xlabel('Over', fontweight='bold', labelpad=15, fontsize=fs['labels'])
        ax.set_ylabel('Accuracy', fontweight='bold', labelpad=15, fontsize=fs['labels'])
        ax.legend(frameon=True, shadow=True, loc='best', fontsize=fs['legend'])
        ax.tick_params(axis='both', which='major', width=2.5, length=10, labelsize=fs['ticks'])
        ax.grid(True, alpha=0.4, linewidth=1.5)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # ============================================================
    # 4. MODEL SELECTION FREQUENCY - HORIZONTAL
    # ============================================================
    filename = filenames_dict['model_freq']
    print(f"... regenerating {filename}")
    fs = font_settings_dict['model_freq']
    model_counts = pd.Series(model_usage.values()).value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    
    bars = ax.barh(range(len(model_counts)), model_counts.values, color=plt.cm.Set3(range(len(model_counts))),
                   edgecolor='black', linewidth=2.5, height=0.7)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center', 
                fontweight='bold', fontsize=fs['annot'])
    
    ax.set_title('Best Model Selection Frequency', fontweight='bold', pad=25, fontsize=fs['title'])
    ax.set_ylabel('Model', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.set_xlabel('Number of Times Selected', fontweight='bold', labelpad=15, fontsize=fs['labels'])
    ax.set_yticks(range(len(model_counts)))
    ax.set_yticklabels(model_counts.index, rotation=0, fontsize=fs['ticks'])
    ax.tick_params(axis='x', width=2.5, length=10, labelsize=fs['ticks'])
    ax.margins(y=0.01)
    
    if not model_counts.empty:
        ax.set_xlim(right=model_counts.values.max() * 1.15)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 5. ACCURACY DISTRIBUTION BOX PLOT
    # ============================================================
    filename = filenames_dict['box_plot']
    print(f"... regenerating {filename}")
    fs = font_settings_dict['box_plot']
    accuracy_data = []
    model_names = []
    
    for model in all_models:
        if f'{model}_Test_Accuracy' in results_df.columns:
            accuracies = results_df[f'{model}_Test_Accuracy'].dropna()
            accuracy_data.extend(accuracies.tolist())
            model_names.extend([model] * len(accuracies))
    
    if accuracy_data:
        fig, ax = plt.subplots(figsize=(17, 10))
        df_box = pd.DataFrame({'Model': model_names, 'Test_Accuracy': accuracy_data})
        
        bp = sns.boxplot(data=df_box, x='Model', y='Test_Accuracy', ax=ax, width=0.6, linewidth=2.5)
        
        for patch in bp.patches:
            patch.set_linewidth(2.5)
        
        ax.set_title('Test Accuracy Distribution by Model', fontweight='bold', pad=25, fontsize=fs['title'])
        ax.set_xlabel('Model', fontweight='bold', labelpad=15, fontsize=fs['labels'])
        ax.set_ylabel('Test Accuracy', fontweight='bold', labelpad=15, fontsize=fs['labels'])
        ax.tick_params(axis='x', rotation=45, labelsize=fs['ticks_x'])
        ax.tick_params(axis='y', width=2.5, length=10, labelsize=fs['ticks_y'])
        plt.xticks(rotation=45, ha='right')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    print(f"✅ All comprehensive plots saved to: {plots_dir}")


def save_feature_selection_analysis(feature_selection_log, output_dir, font_settings, filename):
    """
    Plots feature selection frequency using custom font sizes and filename.
    """
    fs_dir = os.path.join(output_dir, 'feature_selection_analysis')
    os.makedirs(fs_dir, exist_ok=True)
    
    # ============================================================
    # 4. FEATURE SELECTION FREQUENCY PLOT - HORIZONTAL
    # ============================================================
    print(f"... regenerating {filename}")
    fs = font_settings
    
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
        fig, ax = plt.subplots(figsize=(18, 11))
        features_sorted = sorted(feature_counts.items(), key=lambda x: x[1], reverse=False)
        features, counts = zip(*features_sorted)
        
        bars = ax.barh(range(len(features)), counts, 
                      color=plt.cm.viridis(np.linspace(0.3, 0.9, len(features))),
                      edgecolor='black', linewidth=2.5, height=0.75)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 3, 
                    bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', 
                    fontweight='bold', fontsize=fs['annot'])
        
        ax.set_title('Feature Selection Frequency Across All Models and Overs', 
                     fontweight='bold', pad=25, fontsize=fs['title'])
        ax.set_ylabel('Features',  fontweight='bold', labelpad=15, fontsize=fs['labels'])
        ax.set_xlabel('Selection Count', fontweight='bold', labelpad=15, fontsize=fs['labels'])
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, rotation=0, fontsize=fs['ticks'])
        ax.tick_params(axis='x', width=2.5, length=10, labelsize=fs['ticks'])
        
        if counts:
             ax.set_xlim(right=max(counts) * 1.15)
        ax.margins(y=0.01)

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    print(f"✅ Feature selection analysis saved to: {fs_dir}")

# ============================================================
# NEW MAIN FUNCTION TO LOAD RESULTS AND RE-PLOT
# ============================================================
def generate_plots_for_innings(innings_num):
    print(f"\n{'='*80}")
    print(f"GENERATING PLOTS FOR INNINGS {innings_num}")
    print(f"{'='*80}")
    
    output_dir = f"comprehensive_analysis_innings{innings_num}"
    results_file = os.path.join(output_dir, 'comprehensive_results.xlsx')
    info_file = os.path.join(output_dir, 'model_information.json')

    if not os.path.exists(results_file) or not os.path.exists(info_file):
        print(f"Error: Cannot find result files in '{output_dir}'.")
        print(f"Please run 'final_in{innings_num}.py' at least once to generate these files.")
        return

    # ##################################################################
    # ### FONT, FILENAME, AND FILTER CONTROL PANEL ###
    # ##################################################################
    
    corr_matrix_fonts = {}
    comprehensive_plot_fonts = {}
    feature_selection_fonts = {}
    corr_matrix_filename = ""
    comprehensive_filenames = {}
    feature_selection_filename = ""
    max_over = 49 # Default

    if innings_num == 1:
        print("--- Using INNINGS 1 Settings ---")
        
        ### <-- MODIFIED: Set Max Over ###
        max_over = 49
        
        # === FILENAMES for Innings 1 ===
        corr_matrix_filename = "corr_matrix_1.pdf"
        comprehensive_filenames = {
            'test_acc':   "test_accuracies_in1.pdf",
            'val_acc':    "validation_accuracies_in1.pdf",
            'nn_plot':    "nn_plot_in1.pdf",
            'model_freq': "model_selection_frequency_1.pdf",
            'box_plot':   "box_plot_in1.pdf"
        }
        feature_selection_filename = "feature_selection_frequency_in1.pdf"
        
        # === FONTS for Innings 1 ===
        corr_matrix_fonts = {
            'title': 30, 'annot': 21, 'labels': 30,
            'ticks': 25, 'cbar_label': 22, 'cbar_ticks': 20
        }
        comprehensive_plot_fonts = {
            'test_acc':   {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'val_acc':    {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'nn_plot':    {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'model_freq': {'title': 28, 'labels': 28, 'ticks': 25, 'annot': 28},
            'box_plot':   {'title': 28, 'labels': 28, 'ticks_x': 25, 'ticks_y': 28}
        }
        feature_selection_fonts = {
            'title': 28, 'labels': 25, 'ticks': 25, 'annot': 21
        }
        
    else: # Innings 2
        print("--- Using INNINGS 2 Settings ---")
        
        ### <-- MODIFIED: Set Max Over ###
        max_over = 48
        
        # === FILENAMES for Innings 2 ===
        corr_matrix_filename = "corr_matrix_2.pdf"
        comprehensive_filenames = {
            'test_acc':   "test_accuracies_2.pdf", # As you requested
            'val_acc':    "validation_accuracies_in2.pdf",
            'nn_plot':    "nn_plot_in2.pdf",
            'model_freq': "model_selection_frequency_in2.pdf",
            'box_plot':   "box_plot_in2.pdf"
        }
        feature_selection_filename = "feature_selection_frequency_in2.pdf"

        # === FONTS for Innings 2 ===
        corr_matrix_fonts = {
            'title': 30, 'annot': 21, 'labels': 30,
            'ticks': 25, 'cbar_label': 22, 'cbar_ticks': 20
        }
        comprehensive_plot_fonts = {
            'test_acc':   {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'val_acc':    {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'nn_plot':    {'title': 28, 'labels': 28, 'ticks': 25, 'legend': 28},
            'model_freq': {'title': 28, 'labels': 28, 'ticks': 25, 'annot': 28},
            'box_plot':   {'title': 28, 'labels': 28, 'ticks_x': 25, 'ticks_y': 28}
        }
        feature_selection_fonts = {
            'title': 28, 'labels': 25, 'ticks': 25, 'annot': 21
        }

    # --- 1. Generate Correlation Matrix ---
    print(f"Loading raw data for Innings {innings_num} correlation matrix...")
    try:
        if innings_num == 1:
            train_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\training_data_innings1_encoded"
            train_data = load_data_from_folder(train_folder)
            train_data, features = feature_engineering_in1(train_data)
        else:
            train_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
            train_data = load_data_from_folder(train_folder)
            train_data, features = feature_engineering_in2(train_data)
        
        if train_data.empty:
            print("No data loaded, skipping correlation matrix.")
        else:
            ### <-- MODIFIED: Filter data based on max_over ###
            print(f"Filtering raw data for overs <= {max_over}")
            train_data_filtered = train_data[train_data['over'] <= max_over].copy()
            create_correlation_matrix_plot(train_data_filtered, features, output_dir, corr_matrix_fonts, corr_matrix_filename)
        
    except Exception as e:
        print(f"Error generating correlation matrix for Innings {innings_num}: {e}")
        print("Please check raw data paths.")

    # --- 2. Generate All Other Plots ---
    print(f"Loading results data from '{output_dir}'...")
    results_df = pd.read_excel(results_file)
    with open(info_file, 'r') as f:
        model_info = json.load(f)

    ### <-- MODIFIED: Filter all results data based on max_over ###
    print(f"Filtering results data for Overs <= {max_over}")
    results_df = results_df[results_df['Over'] <= max_over].copy()
    model_usage = {int(k): v for k, v in model_info.get('model_usage_per_over', {}).items() if int(k) <= max_over}
    feature_selection_log = {k: v for k, v in model_info.get('feature_selection_log', {}).items() if int(k) <= max_over}

    if not model_usage or not feature_selection_log or results_df.empty:
         print("Error: Filtered data is empty or JSON file is missing key information.")
         return

    print("\nCalling create_comprehensive_plots...")
    create_comprehensive_plots(results_df, model_usage, {}, output_dir, comprehensive_plot_fonts, comprehensive_filenames)
    
    print("\nCalling save_feature_selection_analysis...")
    save_feature_selection_analysis(feature_selection_log, output_dir, feature_selection_fonts, feature_selection_filename)
    
    print(f"\n✅ All plots for Innings {innings_num} have been re-generated in '{output_dir}'")

if __name__ == "__main__":
    generate_plots_for_innings(1)
    generate_plots_for_innings(2)