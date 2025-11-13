import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURATION
# ============================================================
# --- Global matplotlib_config REMOVED as requested ---

# ============================================================
# FILE PATHS
# ============================================================

# Innings 1 paths
IN1_PER_OVER_SELECTED = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\comprehensive_analysis_innings1\comprehensive_results.xlsx"
IN1_PER_OVER_ALL = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\comprehensive_analysis_innings1_all_features\comprehensive_results.xlsx"
IN1_UNIFIED_MEAN = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\per_over_evaluation_of_unified_models_innings1\model_mean_accuracy_summary_innings1.csv"

# Innings 2 paths
IN2_PER_OVER_SELECTED = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\comprehensive_analysis_innings2\comprehensive_results.xlsx"
IN2_PER_OVER_ALL = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\comprehensive_analysis_innings2_all_features\comprehensive_results.xlsx"
IN2_UNIFIED_XLSX = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\plot_for_paper\per_over_evaluation_of_unified_models_in2\per_over_accuracies_of_unified_models.xlsx"

# Output directory
OUTPUT_DIR = "three_strategy_comparison_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS (Unchanged)
# ============================================================

def calculate_mean_accuracy_from_per_over(df, model_columns):
    """Calculate mean test accuracy for each model from per-over data"""
    mean_accuracies = {}
    for col in model_columns:
        if col in df.columns:
            mean_accuracies[col] = df[col].mean()
    return mean_accuracies

def extract_model_names(columns):
    """Extract unique model names from column names"""
    models = set()
    for col in columns:
        if '_Test_Accuracy' in col:
            model_name = col.replace('_Test_Accuracy', '')
            models.add(model_name)
    return sorted(list(models))

# ============================================================
# DATA LOADING FUNCTIONS (Unchanged)
# ============================================================

print("Loading Innings 1 data...")
# (All data loading code remains the same)
in1_selected_df = pd.read_excel(IN1_PER_OVER_SELECTED)
in1_selected_models = [col for col in in1_selected_df.columns if col.endswith('_Test_Accuracy')]
in1_selected_mean = calculate_mean_accuracy_from_per_over(in1_selected_df, in1_selected_models)
in1_all_df = pd.read_excel(IN1_PER_OVER_ALL)
in1_all_models = [col for col in in1_all_df.columns if col.endswith('_Test_Accuracy')]
in1_all_mean = calculate_mean_accuracy_from_per_over(in1_all_df, in1_all_models)
in1_unified_df = pd.read_csv(IN1_UNIFIED_MEAN, index_col=0)
in1_unified_mean = in1_unified_df.to_dict()[in1_unified_df.columns[0]]
all_models_in1 = set()
all_models_in1.update([col.replace('_Test_Accuracy', '') for col in in1_selected_models])
all_models_in1.update([col.replace('_Test_Accuracy', '') for col in in1_all_models])
all_models_in1.update(in1_unified_mean.keys())
if 'Best_Model_Test_Accuracy' in in1_selected_df.columns:
    all_models_in1.add('Best_Model')
all_models_in1 = sorted(list(all_models_in1))
print(f"Found {len(all_models_in1)} models for Innings 1: {all_models_in1}")

print("\nLoading Innings 2 data...")
in2_selected_df = pd.read_excel(IN2_PER_OVER_SELECTED)
in2_selected_models = [col for col in in2_selected_df.columns if col.endswith('_Test_Accuracy')]
in2_selected_mean = calculate_mean_accuracy_from_per_over(in2_selected_df, in2_selected_models)
in2_all_df = pd.read_excel(IN2_PER_OVER_ALL)
in2_all_models = [col for col in in2_all_df.columns if col.endswith('_Test_Accuracy')]
in2_all_mean = calculate_mean_accuracy_from_per_over(in2_all_df, in2_all_models)
in2_unified_df = pd.read_excel(IN2_UNIFIED_XLSX)
in2_unified_models = [col for col in in2_unified_df.columns if col not in ['Over', 'Unnamed: 0']]
in2_unified_mean = calculate_mean_accuracy_from_per_over(in2_unified_df, in2_unified_models)
all_models_in2 = set()
all_models_in2.update([col.replace('_Test_Accuracy', '') for col in in2_selected_models])
all_models_in2.update([col.replace('_Test_Accuracy', '') for col in in2_all_models])
all_models_in2.update(in2_unified_models)
if 'Best_Model_Test_Accuracy' in in2_selected_df.columns:
    all_models_in2.add('Best_Model')
all_models_in2 = sorted(list(all_models_in2))
print(f"Found {len(all_models_in2)} models for Innings 2: {all_models_in2}")


# ============================================================
# MODIFIED PLOTTING FUNCTION (Generates 1 plot)
# ============================================================

def create_comparison_plot(all_models, selected_mean, all_mean, unified_mean, 
                               innings_name, output_dir):
    """
    MODIFIED: Creates a single HORIZONTAL bar plot for one innings
    with MUCH LARGER fonts for all elements.
    """
    
    # --- 1. Prepare Data (Unchanged) ---
    comparison_data = []
    for model in all_models:
        selected_acc = selected_mean.get(f"{model}_Test_Accuracy", 
                                         selected_mean.get(model, np.nan)) * 100
        all_acc = all_mean.get(f"{model}_Test_Accuracy", 
                                all_mean.get(model, np.nan)) * 100
        
        if model == 'Best_Model':
            individual_unified_accs = []
            for m in all_models:
                if m != 'Best_Model':
                    acc = unified_mean.get(m, np.nan)
                    if not np.isnan(acc):
                        individual_unified_accs.append(acc * 100 if acc < 1.1 else acc)
            unified_acc = max(individual_unified_accs) if individual_unified_accs else np.nan
        else:
            unified_acc = unified_mean.get(model, np.nan)
            if not np.isnan(unified_acc):
                unified_acc = unified_acc * 100 if unified_acc < 1.1 else unified_acc
        
        # --- THIS IS THE FIX: Renames "Best_Model" to "Adaptive" ---
        plot_model_name = 'Adaptive' if model == 'Best_Model' else model
        # --- END OF FIX ---
        
        if not np.isnan(selected_acc):
            comparison_data.append({'Model': plot_model_name, 'Training Scenario': 'Per-Over, Selected Features', 'Test Accuracy (%)': selected_acc})
        if not np.isnan(all_acc):
            comparison_data.append({'Model': plot_model_name, 'Training Scenario': 'Per-Over, All Features', 'Test Accuracy (%)': all_acc})
        if not np.isnan(unified_acc):
            comparison_data.append({'Model': plot_model_name, 'Training Scenario': 'Unified Model', 'Test Accuracy (%)': unified_acc})

    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort models by the performance of 'Per-Over, Selected Features'
    sorter = df_comparison[
        df_comparison['Training Scenario'] == 'Per-Over, Selected Features'
    ].groupby('Model')['Test Accuracy (%)'].mean().sort_values(ascending=False).index
    
    df_comparison['Model'] = pd.Categorical(df_comparison['Model'], categories=sorter, ordered=True)
    df_comparison.sort_values('Model', inplace=True)
    
    excel_path = os.path.join(output_dir, f'comparison_data_{innings_name}.xlsx')
    df_comparison.to_excel(excel_path, index=False)
    print(f"Saved comparison data to: {excel_path}")
    
    # --- 2. Create Plot ---
    # (The plotting code from here down is unchanged and correct)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(26, 18)) 

    palette = {
        'Per-Over, Selected Features': '#1f77b4', # Blue
        'Per-Over, All Features': '#ff7f0e',      # Orange
        'Unified Model': '#2ca02c'                # Green
    }
    
    sns.barplot(
        data=df_comparison,
        y='Model',                   
        x='Test Accuracy (%)',       
        hue='Training Scenario',
        palette=palette,
        edgecolor='black',
        linewidth=1.5,
        ax=ax
    )
    
    for p in ax.patches:
        width = p.get_width() 
        if not np.isnan(width):
            ax.annotate(
                f'{width:.2f}', 
                (width, p.get_y() + p.get_height() / 2.), 
                ha='left',   
                va='center', 
                fontsize=33, 
                fontweight='bold',
                xytext=(9, 0), 
                textcoords='offset points'
            )
    
    ax.set_title(f'{innings_name}: Test Accuracy Comparison', 
                 fontsize=43, fontweight='bold', pad=25) 
    
    ax.set_ylabel('Model', 
                 fontsize=39, fontweight='bold', labelpad=20) 
    
    ax.set_xlabel('Test Accuracy (%)', 
                 fontsize=39, fontweight='bold', labelpad=20) 
    
    ax.tick_params(axis='y', labelsize=35) 
    ax.tick_params(axis='x', labelsize=35) 
    
    ax.grid(axis='x', alpha=0.3, linewidth=1.5) 
    
    all_values = df_comparison['Test Accuracy (%)'].dropna()
    if not all_values.empty:
        x_min = all_values.min() - 5
        x_max = all_values.max() + 10 
        ax.set_xlim(x_min, x_max)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=['(Per-Over, Selected Features)', '(Per-Over, All Features)', '(All Overs, All Features)'], 
        title='Training Scenario',
        loc='lower right', 
        fontsize=33,       
        title_fontsize=35, 
        frameon=True, 
        shadow=True
    )
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'comparison_plot_{innings_name}_Horizontal.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.show()
    plt.close()
    
    return df_comparison

# ============================================================
# GENERATE PLOTS (Calls the modified function)
# ============================================================

print("\n" + "="*80)
print("GENERATING INNINGS 1 COMPARISON PLOT")
print("="*80)
df_in1 = create_comparison_plot(all_models_in1, in1_selected_mean, in1_all_mean, 
                                  in1_unified_mean, "Innings 1", OUTPUT_DIR)

print("\n" + "="*80)
print("GENERATING INNINGS 2 COMPARISON PLOT")
print("="*80)
df_in2 = create_comparison_plot(all_models_in2, in2_selected_mean, in2_all_mean, 
                                  in2_unified_mean, "Innings 2", OUTPUT_DIR)

# ============================================================
# SUMMARY STATISTICS (Unchanged)
# ============================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

def print_strategy_summary(df_long, innings_name):
    print(f"\n{innings_name}:")
    strategies = ['Per-Over, Selected Features', 'Per-Over, All Features', 'Unified Model']
    # Group by strategy to get stats
    for strategy in strategies:
        strategy_df = df_long[df_long['Training Scenario'] == strategy]
        if not strategy_df.empty:
            mean_acc = strategy_df['Test Accuracy (%)'].mean()
            std_acc = strategy_df['Test Accuracy (%)'].std()
            max_acc = strategy_df['Test Accuracy (%)'].max()
            max_model = strategy_df.loc[strategy_df['Test Accuracy (%)'].idxmax(), 'Model']
            min_acc = strategy_df['Test Accuracy (%)'].min()
            min_model = strategy_df.loc[strategy_df['Test Accuracy (%)'].idxmin(), 'Model']
            
            print(f"  {strategy}:")
            print(f"    Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
            print(f"    Best: {max_acc:.2f}% ({max_model})")
            print(f"    Worst: {min_acc:.2f}% ({min_model})")
            print(f"    Range: {max_acc - min_acc:.2f}%")

# We use the returned dataframes (which are now in long format)
print_strategy_summary(df_in1, "Innings 1")
print_strategy_summary(df_in2, "Innings 2")

# ============================================================
# DETAILED STD DEVIATION ANALYSIS (Unchanged)
# ============================================================

print("\n" + "="*80)
print("DETAILED STANDARD DEVIATION ANALYSIS")
print("="*80)

def calculate_per_over_std(df, model_columns, strategy_name):
    """Calculate standard deviation across overs for each model"""
    print(f"\n{strategy_name}:")
    print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev (Across Overs)':<25}")
    print("-" * 60)
    
    std_results = {}
    for col in model_columns:
        if col in df.columns:
            mean_val = df[col].mean() * 100
            std_val = df[col].std() * 100
            model_name = col.replace('_Test_Accuracy', '') if '_Test_Accuracy' in col else col
            std_results[model_name] = {'mean': mean_val, 'std': std_val}
            print(f"{model_name:<20} {mean_val:>6.2f}% {std_val:>12.2f}%")
    
    return std_results

print("\n" + "="*50)
print("INNINGS 1 - Per-Over Variation")
print("="*50)
print("\n--- Strategy 1: Per-Over, Selected Features ---")
in1_selected_std = calculate_per_over_std(in1_selected_df, in1_selected_models, "Innings 1 - Selected Features")
print("\n--- Strategy 2: Per-Over, All Features ---")
in1_all_std = calculate_per_over_std(in1_all_df, in1_all_models, "Innings 1 - All Features")
print("\n--- Strategy 3: Unified Model (All Over, All Features) ---")
print("Note: Calculating variation from per-over predictions of unified models")
if 'NeuralNetwork' in in1_unified_df.index:
    print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev (Across Overs)':<25}")
    print("-" * 60)
    for model in in1_unified_df.index:
        mean_val = in1_unified_df.loc[model].values[0]
        mean_val_pct = mean_val * 100 if mean_val < 1.1 else mean_val
        print(f"{model:<20} {mean_val_pct:>6.2f}% {'N/A (mean only)':>20}")

print("\n" + "="*50)
print("INNINGS 2 - Per-Over Variation")
print("="*50)
print("\n--- Strategy 1: Per-Over, Selected Features ---")
in2_selected_std = calculate_per_over_std(in2_selected_df, in2_selected_models, "Innings 2 - Selected Features")
print("\n--- Strategy 2: Per-Over, All Features ---")
in2_all_std = calculate_per_over_std(in2_all_df, in2_all_models, "Innings 2 - All Features")
print("\n--- Strategy 3: Unified Model (All Over, All Features) ---")
in2_unified_std = calculate_per_over_std(in2_unified_df, in2_unified_models, "Innings 2 - Unified Model")

# ============================================================
# STABILITY COMPARISON (Unchanged)
# ============================================================

print("\n" + "="*80)
print("MODEL STABILITY COMPARISON (Lower Std Dev = More Stable)")
print("="*80)

def compare_stability(std_dict1, std_dict2, std_dict3, innings_name):
    print(f"\n{innings_name}:")
    print(f"{'Model':<20} {'Selected Std':<15} {'All Features Std':<20} {'Unified Std':<15}")
    print("-" * 70)
    
    all_models = set()
    all_models.update(std_dict1.keys())
    all_models.update(std_dict2.keys())
    if std_dict3:
        all_models.update(std_dict3.keys())
    
    for model in sorted(all_models):
        std1 = std_dict1.get(model, {}).get('std', np.nan)
        std2 = std_dict2.get(model, {}).get('std', np.nan)
        std3 = std_dict3.get(model, {}).get('std', np.nan) if std_dict3 else np.nan
        
        std1_str = f"{std1:.2f}%" if not np.isnan(std1) else "N/A"
        std2_str = f"{std2:.2f}%" if not np.isnan(std2) else "N/A"
        std3_str = f"{std3:.2f}%" if not np.isnan(std3) else "N/A"
        
        print(f"{model:<20} {std1_str:<15} {std2_str:<20} {std3_str:<15}")

# Note: For Innings 1 unified, we don't have per-over data, so we can't calculate std
compare_stability(in1_selected_std, in1_all_std, None, "Innings 1")
compare_stability(in2_selected_std, in2_all_std, in2_unified_std, "Innings 2")

print("\n" + "="*80)
print("✅ ALL COMPARISON PLOTS AND STATISTICS GENERATED SUCCESSFULLY!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)