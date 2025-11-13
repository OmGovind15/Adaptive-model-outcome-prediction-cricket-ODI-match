import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING FUNCTION (Unchanged)
# ============================================================
def load_unique_match_results(folder_path):
    """
    Loads the unique match result from each CSV file in a folder.
    """
    if not os.path.isdir(folder_path):
        print(f"⚠️  Warning: Directory not found at '{folder_path}'. Skipping.")
        return pd.DataFrame()

    unique_matches = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                # Read only the first row to get the match result
                match_data = pd.read_csv(file_path, nrows=1)
                if 'match_result' in match_data.columns:
                    unique_matches.append(match_data[['match_result']])
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty file: {file}")
            except Exception as e:
                print(f"Could not read file {file}: {e}")
    
    if not unique_matches:
        print(f"🤷 No valid data found in '{folder_path}'.")
        return pd.DataFrame()

    return pd.concat(unique_matches, ignore_index=True)

# ============================================================
# NEW PLOTTING FUNCTION
# ============================================================
def plot_win_loss_comparison(train_counts, test_counts, innings_num, output_dir, font_settings):
    """
    Generates a single figure with two subplots (train and test)
    to ensure consistent sizing and a single legend.
    """
    print(f"\n--- Analyzing Innings {innings_num} ---")
    fs = font_settings # Shortcut
    
    # --- Prepare Data ---
    labels = ['Bowling Team Wins', 'Bowling Team Loses']
    colors = ['#a6cee3', '#fdbf6f']  # Light Blue, Light Orange
    
    # Training Data
    train_sizes = [train_counts.get('Win', 0), train_counts.get('Loss', 0)]
    train_total = sum(train_sizes)
    
    # Test Data
    test_sizes = [test_counts.get('Win', 0), test_counts.get('Loss', 0)]
    test_total = sum(test_sizes)
    
    if train_total == 0 or test_total == 0:
        print(f"Skipping Innings {innings_num}: Not enough data.")
        return

    # --- Create Figure with 2 Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14)) # Made figure a bit taller
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- (a) Training Data Plot ---
    patches, texts, autotexts = ax1.pie(
        train_sizes, 
        explode=(0, 0), 
        labels=None, 
        colors=colors,
        autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * train_total / 100),
        shadow=False, 
        startangle=140, 
        textprops={'fontsize': fs['annot']}
    )
    ax1.set_title(f"Bowling Team Win/Loss Distribution\n(Innings {innings_num} - Training Data)", 
                  fontsize=fs['title'], pad=20)
    ax1.axis('equal')
    
    ax1.text(0.5, -0.1, '(a) Training data', 
             size=fs['subcap'], ha='center', 
             transform=ax1.transAxes)

    # --- (b) Test Data Plot ---
    ax2.pie(
        test_sizes, 
        explode=(0, 0), 
        labels=None, 
        colors=colors,
        autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * test_total / 100),
        shadow=False, 
        startangle=140, 
        textprops={'fontsize': fs['annot']}
    )
    ax2.set_title(f"Bowling Team Win/Loss Distribution\n(Innings {innings_num} - Test Data)", 
                  fontsize=fs['title'], pad=20)
    ax2.axis('equal')
    
    ax2.text(0.5, -0.1, '(b) Test data', 
             size=fs['subcap'], ha='center', 
             transform=ax2.transAxes)

    # --- MODIFIED: Add ONE Legend centered at the BOTTOM ---
    fig.legend(patches, labels,
               title="",
               loc="lower center", # Position at the bottom
               bbox_to_anchor=(0.5, 0.05), # Center it horizontally (0.5) and place near bottom (0.05)
               ncol=3, # Arrange in 3 columns: Title, Wins, Loses
               fontsize=fs['legend'],
               title_fontsize=fs['legend_title'])
    
    # Adjust subplot layout to make room for the legend
    fig.subplots_adjust(bottom=0.2, top=0.9)

    # Save the combined figure
    filename = f'innings_{innings_num}_balance_comparison.pdf'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as '{filename}'")
    plt.show()


# ============================================================
# MAIN FUNCTION (MODIFIED)
# ============================================================
def main():
    """
    Main function to analyze train/test sets and generate plots.
    """
    # Define folder paths
    train_folder_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\training_data_innings1_encoded"
    test_folder_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\encoded_test_inning1"
    
    # --- CORRECTED PATHS FOR INNINGS 2 ---
    train_folder_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
    test_folder_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\encoded_test_inning2"

    output_dir = "balance_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)

    # ##################################################################
    # ### FONT CONTROL PANEL ###
    # ##################################################################
    
    fonts_in1 = {
        'title': 42,
        'annot': 36,
        'legend': 30,
        'legend_title': 32,
        'subcap': 40
    }
    
    fonts_in2 = {
        'title': 42,
        'annot': 36,
        'legend': 30,
        'legend_title': 32,
        'subcap': 40
    }
    
    # --- Load all data first ---
    print("Loading data...")
    train_data_1 = load_unique_match_results(train_folder_innings1)
    test_data_1 = load_unique_match_results(test_folder_innings1)
    train_data_2 = load_unique_match_results(train_folder_innings2)
    test_data_2 = load_unique_match_results(test_folder_innings2)

    # --- Get counts for Innings 1 ---
    if not train_data_1.empty and not test_data_1.empty:
        train_data_1['outcome'] = train_data_1['match_result'].apply(lambda x: 'Win' if x == 1 else 'Loss')
        train_counts_1 = train_data_1['outcome'].value_counts()
        
        test_data_1['outcome'] = test_data_1['match_result'].apply(lambda x: 'Win' if x == 1 else 'Loss')
        test_counts_1 = test_data_1['outcome'].value_counts()
        
        # Plot Innings 1 (Train + Test)
        plot_win_loss_comparison(train_counts_1, test_counts_1, 1, output_dir, fonts_in1)
    
    # --- Get counts for Innings 2 ---
    if not train_data_2.empty and not test_data_2.empty:
        train_data_2['outcome'] = train_data_2['match_result'].apply(lambda x: 'Win' if x == 1 else 'Loss')
        train_counts_2 = train_data_2['outcome'].value_counts()
        
        test_data_2['outcome'] = test_data_2['match_result'].apply(lambda x: 'Win' if x == 1 else 'Loss')
        test_counts_2 = test_data_2['outcome'].value_counts()
        
        # Plot Innings 2 (Train + Test)
        plot_win_loss_comparison(train_counts_2, test_counts_2, 2, output_dir, fonts_in2)

if __name__ == "__main__":
    main()