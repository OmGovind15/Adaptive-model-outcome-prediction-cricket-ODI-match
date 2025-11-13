import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np # Import numpy

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL MATPLOTLIB CONFIG (Optional, as sizes are set below)
# ============================================================
# You can add the large rcParams block here if you want
# matplotlib.rcParams.update({ ... })


def load_data_from_folder(folder_path):
    """
    Loads all CSV files from a specified folder into a single pandas DataFrame.
    """
    # Check if the folder exists to avoid errors
    if not os.path.isdir(folder_path):
        print(f"⚠️  Warning: Directory not found at '{folder_path}'. Skipping.")
        return pd.DataFrame()

    all_data = []
    # Loop through each file in the directory
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                match_data = pd.read_csv(file_path)
                all_data.append(match_data)
            except Exception as e:
                print(f"Could not read file {file}: {e}")

    if not all_data:
        print(f"🤷 No CSV files were found or loaded from '{folder_path}'.")
        return pd.DataFrame()

    # Combine all individual DataFrames into one
    return pd.concat(all_data, ignore_index=True)

def plot_combined_distribution(data_counts_1, data_counts_2, title, output_filename):
    """
    ### UPDATED FUNCTION ###
    Plots the data distribution with larger fonts and a customized x-axis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 11))

    max_x = 0 # To store the maximum over number

    # Plot for Innings 1 if its data is available
    if not data_counts_1.empty:
        # --- MODIFIED: Add +1 to x-axis data ---
        plot_x_1 = data_counts_1.index + 1
        ax.plot(plot_x_1, data_counts_1.values,
                marker='o', linestyle='-', color='dodgerblue', 
                label='Innings 1', linewidth=3.5, markersize=9)
        max_x = max(max_x, plot_x_1.max())

    if not data_counts_2.empty:
        # --- MODIFIED: Add +1 to x-axis data ---
        plot_x_2 = data_counts_2.index + 1
        ax.plot(plot_x_2, data_counts_2.values,
                marker='s', linestyle='--', color='orangered', 
                label='Innings 2', linewidth=3.5, markersize=9)
        max_x = max(max_x, plot_x_2.max())

    # ✅ EVEN LARGER FONTS
    ax.set_title(title, fontsize=48, fontweight='bold', pad=20)
    ax.set_xlabel('Over Number', fontsize=48, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Data Points (Rows)', fontsize=48, fontweight='bold', labelpad=10)
    
    # ✅ LEGEND location changed to 'best'
    ax.legend(title='Innings', fontsize=48, title_fontsize=48, 
              frameon=True, shadow=True, loc='best')
              
    ax.tick_params(axis='both', which='major', labelsize=48, width=2, length=8)
    
    # --- MODIFIED: Set X-axis limits and ticks ---
    ax.set_xlim(left=0.5) # Add padding to the left of '1'
    if max_x > 0:
        # Create a list like [1, 10, 20, 30, 40, 50]
        ticks = [1] + list(range(10, int(max_x), 10)) + [int(max_x)]
        ax.set_xticks(ticks)
    
    plt.xticks(rotation=0) # No rotation needed
    # ---------------------------------------------
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Plot successfully saved as '{output_filename}'")
    plt.show()

def main():
    """
    Main function to load data and generate the combined plots.
    """
    # --- Define folder paths from your scripts ---
    # Innings 1 data paths
    train_folder_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\training_data_innings1_encoded"
    test_folder_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\encoded_test_inning1"
    # Innings 2 data paths
    train_folder_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
    test_folder_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\encoded_test_inning2"

    # Create an output directory for the plots
    output_dir = "data_distribution_plots"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load all datasets ---
    print("\n--- Loading Data ---")
    train_data_1 = load_data_from_folder(train_folder_innings1)
    test_data_1 = load_data_from_folder(test_folder_innings1)
    train_data_2 = load_data_from_folder(train_folder_innings2)
    test_data_2 = load_data_from_folder(test_folder_innings2)

    # --- Generate Combined Training Data Plot ---
    print("\n--- Processing Training Data ---")
    if not train_data_1.empty or not train_data_2.empty:
        # --- MODIFIED: Filter Innings 1 (0-49) and Innings 2 (0-48) ---
        train_counts_1 = train_data_1[train_data_1['over'] <= 49]['over'].value_counts().sort_index()
        train_counts_2 = train_data_2[train_data_2['over'] <= 48]['over'].value_counts().sort_index()
        plot_combined_distribution(train_counts_1, train_counts_2,
                                   'Distribution of Training Data Points per Over',
                                   os.path.join(output_dir, 'combined_train_data_dist.pdf'))

    # --- Generate Combined Test Data Plot ---
    print("\n--- Processing Test Data ---")
    if not test_data_1.empty or not test_data_2.empty:
        # --- MODIFIED: Filter Innings 1 (0-49) and Innings 2 (0-48) ---
        test_counts_1 = test_data_1[test_data_1['over'] <= 49]['over'].value_counts().sort_index()
        test_counts_2 = test_data_2[test_data_2['over'] <= 48]['over'].value_counts().sort_index()
        plot_combined_distribution(test_counts_1, test_counts_2,
                                   'Distribution of Test Data Points per Over',
                                   os.path.join(output_dir, 'combined_test_data_dist.pdf'))

if __name__ == "__main__":
    main()
