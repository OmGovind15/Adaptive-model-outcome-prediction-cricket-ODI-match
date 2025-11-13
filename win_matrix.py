
import matplotlib
import matplotlib.pyplot as plt

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data_from_folder(folder_path):
    """
    Loads all CSV files from a folder and adds a 'match_id' column
    extracted from the filename.
    """
    if not os.path.isdir(folder_path):
        print(f"⚠️ Warning: Directory not found at '{folder_path}'. Skipping.")
        return pd.DataFrame()

    all_data = []
    # Regex to find numbers in the filename (the match ID)
    id_extractor = re.compile(r'\d+')

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            match_id_search = id_extractor.search(file)
            if not match_id_search:
                continue # Skip files without a number in their name

            match_id = match_id_search.group()
            file_path = os.path.join(folder_path, file)
            try:
                match_data = pd.read_csv(file_path, usecols=['batting_team', 'bowling_team', 'match_result'])
                # Add the extracted match_id to every row of the data from this file
                match_data['match_id'] = match_id
                all_data.append(match_data)
            except Exception as e:
                print(f"Could not read or process file {file}: {e}")

    if not all_data:
        print(f"🤷 No valid CSV files were loaded from '{folder_path}'.")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

def create_head_to_head_matrix(matches_df):
    """
    Creates a head-to-head win matrix using the 'match_id' to identify unique matches.
    """
    if matches_df.empty:
        print("❌ Error: The DataFrame is empty. Cannot create matrix.")
        return

    # --- Use 'match_id' to get one row per unique match ---
    # This is the key change based on your feedback.
    unique_matches = matches_df.drop_duplicates(subset=['match_id'])
    
    # --- 1. Get a list of all unique teams ---
    teams = pd.concat([unique_matches['batting_team'], unique_matches['bowling_team']]).unique()
    teams.sort()
    
    # --- 2. Create an empty matrix to store wins ---
    h2h_matrix = pd.DataFrame(0, index=teams, columns=teams)

    # --- 3. Populate the matrix with win counts ---
    for _, row in unique_matches.iterrows():
        if row['match_result'] == 1:
            winner, loser = row['bowling_team'], row['batting_team']
        else:
            winner, loser = row['batting_team'], row['bowling_team']
        
        if winner in h2h_matrix.index and loser in h2h_matrix.columns:
            h2h_matrix.loc[winner, loser] += 1

    # --- 4. Calculate total matches played by each team ---
    matches_played = pd.concat([
        unique_matches['batting_team'], 
        unique_matches['bowling_team']
    ]).value_counts().sort_index()
    
    
    print("--- Total Unique Matches Played by Each Team (from Filenames) ---")
    print(matches_played.to_string())

    # --- 5. Plot the colorful head-to-head matrix ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 14))  # ✅ Larger figure
    
    # ✅ Larger annotation text
    sns.heatmap(h2h_matrix, annot=True, cmap='Blues', fmt='g', 
                linewidths=.5, ax=ax,
                annot_kws={'fontsize': 25, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8})
    
    # ✅ LARGER FONTS
    ax.set_title('Head-to-Head Team Wins Matrix', fontsize=30, fontweight='bold', pad=20)
    ax.set_xlabel('Losing Team', fontsize=30, fontweight='bold', labelpad=10)
    ax.set_ylabel('Winning Team', fontsize=30, fontweight='bold', labelpad=10)
    ax.tick_params(axis='x', labelsize=30, rotation=45)
    ax.tick_params(axis='y', labelsize=30, rotation=0)
    
    # ✅ Larger colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30, width=2, length=8)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_filename = 'head_to_head_wins_matrix_by_id.pdf'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ Matrix plot successfully saved as '{output_filename}'")
    
    plt.show()

def main():
    """
    Main function to load ONLY TRAINING data and run the analysis.
    """
    # --- Define folder paths ---
    # We will ONLY use the training folders to build the matrix
    train_folder_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\training_data_innings1_encoded"
    train_folder_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\training_data_innings2_encoded"
    
    # --- MODIFIED: Only use training paths ---
    folder_paths = [
        train_folder_innings1,
        train_folder_innings2
    ]

    print("--- Loading TRAINING data and extracting match IDs from filenames... ---")
    all_dataframes = [load_data_from_folder(path) for path in folder_paths]
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    if not combined_df.empty:
        print("\nTraining data loaded successfully. Creating matrix...")
        create_head_to_head_matrix(combined_df)
    else:
        print("\nNo data was loaded. Please check folder paths and file contents.")

if __name__ == "__main__":
    main()