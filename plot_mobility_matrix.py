import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Path to the file
    file_path = Path("mobility_Madrid/mobility_matrix_Madrid.csv")

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        print("Please ensure the 'mobility_Madrid' folder exists in the current directory.")
        return

    print(f"Reading {file_path}...")
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Data cleaning logic (similar to data_loader.py)
        # Check if the matrix is square. If columns = rows + 1, assume first column is index/ID.
        if df.shape[1] == df.shape[0] + 1:
            print("  -> Detected extra column (likely index). Dropping first column to align matrix.")
            df = df.iloc[:, 1:]
        
        # Ensure matrix is square now
        if df.shape[0] != df.shape[1]:
            print(f"Warning: Matrix is not square ({df.shape[0]} rows, {df.shape[1]} columns). Heatmap might be distorted.")

        # Use integer indices for axes instead of patch names
        df.columns = range(df.shape[1])
        df.index = range(df.shape[0])

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        # cmap="Reds" ensures darker red for higher values
        ax = sns.heatmap(df, cmap="Reds", square=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label('fraction of residents', fontsize=12)
        
        # plt.title(f"Mobility Matrix Heatmap: {file_path.name}")
        plt.title(f"Mobility Matrix Heatmap: Madrid", fontsize=14)
        plt.xlabel("Destination District Index", fontsize=12)
        plt.ylabel("Origin District Index", fontsize=12)
        
        # Save plot
        output_filename = "mobility_matrix_heatmap.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved heatmap to '{output_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()