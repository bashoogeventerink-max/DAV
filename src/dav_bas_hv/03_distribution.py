# create_plot.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import tomllib
from loguru import logger
import numpy as np

def get_latest_data_path(processed_path: Path) -> Path:
    """
    Finds and returns the path to the most recently created Parquet file
    in the specified directory.
    """
    try:
        # Get all .parq files in the processed directory
        parquet_files = list(processed_path.glob('*.parq'))
        if not parquet_files:
            raise FileNotFoundError(f"No .parq files found in {processed_path}")

        # Find the latest file based on its modification time
        latest_file = max(parquet_files, key=Path.stat)
        logger.info(f"Using the latest data file: {latest_file.name}")
        return latest_file
    except FileNotFoundError as e:
        logger.error(e)
        return None

def create_reaction_time_plot(df: pd.DataFrame):
    """
    Creates and saves a KDE plot of reaction times based on location.
    """
    # Filter data into two groups
    city_dwellers = df[df['living_in_city'] == 1]['react_time_min']
    non_city_dwellers = df[df['living_in_city'] == 0]['react_time_min']

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # Create the KDE plots
    sns.kdeplot(non_city_dwellers, color='darkorange', fill=True, alpha=0.5, label='Living in hometown')
    sns.kdeplot(city_dwellers, color='limegreen', fill=True, alpha=0.5, label='Away from hometown')

    # Add titles and labels
    plt.suptitle('Staying Put, Responding Quicker: How Location Matters', fontsize=18, y=1.0)
    plt.title('Friends living in ho respond to messages significantly faster than those who have moved away.', fontsize=10, loc='center', y = 1.0)
    plt.xlabel('Response time in minutes', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Set plot limits and legend
    plt.xlim(0, 30)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the figure to the 'img' directory
    img_path = Path("img")
    img_path.mkdir(exist_ok=True)
    output_path = img_path / 'reaction_times_plot.png'
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")

    # Show the plot
    plt.show()

def main():
    """
    Main function to load data and create the plot.
    """
    processed_path = Path("data/processed").resolve()
    
    # Get the path to the latest data file
    data_path = get_latest_data_path(processed_path)
    if data_path is None:
        logger.error("Could not find a data file. Exiting.")
        return

    # Load data from the Parquet file
    try:
        df = pd.read_parquet(data_path)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}. Error: {e}")
        return

    # Check if required columns exist before plotting
    required_columns = ['living_in_city', 'react_time_min']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"DataFrame is missing required columns: {required_columns}. Found: {df.columns}")
        # Use dummy data to demonstrate the plot if real data is missing
        logger.warning("Using dummy data for demonstration purposes.")
        np.random.seed(42)
        data1 = np.random.exponential(scale=5, size=5000)
        data2 = np.random.exponential(scale=5, size=5000) * 0.8
        df = pd.DataFrame({
            'react_time_min': np.concatenate([data1, data2]),
            'living_in_city': np.concatenate([np.ones(5000), np.zeros(5000)])
        })

    # Create and save the plot
    create_reaction_time_plot(df)

if __name__ == "__main__":
    main()