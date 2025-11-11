# distribution_plot.py

# import packages
import sys
import tomllib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os

# Assuming correct path for settings is now:
from data_handling.settings import Folders, CleanConfig 

# --- LOGGING SETUP (Copied from time_series.py) ---
logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")


class DistributionAnalyzer:
    """
    A class to handle distribution analysis and visualization based on 
    the feature-engineered DataFrame.
    """
    def __init__(self, config: CleanConfig, output_filename: str):
        """
        Initializes the analyzer, using config for folder paths.
        
        :param config: The loaded application configuration object.
        :param output_filename: The name of the final plot image file.
        """
        self.folders = config.folders
        self.output_filename = output_filename
        self.df = None
        # self.output_path will be constructed in the run method
        
    def _generate_plot(self, df: pd.DataFrame) -> plt.Figure:
        """
        Generates the KDE plot comparing 'react_time_min' distribution 
        based on the 'living_in_city' flag.
        
        :param df: The input DataFrame.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating KDE distribution plot.")
        
        # NOTE: The plotting logic requires a 'react_time_min_log' column, 
        # but the required_cols check only asks for 'react_time_min'.
        # We assume 'react_time_min_log' is either created during feature engineering
        # or we create it here if it's based directly on 'react_time_min'.
        if 'react_time_min_log' not in df.columns:
            # Safely create the log column before splitting (adding a small epsilon to avoid log(0))
            df['react_time_min_log'] = np.log(df['react_time_min'] + 1e-6)
            logger.warning("    -> 'react_time_min_log' was not found, generating log transformation now.")

        # Split data based on the binary feature
        city_dwellers = df[df['living_in_city'] == 1]['react_time_min_log']
        non_city_dwellers = df[df['living_in_city'] == 0]['react_time_min_log']

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use seaborn for KDE to get the smooth distribution curves
        sns.kdeplot(non_city_dwellers, color='limegreen', fill=True, alpha=0.5, label='Living in hometown', ax=ax)
        sns.kdeplot(city_dwellers, color='darkorange', fill=True, alpha=0.5, label='Away from hometown', ax=ax)

        # Apply titles and labels
        fig.suptitle('Response Time: How Location Matters', fontsize=18, y=1.0)
        ax.set_title('Friends living in hometown respond to messages slower than those who have moved away.', fontsize=10, loc='center', y=1.0)
        
        ax.set_xlabel('Response time in log minutes', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        ax.legend(loc='upper right')
        
        # Ensures that titles/labels don't get cut off when saving
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        return fig

    def run(self) -> Path:
        """
        Runs the distribution analysis pipeline: loads latest data, generates plot, 
        and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        # --- File Discovery (Input) ---
        input_files = list(self.folders.feature_added.glob("*-features.csv"))
        
        if not input_files:
            logger.error(f"No *-features.csv files found in {self.folders.feature_added}. Exiting.")
            raise FileNotFoundError(f"No feature-engineered CSV files found in {self.folders.feature_added}")
            
        input_path = max(input_files, key=os.path.getmtime)
        logger.info(f"Selected latest file for analysis: {input_path.name}")
        
        # --- Output Path Construction ---
        plot_output_dir = Path("img/final").resolve()
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = plot_output_dir / self.output_filename

        logger.info(f"Loading data for distribution analysis from: {input_path.name}")
        
        try:
            # Attempt to load Parquet first, then fallback to CSV
            parquet_path = input_path.with_suffix(".parq")
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                # Assuming 'timestamp' is not needed for this plot, but keeping 
                # parse_dates just in case, or removing if it fails/is unnecessary.
                self.df = pd.read_csv(input_path) 
                
        except Exception as e:
            logger.error(f"Failed to load data from {input_path}: {e}")
            raise
        
        # Check if the necessary columns are present
        required_cols = ['living_in_city', 'react_time_min']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting distribution analysis and visualization...")
        
        # Generate the plot
        fig = self._generate_plot(self.df.copy()) # Use a copy for safe log transformation
        
        # Save the figure
        logger.info(f"Saving distribution plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Distribution analysis complete.")
        plt.close(fig) 
        return self.output_path


# --- Configuration Loading and Public Function (Copied from time_series.py) ---
def _load_config() -> CleanConfig:
    """Loads configuration from config.toml and returns a CleanConfig object."""
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    raw = Path(config["raw"])
    preprocessed = Path(config["preprocessed"])
    cleaned = Path(config["cleaned"])
    feature_added = Path(config["feature_added"])
    datafile = Path(config["input"])

    folders = Folders(
        raw=raw,
        preprocessed=preprocessed,
        cleaned=cleaned,
        feature_added=feature_added,
        datafile=datafile,
    )
    
    clean_config = CleanConfig(folders=folders)
    return clean_config

def run_distribution_analysis(output_filename: str) -> Path:
    """
    Main entry point for the distribution analysis and visualization process.
    
    This function loads the configuration, instantiates the DistributionAnalyzer, 
    and runs the full pipeline.
    
    :param output_filename: The name of the final plot image file to save.
    :return: Path to the final plot image file.
    """
    try:
        config = _load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise RuntimeError("Failed to load plotting configuration.")
        
    analyzer = DistributionAnalyzer(
        config=config,
        output_filename=output_filename
    )
    return analyzer.run()