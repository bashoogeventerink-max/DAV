# import packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

class DistributionAnalyzer:
    """
    A class to handle distribution analysis and visualization based on 
    the feature-engineered DataFrame.
    """
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the feature-engineered CSV/Parquet file.
        :param output_path: Path where the generated plot image (.png) will be saved.
        :param config: The loaded configuration dictionary.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _generate_plot(self, df: pd.DataFrame) -> plt.Figure:
        """
        Generates the KDE plot comparing 'react_time_min' distribution 
        based on the 'living_in_city' flag.
        
        :param df: The input DataFrame.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating KDE distribution plot.")
        
        # Split data based on the binary feature
        city_dwellers = df[df['living_in_city'] == 1]['react_time_min']
        non_city_dwellers = df[df['living_in_city'] == 0]['react_time_min']

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use seaborn for KDE to get the smooth distribution curves
        sns.kdeplot(non_city_dwellers, color='darkorange', fill=True, alpha=0.5, label='Living in hometown', ax=ax)
        sns.kdeplot(city_dwellers, color='limegreen', fill=True, alpha=0.5, label='Away from hometown', ax=ax)

        # Apply titles and labels
        fig.suptitle('Staying Put, Responding Quicker: How Location Matters', fontsize=18, y=1.0)
        ax.set_title('Friends living in hometown respond to messages significantly faster than those who have moved away.', fontsize=10, loc='center', y=1.0)
        
        ax.set_xlabel('Response time in minutes', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Set x-limit for better visualization of the dense area
        ax.set_xlim(0, 30) 
        
        ax.legend(loc='upper right')
        
        # Ensures that titles/labels don't get cut off when saving
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        return fig

    def run(self) -> Path:
        """
        Runs the distribution analysis pipeline: loads data, generates plot, 
        and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        logger.info(f"Loading data for distribution analysis from: {self.input_path.name}")
        
        try:
            # Attempt to load Parquet first, then fallback to CSV
            parquet_path = self.input_path.with_suffix(".parq")
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                self.df = pd.read_csv(self.input_path, parse_dates=["timestamp"])
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
        
        # Check if the necessary columns are present
        required_cols = ['living_in_city', 'react_time_min']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting distribution analysis and visualization...")
        
        # Generate the plot
        fig = self._generate_plot(self.df)
        
        # Save the figure
        logger.info(f"Saving distribution plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Distribution analysis complete.")
        # Close the plot to free up memory
        plt.close(fig) 
        return self.output_path

# Public function to be used in main.py

def run_distribution_analysis(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the distribution analysis and visualization process.
    
    :param input_path: Path to the feature-engineered CSV.
    :param output_path: Path to save the final plot image (.png).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = DistributionAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()