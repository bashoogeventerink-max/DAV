# dual_axis_trends_analyzer.py

# import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

class DualAxisTrendsAnalyzer:
    """
    A class to handle the analysis and visualization of monthly message volume
    and average word count over time using a dual-axis plot.
    """
    
    def __init__(self, input_path: Path, output_path: Path, config: dict = None):
        """
        :param input_path: Path to the main feature-engineered data file (CSV).
        :param output_path: Path where the generated plot image (.png) will be saved.
        :param config: The loaded configuration dictionary (kept for style).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config if config is not None else {}
        self.df = None
        self.trends_df = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw DataFrame to calculate monthly message volume, 
        average word count, and their 3-month rolling mean trend lines.
        
        :param df: The input DataFrame.
        :return: A DataFrame with calculated monthly trends and trend lines.
        """
        logger.info("    -> Resampling data to monthly frequency and calculating trends.")
        
        # 1. Ensure the timestamp column is the index
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')

        # 2. Calculate Monthly Trends
        monthly_message_count = df.resample('M').size().rename('message_count')
        monthly_avg_word_count = df['word_count'].resample('M').mean().rename('avg_word_count')

        # Combine the two series into one DataFrame
        trends_df = pd.concat([monthly_message_count, monthly_avg_word_count], axis=1).dropna(how='all')

        # 3. Calculate Trend Lines (3-month Rolling Mean)
        window = 6
        trends_df['volume_trend'] = trends_df['message_count'].rolling(
            window=window, center=True
        ).mean()
        trends_df['word_count_trend'] = trends_df['avg_word_count'].rolling(
            window=window, center=True
        ).mean()
        
        return trends_df
        
    def _generate_plot(self, trends_df: pd.DataFrame) -> plt.Figure:
        """
        Generates the dual-axis line plot with custom colors, line styles, and axis ranges.
        
        :param trends_df: The DataFrame containing the monthly trend data.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating dual-axis plot.")
        
        # Create the figure and axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Define colors based on user request
        color_volume = '#ff7f0e'  # Orange
        color_words = '#808080'   # Grey

        # --- Left Y-Axis (Message Volume) ---
        ax1.set_xlabel('Date (Monthly)')
        ax1.set_ylabel('Monthly Message Volume', color=color_volume)

        # Plot raw data (solid line)
        ax1.plot(
            trends_df.index, trends_df['message_count'], color=color_volume, 
            linewidth=1.5, linestyle='-', alpha=0.8, label='Monthly Volume Messages'
        )

        # Plot trend line (dashed line)
        ax1.plot(
            trends_df.index, trends_df['volume_trend'], color=color_volume, 
            linewidth=2.5, linestyle='--', label='Monthly Volume Messages (3-month Rolling Mean)'
        )

        ax1.tick_params(axis='y', labelcolor=color_volume)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 800) # Increased range

        # --- Right Y-Axis (Average Word Count) ---
        ax2 = ax1.twinx()

        ax2.set_ylabel('Monthly Average Word Count', color=color_words)

        # Plot raw data (solid line)
        ax2.plot(
            trends_df.index, trends_df['avg_word_count'], color=color_words, 
            linewidth=1.5, linestyle='-', alpha=0.8, label='Average Word Count'
        )

        # Plot trend line (dashed line)
        ax2.plot(
            trends_df.index, trends_df['word_count_trend'], color=color_words, 
            linewidth=2.5, linestyle='--', label='Average Word Count (3-month Rolling Mean)'
        )

        ax2.tick_params(axis='y', labelcolor=color_words)
        ax2.grid(False)
        ax2.set_ylim(0, 15) # Increased range

        # Title and Layout
        plt.title(
            'Less, but longer messages over time', 
            fontsize=14
        )
        
        # Rotate x-axis labels
        fig.autofmt_xdate(rotation=45)

        # Adding legends manually since they are on different axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()
        
        return fig

    def run(self) -> Path:
        """
        Runs the plotting pipeline: loads data, performs preparation, 
        generates plot, and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        logger.info(f"Loading data for trends analysis from: {self.input_path.name}")
        
        try:
            self.df = pd.read_csv(self.input_path)
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
        
        # Check if the necessary column is present
        required_cols = ['timestamp', 'word_count']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting dual-axis trends analysis and visualization...")
        
        # 1. Prepare the data
        self.trends_df = self._prepare_data(self.df.copy())
        
        # 2. Generate the plot
        fig = self._generate_plot(self.trends_df)
        
        # 3. Save the figure
        logger.info(f"Saving dual-axis trends plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Dual-axis trends analysis complete.")
        # Close the plot to free up memory
        plt.close(fig) 
        return self.output_path

# --- Public function to be used in main.py ---

def run_dual_axis_analysis(input_path: Path, output_path: Path, config: dict = None) -> Path:
    """
    Main entry point for the dual-axis trends analysis and visualization process.
    
    :param input_path: Path to the feature-engineered data file.
    :param output_path: Path to save the final plot image (.png).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = DualAxisTrendsAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()