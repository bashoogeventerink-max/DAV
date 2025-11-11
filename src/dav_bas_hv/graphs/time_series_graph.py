# dual_axis_trends_analyzer.py (time_series.py)

# import packages
import sys
import tomllib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os

# Assuming correct path for settings is now:
from data_handling.settings import Folders, CleanConfig 

logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")


class DualAxisTrendsAnalyzer:
    """
    A class to handle the analysis and visualization of monthly message volume
    and average word count over time using a dual-axis plot.
    """
    
    def __init__(self, config: CleanConfig, output_filename: str):
        self.folders = config.folders
        self.output_filename = output_filename
        self.df = None
        self.trends_df = None
        
        # Output path is constructed in the run method
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw DataFrame to calculate monthly message volume, 
        average word count, and their 6-month rolling mean trend lines.
        
        :param df: The input DataFrame.
        :return: A DataFrame with calculated monthly trends and trend lines.
        """
        logger.info("    -> Resampling data to monthly frequency and calculating trends.")
        
        # 1. Ensure the timestamp column is the index
        # We assume 'timestamp' is already datetime from the loading step in run()
        df = df.set_index('timestamp')

        # 2. Calculate Monthly Trends
        monthly_message_count = df.resample('M').size().rename('message_count')
        monthly_avg_word_count = df['word_count'].resample('M').mean().rename('avg_word_count')

        # Combine the two series into one DataFrame
        trends_df = pd.concat([monthly_message_count, monthly_avg_word_count], axis=1).dropna(how='all')

        # 3. Calculate Trend Lines (6-month Rolling Mean - kept 6 for smoother trend)
        window = 3
        trends_df['volume_trend'] = trends_df['message_count'].rolling(
            window=window, center=True
        ).mean()
        trends_df['word_count_trend'] = trends_df['avg_word_count'].rolling(
            window=window, center=True
        ).mean()
        
        return trends_df
        
    def _generate_plot(self, trends_df: pd.DataFrame, partnership_dates: pd.Series) -> plt.Figure:
        """
        Generates the dual-axis line plot with custom colors, line styles, and axis ranges.
        
        :param trends_df: The DataFrame containing the monthly trend data.
        :param partnership_dates: Series of unique partnership start dates.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating dual-axis plot.")
        
        # Create the figure and axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Define colors 
        color_volume = '#ff7f0e'  # Orange
        color_words = '#808080'   # Grey

        # --- Left Y-Axis (Message Volume) ---
        ax1.set_xlabel('Date (Monthly)')
        ax1.set_ylabel('Monthly Message Volume (6-Month Trend)', color=color_volume)

        # CHANGE 1: Plot ONLY the trend line (now solid)
        ax1.plot(
            trends_df.index, trends_df['volume_trend'], color=color_volume, 
            linewidth=2.5, linestyle='-', label='Volume Trend'
        )

        ax1.tick_params(axis='y', labelcolor=color_volume)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 800) 

        # --- Right Y-Axis (Average Word Count) ---
        ax2 = ax1.twinx()

        ax2.set_ylabel('Monthly Average Word Count (6-Month Trend)', color=color_words)

        # CHANGE 1: Plot ONLY the trend line (now solid)
        ax2.plot(
            trends_df.index, trends_df['word_count_trend'], color=color_words, 
            linewidth=2.5, linestyle='-', label='Avg. Word Count Trend'
        )

        ax2.tick_params(axis='y', labelcolor=color_words)
        ax2.grid(False)
        ax2.set_ylim(0, 15)

        # --- NEW CHANGE 3: Add Vertical Lines for Partnership Dates ---
        # Get unique, non-NaT dates for partnership
        unique_dates = partnership_dates.dropna().unique()
        
        if len(unique_dates) > 0:
            logger.info(f"    -> Adding {len(unique_dates)} vertical lines for partnership start dates.")
            
            # Use ax1 to plot the vertical lines (they span both axes)
            for i, date in enumerate(unique_dates):
                # Ensure the date is a timestamp object for plotting
                date_dt = pd.to_datetime(date)
                
                # Add a vertical dashed line
                ax1.axvline(
                    x=date_dt, 
                    color='red', 
                    linestyle=':', 
                    linewidth=1.0, 
                    label=f'Moving in with girlfriend' if i == 0 else "" # Only label the first one for the legend
                )

        # Title and Layout
        plt.title(
            'Moving In Together Causes In Less Group Chat Activity With Friends, but Results In Longer, More Detailed Messages.', 
            fontsize=14
        )
        
        # Rotate x-axis labels
        fig.autofmt_xdate(rotation=45)

        # Adding legends manually since they are on different axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Find the line for the vertical event (if present)
        event_lines = [line for line, label in zip(lines1, labels1) if 'Moving In Event' in label]
        event_labels = [label for label in labels1 if 'Moving In Event' in label]
        
        # Combine all legends: ax1 trend + ax2 trend + event line
        ax1.legend(
            [lines1[0]] + lines2 + event_lines, 
            [labels1[0]] + labels2 + event_labels, 
            loc='upper left'
        )
        
        fig.tight_layout()
        
        return fig

    def run(self) -> Path:
        """
        Runs the plotting pipeline: loads latest data, performs preparation, 
        generates plot, and saves the final image.
        
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

        logger.info(f"Loading data for trends analysis from: {input_path.name}")
        
        try:
            # Load with parse_dates for necessary columns
            self.df = pd.read_csv(input_path, parse_dates=['timestamp', 'date_living_with_partner'])
        except Exception as e:
            logger.error(f"Failed to load data from {input_path}: {e}")
            raise
        
        # Check if the necessary column is present
        required_cols = ['timestamp', 'word_count', 'date_living_with_partner']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting dual-axis trends analysis and visualization...")
        
        # 1. Prepare the data
        self.trends_df = self._prepare_data(self.df.copy())
        
        # 2. Get unique partnership dates for vertical lines
        partnership_dates = self.df['date_living_with_partner']
        
        # 3. Generate the plot
        fig = self._generate_plot(self.trends_df, partnership_dates)
        
        # 4. Save the figure
        logger.info(f"Saving dual-axis trends plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Dual-axis trends analysis complete.")
        plt.close(fig) 
        return self.output_path


# --- Configuration Loading and Public Function (NO CHANGE) ---
def _load_config() -> CleanConfig:
# ... (function body remains the same) ...
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

def run_dual_axis_analysis(output_filename: str) -> Path:
# ... (function body remains the same) ...
    try:
        config = _load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise RuntimeError("Failed to load plotting configuration.")
        
    analyzer = DualAxisTrendsAnalyzer(
        config=config,
        output_filename=output_filename
    )
    return analyzer.run()