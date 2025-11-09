# import packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy import stats
from matplotlib.ticker import FixedLocator

class CorrelationAnalyzer:
    """
    A class to handle correlation analysis and visualization between a 
    dichotomous feature and a binary target based on the feature-engineered DataFrame.
    
    Specific analysis: Point-Biserial Correlation and T-test for 'tech_background'
    and 'has_emoji'.
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
        
    def _perform_analysis(self, df: pd.DataFrame) -> tuple[float, float, float]:
        """
        Performs the Point-Biserial correlation and T-test.
        
        :param df: The input DataFrame.
        :return: A tuple containing (correlation_coefficient, t_stat, p_value).
        """
        logger.info("    -> Calculating Point-Biserial Correlation and T-test.")
        
        # Calculate the Point-Biserial Correlation Coefficient
        # This is equivalent to Pearson's correlation when one variable is dichotomous
        correlation_coefficient, _ = stats.pearsonr(df['tech_background'], df['has_emoji'])
        
        # T-test for difference in means (proportions) between the two groups
        group_tech = df[df['tech_background'] == 1]['has_emoji']
        group_nontech = df[df['tech_background'] == 0]['has_emoji']
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(group_tech, group_nontech, equal_var=False) 
        
        return correlation_coefficient, t_stat, p_value
        
    def _generate_plot(self, df: pd.DataFrame, correlation_coefficient: float, p_value: float) -> plt.Figure:
        """
        Generates the bar plot showing the proportion of messages with emojis 
        based on the user's technical background.
        
        :param df: The input DataFrame.
        :param correlation_coefficient: The calculated Point-Biserial correlation (r).
        :param p_value: The calculated T-test p-value.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating Bar Plot for proportions.")
        
        # Map the binary variable to clear labels for plotting
        df['tech_label'] = df['tech_background'].astype(int).astype(str).replace(
            {'0': 'No Technical Background', '1': 'Technical Background'}
        )

        # --- Calculate Group Sizes (N) ---
        group_counts = df['tech_label'].value_counts().to_dict()
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Use seaborn for a bar plot. The default estimator 'mean' will correctly 
        # represent the proportion of 'has_emoji' (since it is 0 or 1).
        sns.barplot(
            x='tech_label', 
            y='has_emoji', 
            data=df,
            hue='tech_label',
            legend=False,
            palette=['#4c72b0', '#55a868'],
            errcolor='gray', 
            capsize=0.1,
            linewidth=1.5,
            ax=ax
        )


        # --- NEW: Add N-labels above the bars ---
        for i, bar in enumerate(ax.patches):
            # Get the bar's x-label (e.g., 'No Technical Background')
            # The order of the bars matches the order in value_counts() if not sorted 
            # by default, but iterating through the categories is safer.
            category_label = df['tech_label'].unique()[i]
            
            # Get the count for this label
            N_count = group_counts.get(category_label, 0)
            N_text = f"N={N_count}"

            # Add the text label slightly above the bar/error bar
            # bar.get_height() is the mean (proportion) of 'has_emoji'
            # We use an offset of 0.01 for visual spacing
            ax.text(
                bar.get_x() + bar.get_width() / 2, # Center the text horizontally
                bar.get_height() + 0.01,           # Position text slightly above the bar
                N_text,
                ha='center',                       # Horizontal alignment: center
                va='bottom',                       # Vertical alignment: bottom
                fontsize=10,
                color='black'
            )

        # --- Add Statistical Annotation to the Graph ---
        corr_text = f"Correlation (r): {correlation_coefficient:.2f}"
        p_text = f"P-value (t-test): {p_value:.3f}"

        # Determine text position (relative to the plot axes)
        x_pos = 0.05
        y_pos = 0.95
        text_y_offset = 0.05

        ax.text(x_pos, y_pos, corr_text, 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top', 
                horizontalalignment='left')

        ax.text(x_pos, y_pos - text_y_offset, p_text, 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top', 
                horizontalalignment='left')

        # --- Set the title and labels ---
        ax.set_title('Users with Technical Background use Emoji less', fontsize=16, pad=15)
        ax.set_xlabel('Studied in Technical Field', fontsize=12)
        ax.set_ylabel('% of Messages with Emoji', fontsize=12)

        # Format Y-axis as percentage
        # Use fig.canvas.draw() to ensure ticks are calculated before formatting
        fig.canvas.draw()
        tick_locs = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(tick_locs))
        ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in tick_locs])

        # --- Enhance Aesthetics ---
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Ensures that titles/labels don't get cut off when saving
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        return fig

    def run(self) -> Path:
        """
        Runs the correlation analysis pipeline: loads data, performs analysis, 
        generates plot, and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        logger.info(f"Loading data for correlation analysis from: {self.input_path.name}")
        
        try:
            # Attempt to load Parquet first, then fallback to CSV
            parquet_path = self.input_path.with_suffix(".parq")
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                self.df = pd.read_csv(self.input_path)
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
        
        # Check if the necessary columns are present
        required_cols = ['tech_background', 'has_emoji']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting correlation analysis and visualization...")
        
        # 1. Perform analysis
        correlation_coefficient, _, p_value = self._perform_analysis(self.df)
        
        # 2. Generate the plot
        fig = self._generate_plot(self.df, correlation_coefficient, p_value)
        
        # 3. Save the figure
        logger.info(f"Saving correlation plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Correlation analysis complete.")
        # Close the plot to free up memory
        plt.close(fig) 
        return self.output_path

# --- Public function to be used in main.py ---

def run_correlation_analysis(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the correlation analysis and visualization process.
    
    :param input_path: Path to the feature-engineered CSV.
    :param output_path: Path to save the final plot image (.png).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = CorrelationAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()