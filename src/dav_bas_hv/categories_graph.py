# yearly_questions_graph.py

# import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

class MeetingUpQuestionsAnalyzer:
    """
    A class to handle the analysis and visualization of 'meeting up' questions
    over time, stratified by whether the user is living in a city.
    
    The analysis calculates the percentage of meeting up questions from city vs. 
    non-city living users per year.
    """
    
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the main feature-engineered data file (CSV/Parquet).
        :param output_path: Path where the generated plot image (.png) will be saved.
        :param config: The loaded configuration dictionary (currently unused but kept for style).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        self.yearly_stats = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw DataFrame to calculate yearly percentages of 
        meeting up questions by living location (city vs. non-city).
        
        This logic is a direct transcription of steps 1-7 from your Jupyter notebook.
        
        :param df: The input DataFrame.
        :return: A DataFrame with yearly percentages.
        """
        logger.info("    -> Preparing data and calculating yearly percentages.")
        
        # 1. Ensure the combined flag is correctly calculated
        df['is_meeting_up_question'] = (
            (df['is_question'] == 1) & (df['mentions_meet_up'] == 1)
        ).astype(int)

        # 2. Group by both 'year' and the condition ('living_in_city') and sum the questions
        yearly_counts = df.groupby(['year', 'living_in_city'])[
            'is_meeting_up_question'
        ].sum().reset_index()

        # 3. Pivot the table to get the city/non-city counts as separate columns
        yearly_stats = yearly_counts.pivot(
            index='year',
            columns='living_in_city',
            values='is_meeting_up_question'
        ).fillna(0).reset_index()

        # 4. Rename columns for clarity (0 is non_city, 1 is city)
        yearly_stats.columns = [
            'year', 
            'meeting_up_questions_hometown', 
            'meeting_up_questions_away_from_hometown'
        ]

        # 5. Calculate the total meeting up questions
        yearly_stats['total_meeting_up_questions'] = (
            yearly_stats['meeting_up_questions_hometown'] + yearly_stats['meeting_up_questions_away_from_hometown']
        )

        # 6. Calculate the percentages (%)
        # We use .div() for clarity and efficiency when dividing multiple columns
        yearly_stats[['pct_hometown_living', 'pct_away_from_hometown_living']] = (
            yearly_stats[['meeting_up_questions_hometown', 'meeting_up_questions_away_from_hometown']]
            .div(yearly_stats['total_meeting_up_questions'], axis=0) * 100
        )

        # Handle cases where total_meeting_up_questions is 0 (division by zero results in NaN)
        yearly_stats[['pct_hometown_living', 'pct_away_from_hometown_living']] = yearly_stats[
            ['pct_hometown_living', 'pct_away_from_hometown_living']
        ].fillna(0)
        
        return yearly_stats
        
    def _generate_plot(self, yearly_stats: pd.DataFrame) -> plt.Figure:
        """
        Generates the line plot showing the percentage trend over years.
        
        :param yearly_stats: The DataFrame containing the yearly percentage data.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating line plot.")
        
        # Create the figure and axes
        # We use plt.subplots to get the figure object, matching the example style
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply the style *to the figure object* (or globally before, but setting it here is clearer)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot the Non-City Living percentage
        ax.plot(
            yearly_stats['year'],
            yearly_stats['pct_hometown_living'],
            label='Living in Hometown',
            marker='o',
            linestyle='-',
            color='#808080' # Grey
        )

        # Plot the City Living percentage
        ax.plot(
            yearly_stats['year'],
            yearly_stats['pct_city_living'],
            label='Living away from Hometown',
            marker='s',
            linestyle='--',
            color='#ff7f0e' # Orange
        )

        # Set the title, labels
        plot_title = "Location does not seem to matter: Taking initiative in meeting up comes from non-hometowners"
        ax.set_title(plot_title, fontsize=14, pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Percentage of Meeting Up Questions (%)", fontsize=12)

        # Ensure X-axis ticks are integers for years
        # Use a list of year values to set the ticks
        ax.set_xticks(yearly_stats['year'].astype(int).tolist())

        # Add a legend
        ax.legend(loc='upper right', fontsize=10)

        # Add a grid for better readability (using the existing axes grid)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Improve layout
        fig.tight_layout()
        
        return fig

    def run(self) -> Path:
        """
        Runs the plotting pipeline: loads data, performs preparation, 
        generates plot, and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        logger.info(f"Loading data for yearly questions analysis from: {self.input_path.name}")
        
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
        required_cols = ['year', 'is_question', 'mentions_meet_up', 'living_in_city']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Missing required columns for plotting: {required_cols}")
            raise ValueError(f"Data is missing required columns: {required_cols}")
            
        logger.info("Starting yearly questions analysis and visualization...")
        
        # 1. Prepare the data
        self.yearly_stats = self._prepare_data(self.df)
        
        # 2. Generate the plot
        fig = self._generate_plot(self.yearly_stats)
        
        # 3. Save the figure
        logger.info(f"Saving yearly questions plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("Yearly questions analysis complete.")
        # Close the plot to free up memory
        plt.close(fig) 
        return self.output_path

# --- Public function to be used in main.py ---

def run_categories_analysis(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the yearly questions analysis and visualization process.
    
    :param input_path: Path to the feature-engineered data file.
    :param output_path: Path to save the final plot image (.png).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = MeetingUpQuestionsAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()