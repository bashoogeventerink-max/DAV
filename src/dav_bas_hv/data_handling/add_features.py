# add_features.py

# import packages
import re
import sys
import tomllib
from pathlib import Path
import pandas as pd
import numpy as np 
from loguru import logger
from textblob import TextBlob
from typing import Union
from datetime import datetime
import click
import os

# Import from settings - assuming these are defined elsewhere or copied here for completeness
# For this script to work, you need 'Folders' and 'CleanConfig' from your settings
# Assuming 'Folders' and 'CleanConfig' are available, e.g., from 'wa_analyzer.settings'
from .settings import Folders, CleanConfig 

# Configure Loguru (copied from clean_data.py)
logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# --- FeatureEngineer Class ---
class FeatureEngineer:
    """
    A class to handle feature engineering steps on a cleaned DataFrame.
    It takes the output of the cleaning step and adds new analytical features.
    """
    # **CHANGE 1: Use CleanConfig for initialization**
    def __init__(self, config: CleanConfig):
        """
        :param config: The loaded configuration object including Folders.
        """
        self.folders = config.folders
        self.df = None
        
        # **CHANGE 2: Use self.folders.feature_added for the output directory**
        # Ensure the output directory exists
        self.folders.feature_added.mkdir(parents=True, exist_ok=True)


    def _add_timestamp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds timestamp-based features to the DataFrame."""
        logger.info("    -> Adding timestamp features.")
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = np.where(df['day_of_week'].isin(['Saturday', 'Sunday']), 1, 0)
        return df
        
    def _get_sentiment_polarity(self, text: Union[str, float]) -> float:
        """
        Calculates the sentiment polarity (-1.0 to 1.0) of a given text.
        Returns 0.0 (Neutral) for missing or non-string values.
        """
        if pd.isna(text):
            return 0.0 
        # TextBlob analysis
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sentiment features ('sentiment_polarity' and 'sentiment_category') 
        to the DataFrame based on the 'message' column.
        """
        logger.info("    -> Adding sentiment features.")
        # 1. Add numerical polarity score
        df['sentiment_polarity'] = df['message'].apply(self._get_sentiment_polarity)
        
        # 2. Add categorical sentiment for high-level analysis (using a small buffer for Neutral)
        def classify_sentiment(polarity):
            if polarity > 0.05:
                return 'Positive'
            elif polarity < -0.05:
                return 'Negative'
            else:
                return 'Neutral'
                
        df['sentiment_category'] = df['sentiment_polarity'].apply(classify_sentiment)
        return df

    def _emoji_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a feature column 'has_emoji' and 'emoji_count' based on message content."""
        logger.info("    -> Adding 'has_emoji' and 'emoji_count' features.")
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        def has_emoji(text):
            return bool(emoji_pattern.search(str(text)))

        def get_emoji_count(text):
            return len(emoji_pattern.findall(str(text)))

        df["has_emoji"] = df["message"].apply(has_emoji).astype(int)
        df["emoji_count"] = df["message"].apply(get_emoji_count)
        return df
    
    def _is_question(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a feature column 'is_question' indicating if the message is a question."""
        logger.info("    -> Adding 'is_question' feature.")
        df["is_question"] = df["message"].astype(str).apply(lambda x: '?' in x).astype(int)
        return df
    
    def _meet_up_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a feature column 'mentions_meet_up' indicating if the message mentions to meet up."""
        logger.info("    -> Adding 'mentions_meet_up' feature.")
        meet_up_keywords = [
            'afspreken', 'biertje', 'bier', 'vnv', 'vanavond', 'drinken', 'pils', 'pilsje', 'wat doen', 'weekend', 'vrijdag', 'vrijdagavond', 'zaterdag', 'zaterdagavond'            
        ]

        df["mentions_meet_up"] = df["message"].astype(str).str.lower().apply(
            lambda x: any(word in x for word in meet_up_keywords)
        ).astype(int)
        return df

    def _add_drink_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'drink_count': 1 if the message matches drink keywords and/or beer, wine, or champagne emojis."""
        logger.info("    -> Adding 'drink_count' feature.")
        drink_keywords = [
            'biertje', 'biertjes', 'bier', 'pils', 'pilsje', 'pilsjes', 'wijntje', 'pilsemannetje', 'grolsch', 'hertog', 'heiniken'
        ]
        # Beer: mug + clinking mugs; wine: glass; champagne: clinking glasses + bottle with cork
        drink_emojis = (
            "\U0001f37a",  # beer mug
            "\U0001f37b",  # clinking beer mugs
            "\U0001f377",  # wine glass
            "\U0001f942",  # clinking glasses
            "\U0001f37e",  # bottle with popping cork
        )

        def has_drink_signal(text: str) -> bool:
            lower = text.lower()
            if any(word in lower for word in drink_keywords):
                return True
            return any(emoji in text for emoji in drink_emojis)

        df["drink_count"] = df["message"].astype(str).apply(has_drink_signal).astype(int)
        return df

    def _add_word_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the number of words in each message."""
        logger.info("    -> Adding 'word_count' feature.")
        # Ensure message is string and handle NaN gracefully
        df["word_count"] = df["message"].astype(str).str.split().str.len()
        return df
    
    def _add_char_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the length of the message in characters."""
        logger.info("    -> Adding 'char_count' feature.")
        df["char_count"] = df["message"].astype(str).str.len()
        return df

    def _add_time_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the time difference between consecutive messages in seconds,
        minutes, and hours. Requires the DataFrame to be sorted by timestamp.
        """
        logger.info("    -> Adding time difference features ('react_time_sec', etc.).")
        
        # Ensure data is sorted by timestamp and reset index for safe diff
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        df["time_diff"] = df["timestamp"].diff()

        # Seconds, Minutes, Hours
        df["react_time_sec"] = df["time_diff"].dt.total_seconds()
        df["react_time_sec_plus_1"] = df["react_time_sec"] + 1 # To avoid division by zero in analyses
        df["react_time_sec_log"] = np.log(df["react_time_sec_plus_1"]) # Log-transform for skewed distribution
        df["react_time_min"] = df["react_time_sec"] / 60
        df["react_time_min_plus_1"] = df["react_time_min"] + 1 # To avoid division by zero in analyses
        df["react_time_min_log"] = np.log(df["react_time_min_plus_1"]) # Log-transform for skewed distribution                                                                                 
        df["react_time_hr"] = df["react_time_sec"] / 3600
        df["react_time_hr_plus_1"] = df["react_time_hr"] + 1 # To avoid division by zero in analyses
        df["react_time_hr_log"] = np.log(df["react_time_hr_plus_1"]) # Log-transform for skewed distribution

        df.drop(columns=["time_diff"], inplace=True)
        return df

    def _flag_image_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags messages that represent an image."""
        logger.info("    -> Adding 'is_image' flag.")
        df['is_image'] = np.where(
            df['message'].str.contains('<Media weggelaten>', case=False, na=False), 
            1, 
            0
        )
        return df

    def _flag_empty_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags messages that contain 'Wachten op dit bericht'."""
        logger.info("    -> Adding 'is_empty_message' flag.")
        df['is_empty_message'] = np.where(
            df['message'].str.contains('Wachten op dit bericht', case=False, na=False), 
            1, 
            0
        )
        return df

    def _flag_removed_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags messages that have been removed by the user."""
        logger.info("    -> Adding 'is_removed_message' flag.")
        df['is_removed_message'] = np.where(
            df['message'].str.contains('Je hebt dit bericht verwijderd', case=False, na=False), 
            1, 
            0
        )
        return df

    def _save_dataframe(self, df: pd.DataFrame, filename_base: str) -> Path:
        """
        Saves the DataFrame to CSV and Parquet with a timestamped filename.

        :param df: The pandas DataFrame to save.
        :param filename_base: The base name for the file (e.g., "whatsapp-features").
        :return: Path to the final feature-engineered CSV file.
        """
        # Generate the timestamp
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Define output file paths
        # **CHANGE 3: Use self.folders.feature_added for output**
        outfile_csv = self.folders.feature_added / f"{filename_base}-{now}-features.csv"
        outfile_parquet = self.folders.feature_added / f"{filename_base}-{now}-features.parq"
        
        logger.info(f"Writing CSV to {outfile_csv}")
        df.to_csv(outfile_csv, index=False)
        
        logger.info(f"Writing Parquet to {outfile_parquet}")
        df.to_parquet(outfile_parquet, index=False)
        
        logger.success("Saving complete.")
        
        return outfile_csv


    def run(self) -> Path:
        """
        Runs the feature engineering pipeline: loads cleaned data, adds features, 
        and saves the final output.
        
        :return: Path to the final feature-engineered CSV file.
        """
        # **CHANGE 4: Find the latest file in the 'cleaned' folder**
        input_files = list(self.folders.cleaned.glob("*-cleaned.csv"))
        
        if not input_files:
            logger.error(f"No *-cleaned.csv files found in {self.folders.cleaned}. Exiting.")
            raise FileNotFoundError(f"No cleaned CSV files found in {self.folders.cleaned}")
            
        # Find the file with the most recent modification time (mtime)
        input_csv_file = max(input_files, key=os.path.getmtime)
        input_parquet_file = input_csv_file.with_suffix(".parq")
        
        logger.info(f"Selected latest file for feature engineering: {input_csv_file.name}")
        
        try:
            # Load data. Check for Parquet first.
            if input_parquet_file.exists():
                logger.info(f"Loading data from {input_parquet_file.name}")
                self.df = pd.read_parquet(input_parquet_file)
            else:
                logger.info(f"Loading data from {input_csv_file.name}")
                self.df = pd.read_csv(input_csv_file, parse_dates=["timestamp"])
                
        except Exception as e:
            logger.error(f"Failed to load data from {input_csv_file}: {e}")
            raise
        
        logger.info("Starting feature engineering steps...")
        
        # Apply all message-based feature engineering steps
        self.df = self._add_timestamp_features(self.df)
        self.df = self._add_word_count(self.df)
        self.df = self._add_char_count(self.df)
        self.df = self._add_time_differences(self.df)
        self.df = self._add_drink_feature(self.df)
        self.df = self._is_question(self.df)
        self.df = self._meet_up_feature(self.df)
        self.df = self._flag_image_messages(self.df)
        self.df = self._flag_empty_messages(self.df)
        self.df = self._flag_removed_messages(self.df)
        self.df = self._emoji_features(self.df)
        self.df = self._add_sentiment_features(self.df) 
        
        logger.info("Feature engineering steps complete.")
        
        # Save the final data using the new helper method
        return self._save_dataframe(self.df, filename_base="whatsapp")

# --- Main Execution Block ---

# **Helper function to load config (copied from clean_data.py)**
def _load_config() -> CleanConfig:
    """Loads configuration from config.toml and returns a CleanConfig object."""
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # Assume 'raw', 'preprocessed', 'cleaned', 'feature_added' are in config.toml
    raw = Path(config["raw"])
    preprocessed = Path(config["preprocessed"])
    cleaned = Path(config["cleaned"]) 
    feature_added = Path(config["feature_added"]) 
    datafile = Path(config["input"])

    # Create the Folders object
    folders = Folders(
        raw=raw,
        preprocessed=preprocessed,
        cleaned=cleaned, 
        feature_added=feature_added,
        datafile=datafile,
    )
    
    # Create the CleanConfig object (assuming it holds the Folders object)
    clean_config = CleanConfig(
        folders=folders,
    )
    return clean_config

# --- NEW PUBLIC FUNCTION (Same pattern as run_cleaning) ---
def run_feature_engineering() -> Path:
    """
    Public entry point for the feature engineering process to be called from other modules.
    
    :return: Path to the final feature-engineered CSV file.
    """
    try:
        config = _load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Use return instead of sys.exit(1) for cleaner external calls
        raise RuntimeError("Failed to load feature engineering configuration.")

    logger.info(f"Input path assumed from cleaned folder: {config.folders.cleaned}")
    
    # Run the feature engineer
    engineer = FeatureEngineer(config=config)
    # Return the path of the saved file
    return engineer.run()

@click.command()
def main():
    """Main entry point for the feature engineering process (CLI use)."""
    # Simply call the new run_feature_engineering function
    run_feature_engineering()

if __name__ == "__main__":
    main()