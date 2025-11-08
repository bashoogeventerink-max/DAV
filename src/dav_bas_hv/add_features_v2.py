# import packages
import re
from pathlib import Path
import pandas as pd
import numpy as np 
from loguru import logger
from textblob import TextBlob
from typing import Union

class FeatureEngineer:
    """
    A class to handle feature engineering steps on a cleaned DataFrame.
    It takes the output of the cleaning step and adds new analytical features.
    """
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the cleaned CSV file.
        :param output_path: Path where the final feature-engineered CSV/Parquet will be saved.
        :param config: The loaded configuration dictionary (currently unused but kept for consistency).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
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

    def _find_emojis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a feature column 'has_emoji' based on message content."""
        logger.info("    -> Adding 'has_emoji' feature.")
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

        df["has_emoji"] = df["message"].apply(has_emoji).astype(int)
        return df

    def _count_emojis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a feature column 'emoji_count' based on message content."""
        logger.info("    -> Adding 'emoji_count' feature.")
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

        def get_emoji_count(text):
            return len(emoji_pattern.findall(str(text)))

        df["emoji_count"] = df["message"].apply(get_emoji_count)
        return df

    def _add_word_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the number of words in each message."""
        logger.info("    -> Adding 'word_count' feature.")
        # Ensure message is string and handle NaN gracefully
        df["word_count"] = df["message"].astype(str).str.split().str.len()
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


    def run(self) -> Path:
        """
        Runs the feature engineering pipeline: loads cleaned data, adds features, 
        and saves the final output.
        
        :return: Path to the final feature-engineered CSV file.
        """
        logger.info(f"Loading cleaned data from: {self.input_path.name}")
        
        try:
            # Load data, assuming timestamp is already a clean datetime column
            # Use 'parquet' if available, otherwise fallback to 'csv'
            if self.input_path.with_suffix(".parq").exists():
                self.df = pd.read_parquet(self.input_path.with_suffix(".parq"))
            else:
                self.df = pd.read_csv(self.input_path, parse_dates=["timestamp"])
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
        
        logger.info("Starting feature engineering steps...")
        
        # Apply all message-based feature engineering steps
        self.df = self._add_word_count(self.df)
        self.df = self._add_time_differences(self.df)
        self.df = self._flag_image_messages(self.df)
        self.df = self._flag_empty_messages(self.df)
        self.df = self._flag_removed_messages(self.df)
        self.df = self._find_emojis(self.df)
        self.df = self._count_emojis(self.df)
        self.df = self._add_sentiment_features(self.df) 
        
        logger.info("Feature engineering steps complete.")
        
        # Save the final data
        logger.info(f"Saving final data to: {self.output_path.name} and Parquet")
        
        self.df.to_csv(self.output_path, index=False)
        self.df.to_parquet(self.output_path.with_suffix(".parq"), index=False)
        
        logger.info(f"Feature engineering complete. Saved to {self.output_path.name}")
        return self.output_path

# Public function to be used in main.py

def run_feature_engineering(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the feature engineering process.
    
    :param input_path: Path to the cleaned CSV (output of data cleaning).
    :param output_path: Path to save the final feature-engineered CSV/Parquet.
    :param config: The loaded application configuration.
    :return: Path to the final feature-engineered CSV file.
    """
    engineer = FeatureEngineer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return engineer.run()