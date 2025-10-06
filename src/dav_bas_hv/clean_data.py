# import packages
import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import tomllib
from loguru import logger
import pytz
import numpy as np
from wa_analyzer.humanhasher import humanize

class WhatsAppCleaner:
    """
    A class to encapsulate all data cleaning and feature engineering steps
    for WhatsApp chat data.
    """

    def __init__(self, config_path: Path):
        """Initializes the cleaner with the configuration path."""
        self.config_path = config_path
        self.df = None
        self.processed_path = Path("data/processed").resolve()

    def _get_data_path(self) -> Path:
        """Reads the configuration file and returns the path to the data file."""
        with self.config_path.open("rb") as f:
            config = tomllib.load(f)
        data_path = self.processed_path / config["inputpath"]
        if not data_path.exists():
            logger.warning(
                f"{data_path} does not exist. Maybe first run src/preprocess.py, or check the timestamp!"
            )
        return data_path

    def _load_data(self):
        """Loads the raw data into a pandas DataFrame."""
        data_path = self._get_data_path()
        self.df = pd.read_csv(data_path, parse_dates=["timestamp"])
        logger.info(f"Data loaded from {data_path}")

    def _clean_author_names(self):
        """Cleans author names by removing leading tilde characters."""
        clean_tilde = r"^~\u202f"
        self.df["author"] = self.df["author"].apply(lambda x: re.sub(clean_tilde, "", x))
        logger.info("Author names cleaned.")

    def _anonymize_authors(self):
        """Anonymizes author names and saves a reference file."""
        authors = self.df.author.unique()
        anon = {k: humanize(k) for k in authors}
        
        reference_file = self.processed_path / "anon_reference.json"
        with open(reference_file, "w") as f:
            ref = {v: k for k, v in anon.items()}
            ref_sorted = {k: ref[k] for k, v in anon.items()}
            json.dump(ref_sorted, f, indent=4)
        
        if not len(anon) == len(authors):
            raise ValueError("Some authors were lost during anonymization.")
            
        self.df["anon_author"] = self.df["author"].map(anon)
        self.df = self.df.drop(columns=["author"])
        self.df.rename(columns={"anon_author": "author"}, inplace=True)
        logger.info("Authors anonymized.")

    def _find_emojis(self):
        """Adds a feature column 'has_emoji' based on message content."""
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

        self.df["has_emoji"] = self.df["message"].apply(has_emoji)
        logger.info("Emoji feature added.")

    def _add_living_in_city(self):
        """Adds a binary column indicating if the author is living in a city."""
        city_authors = [
            "Bas hooge Venterink", 
            "Robert te Vaarwerk", 
            "Spiderman Spin", 
            "Thies Jan Weijmans", 
            "Smeerbeer van Dijk"
        ]
        self.df["living_in_city"] = np.where(self.df["author"].isin(city_authors), 1, 0)
        logger.info("City author feature added.")

    def _add_word_count(self):
        """Calculates the number of words in each message."""
        self.df["word_count"] = self.df["message"].str.split().str.len()
        logger.info("Word count added.")

    def _add_time_differences(self):
        """Calculates the time difference between consecutive messages."""
        self.df["time_diff"] = self.df["timestamp"].diff()
        self.df["react_time_sec"] = self.df["time_diff"].dt.total_seconds()
        self.df["react_time_min"] = self.df["react_time_sec"] / 60
        self.df["react_time_hr"] = self.df["react_time_sec"] / 3600
        self.df.drop(columns=["time_diff"], inplace=True)
        logger.info("Time differences calculated.")

    def _flag_image_messages(self):
        """Flags messages that represent an image."""
        self.df['is_image'] = np.where(self.df['message'].str.contains('<Media weggelaten>', case=False, na=False), 1, 0)
        logger.info("Image messages flagged.")

    def _flag_empty_messages(self):
        """Flags messages that contain 'Wachten op dit bericht'."""
        self.df['is_empty_message'] = np.where(self.df['message'].str.contains('Wachten op dit bericht', case=False, na=False), 1, 0)
        logger.info("Empty messages flagged.")

    def _flag_removed_messages(self):
        """Flags messages that have been removed by the user."""
        self.df['is_removed_message'] = np.where(self.df['message'].str.contains('Je hebt dit bericht verwijderd', case=False, na=False), 1, 0)
        logger.info("Removed messages flagged.")

    def run(self):
        """Executes the complete data cleaning and saving pipeline."""
        self._load_data()
        self.df = self.df.drop(index=[0]) # Drop first row after loading
        
        self._clean_author_names()
        self._anonymize_authors()
        
        # Feature engineering steps
        self._find_emojis()
        self._add_living_in_city()
        self._add_word_count()
        self._add_time_differences()
        
        # Flagging specific messages
        self._flag_image_messages()
        self._flag_empty_messages()
        self._flag_removed_messages()

        # Save the cleaned data
        now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        output_path_csv = self.processed_path / f"whatsapp-{now}.csv"
        output_path_parq = self.processed_path / f"whatsapp-{now}.parq"
        
        self.df.to_csv(output_path_csv, index=False)
        self.df.to_parquet(output_path_parq, index=False)
        logger.info(f"Cleaned data saved to {output_path_csv} and {output_path_parq}")

def main():
    """Main function to run the data cleaning process using the class."""
    config_path = Path("config.toml").resolve()
    cleaner = WhatsAppCleaner(config_path)
    cleaner.run()

if __name__ == "__main__":
    main()