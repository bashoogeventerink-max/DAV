import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from loguru import logger
from wa_analyzer.humanhasher import humanize
from pathlib import Path

class WADataCleaner:
    def __init__(self, df: pd.DataFrame):
        """Initializes the cleaner with a DataFrame."""
        self.df = df
        
        self.emoji_pattern = re.compile(
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

    def filter_data(self) -> 'WADataCleaner':
        """Filters out 'Unknown' authors and 'Wachten op dit bericht' messages."""
        self.df = self.df[self.df['author'] != 'Unknown']
        self.df = self.df[~self.df['message'].str.contains('Wachten op dit bericht', na=False)]
        logger.info("Filtered out 'Unknown' authors and waiting messages.")
        return self

    def clean_author_names(self) -> 'WADataCleaner':
        """Cleans author names by removing tilde characters."""
        clean_tilde = r"^~\u202f"
        self.df['author'] = self.df['author'].apply(lambda x: re.sub(clean_tilde, "", x))
        logger.info("Cleaned tilde from author names.")
        return self

    def add_city_label(self) -> 'WADataCleaner':
        """Adds a 'living_in_city' label based on specific authors."""
        city_authors = [
            'Bas hooge Venterink', 'Robert te Vaarwerk', 
            'Spiderman Spin', 'Thies Jan Weijmans', 'Smeerbeer van Dijk'
        ]
        self.df['living_in_city'] = np.where(self.df['author'].isin(city_authors), 1, 0)
        logger.info("Added 'living_in_city' label.")
        return self

    def anonymize_authors(self, output_dir: Path) -> 'WADataCleaner':
        """Anonymizes author names and saves a reference file."""
        authors = self.df.author.unique()
        anon = {k: humanize(k) for k in authors}
        
        if not len(anon) == len(authors):
            raise ValueError("Anonymization lost some authors!")

        reference_file = output_dir / "anon_reference.json"
        with open(reference_file, "w") as f:
            ref = {v: k for k, v in anon.items()}
            ref_sorted = {k: ref[k] for k in sorted(ref.keys())}
            json.dump(ref_sorted, f, indent=4)
        
        self.df["anon_author"] = self.df.author.map(anon)
        self.df.drop(columns=["author"], inplace=True)
        self.df.rename(columns={"anon_author": "author"}, inplace=True)
        logger.info(f"Anonymized authors and saved reference file to {reference_file}.")
        return self

    def add_emoji_feature(self) -> 'WADataCleaner':
        """Adds a boolean feature indicating if a message contains an emoji."""
        self.df["has_emoji"] = self.df["message"].apply(
            lambda text: bool(self.emoji_pattern.search(text))
        )
        logger.info("Added 'has_emoji' feature.")
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        """Returns the final cleaned DataFrame."""
        return self.df