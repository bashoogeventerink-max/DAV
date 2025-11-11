# import packages
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np 
from datetime import datetime
from loguru import logger
import pytz

from wa_analyzer.humanhasher import humanize 

class DataCleaner:
    """
    A class to handle core data cleaning and feature engineering steps on a DataFrame.
    """
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the preprocessed (uncleaned) CSV file.
        :param output_path: Path where the final cleaned CSV/Parquet will be saved.
        :param config: The loaded configuration dictionary.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        
        # Ensure the directory for the anon reference file exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _clean_author_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans author names by removing leading tilde characters."""
        clean_tilde = r"^~\u202f"
        df.loc[:, "author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", x))
        return df

    # The following two columns are added based on the name of the author in a unique dataset of the maker of the project. 
    # These features are not generalizable to other datasets.
    
    def _add_living_in_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a binary column indicating if the author is living in a city."""
        logger.info("    -> Adding 'living_in_city' feature.")
        city_authors = [
            "Bas hooge Venterink", 
            "Robert te Vaarwerk", 
            "Spiderman Spin", 
            "Thies Jan Weijmans", 
            "Smeerbeer van Dijk"
        ]
        df.loc[:, "living_in_city"] = np.where(df["author"].isin(city_authors), 1, 0)
        return df

    def _technical_background(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a binary column indicating if the author had a technical background in terms of studying."""
        logger.info("    -> Adding 'tech_background' feature.")
        tech_background = [
            "Weda", 
            "Robert te Vaarwerk", 
            "Schjöpschen", 
            "Smeerbeer van Dijk"
        ]
        df.loc[:, "tech_background"] = np.where(df["author"].isin(tech_background), 1, 0)
        return df
    
    def _living_with_partner(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a binary column indicating if the author is living with a partner."""
        logger.info("    -> Adding 'living_with_partner' feature.")
        partner_authors = [
            "Bas hooge Venterink", 
            "Thies Jan Weijmans",
            "Smeerbeer van Dijk",
            "Thomas Grundel",
            "Jop van der Woning"
        ]

        df.loc[:, "living_with_partner"] = np.where(df["author"].isin(partner_authors), 1, 0)
        return df
    
    def _date_living_with_partner(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a datetime column indicating since when the author is living with a partner."""
        logger.info("    -> Adding 'date_living_with_partner' feature.")
        partner_dates = {
            "Bas hooge Venterink": "2024-12-01",
            "Thies Jan Weijmans": "2024-09-01",
            "Smeerbeer van Dijk": "2023-06-01",
            "Thomas Grundel": "2024-11-01",
            "Jop van der Woning": "2025-03-10"
        }

        def get_date(author):
            date_str = partner_dates.get(author, None)
            if date_str:
                return pd.to_datetime(date_str).tz_localize(pytz.UTC)
            else:
                return pd.NaT

        df.loc[:, "date_living_with_partner"] = df["author"].apply(get_date)
        return df

    def _anonymize_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymizes author names and saves a reference file."""
        logger.info("    -> Anonymizing authors.")
        authors = df.author.unique()
        anon = {k: humanize(k) for k in authors}
        
        reference_file = self.output_path.parent / "anon_reference.json"
        
        with open(reference_file, "w") as f:
            # Create reference mapping: Anonymized Name -> Original Name
            ref = {v: k for k, v in anon.items()}
            ref_sorted = {k: ref[k] for k in sorted(ref.keys())}
            json.dump(ref_sorted, f, indent=4)
        
        if not len(anon) == len(authors):
            logger.error("Author count mismatch during anonymization.")
            raise ValueError("Some authors were lost during anonymization.")
            
        df.loc[:, "anon_author"] = df["author"].map(anon)
        df = df.drop(columns=["author"])
        df.rename(columns={"anon_author": "author"}, inplace=True)
        return df

    def _save_dataframe(self, df: pd.DataFrame, filename_base: str) -> Path:
        """
        Saves the DataFrame to CSV and Parquet with a timestamped filename.

        :param df: The pandas DataFrame to save.
        :param filename_base: The base name for the file (e.g., "whatsapp-cleaned").
        :return: Path to the final cleaned CSV file.
        """
        # Generate the timestamp
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Construct the output file paths using the current output_path directory
        # We replace the stem of the original output_path with our new timestamped name
        # and keep the directory.
        outfile_csv = self.output_path.parent / f"{filename_base}-{now}.csv"
        outfile_parquet = self.output_path.parent / f"{filename_base}-{now}.parq"
        
        logger.info(f"Writing CSV to {outfile_csv}")
        df.to_csv(outfile_csv, index=False)
        
        logger.info(f"Writing Parquet to {outfile_parquet}")
        df.to_parquet(outfile_parquet, index=False)
        
        logger.success("Saving complete.")
        
        # The function must return a Path object, which is the CSV file path
        return outfile_csv

    def run(self) -> Path:
        """
        Runs the cleaning pipeline: loads data, cleans authors, adds features, 
        anonymizes, and saves the final output.
        
        :return: Path to the final cleaned CSV file.
        """
        logger.info(f"Loading data from: {self.input_path.name}")
        
        try:
            # Load data, assuming timestamp is already a clean datetime column
            self.df = pd.read_csv(self.input_path, parse_dates=["timestamp"])
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
        
        # Drop the first row and reset index as per original logic
        self.df = self.df.drop(index=[0]).reset_index(drop=True)
        
        logger.info("Starting cleaning and feature engineering steps...")
        
        # 1. Clean author names (Prerequisite for features)
        self.df = self._clean_author_names(self.df)
        
        # 2. Add features that depend on original author names
        self.df = self._add_living_in_city(self.df)
        self.df = self._living_with_partner(self.df)
        self.df = self._date_living_with_partner(self.df)
        self.df = self._technical_background(self.df)
        
        # 3. Anonymize authors (Last author-dependent step)
        self.df = self._anonymize_authors(self.df)
        
        logger.info("Cleaning steps complete.")
        
        # Save the cleaned data using the new helper method
        return self._save_dataframe(self.df, filename_base="whatsapp-cleaned")

# Public function to be used in main.py
def run_data_cleaning(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the data cleaning process.
    
    :param input_path: Path to the uncleaned CSV (output of preprocessing).
    :param output_path: Path to save the final cleaned CSV/Parquet.
    :param config: The loaded application configuration.
    :return: Path to the final cleaned CSV file.
    """
    cleaner = DataCleaner(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return cleaner.run()