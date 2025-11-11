# import packages
import json
import re
import sys
import tomllib
from pathlib import Path
import pandas as pd
import numpy as np 
from datetime import datetime
from loguru import logger
import pytz
import click
import os

# Import from settings
from .settings import Folders, CleanConfig # CleanConfig will be defined below
from wa_analyzer.humanhasher import humanize 

# Configure Loguru (copied from preprocess.py)
logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# --- DataCleaner Class ---
class DataCleaner:
    """
    A class to handle core data cleaning and feature engineering steps on a DataFrame.
    """
    def __init__(self, config: CleanConfig): 
        """
        :param config: The loaded configuration object including Folders.
        """
        self.folders = config.folders # Use Folders object
        self.df = None
        
        # Ensure the directory for the cleaned folder exists (from self.folders)
        self.folders.cleaned.mkdir(parents=True, exist_ok=True)
        # We assume the parent of the cleaned folder is the correct place for the
        # anonymization reference file.
        self.folders.cleaned.parent.mkdir(parents=True, exist_ok=True) 

    # --- Feature Engineering Methods (Unchanged for brevity) ---
    
    def _clean_author_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans author names by removing leading tilde characters."""
        clean_tilde = r"^~\u202f"
        df.loc[:, "author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", x))
        return df
    
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
        
        # **CHANGE 2: Use self.folders.cleaned.parent for the reference file**
        reference_file = self.folders.cleaned.parent / "anon_reference.json"
        
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
        
        # Define output file paths
        outfile_csv = self.folders.cleaned / f"{filename_base}-{now}-cleaned.csv"
        outfile_parquet = self.folders.cleaned / f"{filename_base}-{now}-cleaned.parq"
        
        logger.info(f"Writing CSV to {outfile_csv}")
        df.to_csv(outfile_csv, index=False)
        
        logger.info(f"Writing Parquet to {outfile_parquet}")
        df.to_parquet(outfile_parquet, index=False)
        
        logger.success("Saving complete.")
        
        return outfile_csv

    def run(self) -> Path:
        """
        Runs the cleaning pipeline: loads data, cleans authors, adds features, 
        anonymizes, and saves the final output.
        
        :return: Path to the final cleaned CSV file.
        """
        input_files = list(self.folders.preprocessed.glob("*-preprocess.csv"))
        
        if not input_files:
            logger.error(f"No *-preprocess.csv files found in {self.folders.preprocessed}. Exiting.")
            raise FileNotFoundError(f"No preprocessed CSV files found in {self.folders.preprocessed}")
            
        # Find the file with the most recent modification time (mtime)
        input_file = max(input_files, key=os.path.getmtime)
        logger.info(f"Selected latest file for cleaning: {input_file.name}")
        
        try:
            # Load data, assuming timestamp is already a clean datetime column
            self.df = pd.read_csv(input_file, parse_dates=["timestamp"])
        except Exception as e:
            logger.error(f"Failed to load data from {input_file}: {e}")
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
        return self._save_dataframe(self.df, filename_base="whatsapp")

# --- Main Execution Block (New) ---

# Helper function to load config (similar to preprocess.py's main)
def _load_config() -> CleanConfig:
    """Loads configuration from config.toml and returns a CleanConfig object."""
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # Use the 'cleaned' folder path from the config for saving
    raw = Path(config["raw"])
    preprocessed = Path(config["preprocessed"])
    cleaned = Path(config["cleaned"]) # Assuming 'cleaned' is in config.toml
    datafile = Path(config["input"])

    # Create the Folders object
    folders = Folders(
        raw=raw,
        preprocessed=preprocessed,
        cleaned=cleaned, # Now included
        feature_added=Path("."), # Placeholder since it's not used here, but needed by Folders dataclass
        datafile=datafile,
    )
    
    # Create the CleanConfig object
    clean_config = CleanConfig(
        folders=folders,
        # No other specific config items needed for DataCleaner's init currently
    )
    return clean_config

# --- NEW PUBLIC FUNCTION ---
def run_cleaning() -> Path:
    """
    Public entry point for the data cleaning process to be called from other modules.
    
    :return: Path to the final cleaned CSV file.
    """
    try:
        config = _load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Use return instead of sys.exit(1) for cleaner external calls
        raise RuntimeError("Failed to load cleaning configuration.")

    logger.info(f"Input path assumed from preprocessed folder: {config.folders.preprocessed}")
    
    # Run the cleaner
    cleaner = DataCleaner(config=config)
    # Return the path of the saved file
    return cleaner.run()

@click.command()
def main():
    """Main entry point for the data cleaning process (CLI use)."""
    # Simply call the new run_cleaning function
    run_cleaning()

if __name__ == "__main__":
    main()