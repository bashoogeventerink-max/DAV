# import packages
import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import tomllib
from loguru import logger
import pytz
from wa_analyzer.humanhasher import humanize


def get_data_path(config_path: Path) -> Path:
    """Reads the configuration file and returns the path to the data file."""
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    processed_path = Path("../data/processed").resolve()
    data_path = processed_path / config["inputpath"]
    if not data_path.exists():
        logger.warning(
            f"{data_path} does not exist. Maybe first run src/preprocess.py, or check the timestamp!"
        )
    return data_path


def clean_author_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans author names by removing leading tilde characters."""
    clean_tilde = r"^~\u202f"
    df["author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", x))
    return df


def anonymize_authors(df: pd.DataFrame, processed_path: Path) -> pd.DataFrame:
    """Anonymizes author names and saves a reference file."""
    authors = df.author.unique()
    anon = {k: humanize(k) for k in authors}
    
    reference_file = processed_path / "anon_reference.json"
    
    with open(reference_file, "w") as f:
        ref = {v: k for k, v in anon.items()}
        ref_sorted = {k: ref[k] for k in sorted(ref.keys())}
        json.dump(ref_sorted, f)
        
    if not len(anon) == len(authors):
        raise ValueError("Some authors were lost during anonymization.")
        
    df["anon_author"] = df["author"].map(anon)
    df = df.drop(columns=["author"])
    df.rename(columns={"anon_author": "author"}, inplace=True)
    return df


def find_emojis(df: pd.DataFrame) -> pd.DataFrame:
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

    df["has_emoji"] = df["message"].apply(has_emoji)
    return df


def main():
    """Main function to run the data cleaning process."""
    config_path = Path("../config.toml").resolve()
    processed_path = Path("../data/processed").resolve()

    # Load data
    data_path = get_data_path(config_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    
    # Clean and transform data
    df = clean_author_names(df)
    df = anonymize_authors(df, processed_path)
    df = df.drop(index=[0])
    df = find_emojis(df)
    
    # Save the cleaned data
    now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
    output_path = processed_path / f"whatsapp-{now}.csv"
    
    df.to_csv(output_path, index=False)
    df.to_parquet(output_path.with_suffix(".parq"), index=False)
    logger.info(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()