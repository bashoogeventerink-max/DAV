# import packages
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
from loguru import logger
import pytz
import numpy as np
from textblob import TextBlob
from wa_analyzer.humanhasher import humanize

# -------------- Add features ----------------

def get_sentiment_polarity(text: str) -> float:
    """
    Calculates the sentiment polarity (-1.0 to 1.0) of a given text.
    Returns 0.0 (Neutral) for missing or non-string values.
    """
    if pd.isna(text):
        return 0.0 
    # TextBlob can sometimes struggle with non-string types, so we ensure conversion
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment features ('sentiment_polarity' and 'sentiment_category') 
    to the DataFrame based on the 'message' column.
    """
    # 1. Add numerical polarity score
    df['sentiment_polarity'] = df['message'].apply(get_sentiment_polarity)
    
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

def count_emojis(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a feature column 'emoji_count' based on message content."""
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

def add_living_in_city(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a binary column indicating if the author is living in a city."""
    city_authors = [
        "Bas hooge Venterink", 
        "Robert te Vaarwerk", 
        "Spiderman Spin", 
        "Thies Jan Weijmans", 
        "Smeerbeer van Dijk"
    ]
    df["living_in_city"] = np.where(df["author"].isin(city_authors), 1, 0)
    return df

def technical_background(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a binary column indicating if the author had a technical background in terms of studying."""
    tech_background = [
        "Weda", 
        "Robert te Vaarwerk", 
        "Schjöpschen", 
        "Smeerbeer van Dijk"
    ]
    df["tech_background"] = np.where(df["author"].isin(tech_background), 1, 0)
    return df

def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the number of words in each message."""
    df["word_count"] = df["message"].str.split().str.len()
    return df


def add_time_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time difference between consecutive messages in seconds,
    minutes, and hours.
    """
    df["time_diff"] = df["timestamp"].diff()
    df["react_time_sec"] = df["time_diff"].dt.total_seconds()
    df["react_time_min"] = df["react_time_sec"] / 60
    df["react_time_hr"] = df["react_time_sec"] / 3600
    df.drop(columns=["time_diff"], inplace=True)
    return df

def flag_image_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Flags messages that represent an image."""
    df['is_image'] = np.where(df['message'].str.contains('<Media weggelaten>', case=False, na=False), 1, 0)
    return df

def flag_empty_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Flags messages that contain 'Wachten op dit bericht'."""
    df['is_empty_message'] = np.where(df['message'].str.contains('Wachten op dit bericht', case=False, na=False), 1, 0)
    return df

def flag_removed_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Flags messages that have been removed by the user."""
    df['is_removed_message'] = np.where(df['message'].str.contains('Je hebt dit bericht verwijderd', case=False, na=False), 1, 0)
    return df

def main():
    """Main function to run the data cleaning process."""
    config_path = Path("config.toml").resolve()
    processed_path = Path("data/processed").resolve()

    # Load data
    data_path = get_data_path(config_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    
    # Clean and transform data
    df = df.drop(index=[0])
    df = clean_author_names(df)
    
    # Add features based on original author names
    df = add_living_in_city(df)
    df = technical_background(df)
    
    # Anonymize authors after features have been added
    df = anonymize_authors(df, processed_path)
    
    # Add other features
    df = find_emojis(df)
    df = count_emojis(df)
    df = add_word_count(df)
    df = add_time_differences(df)
    df = flag_image_messages(df)
    df = flag_empty_messages(df)
    df = flag_removed_messages(df)
    df = add_sentiment_features(df) # <-- New sentiment analysis feature added here
    
    # Save the cleaned data
    now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
    output_path = processed_path / f"whatsapp-{now}.csv"
    
    df.to_csv(output_path, index=False)
    df.to_parquet(output_path.with_suffix(".parq"), index=False)
    logger.info(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()