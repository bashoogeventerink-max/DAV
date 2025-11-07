import pandas as pd
from loguru import logger
import tomllib
from pathlib import Path
from datetime import datetime
import pytz

# Import the DataCleaner class from your cleaning module
from cleaning_data import WADataCleaner

def run_pipeline(df_input: pd.DataFrame, output_dir: Path):
    """
    Executes the cleaning and feature engineering pipeline.
    """
    logger.info("Starting data cleaning pipeline...")
    
    try:
        # Instantiate the cleaner and chain the methods
        cleaner = WADataCleaner(df=df_input)
        
        cleaned_df = cleaner.filter_data() \
            .clean_author_names() \
            .add_city_label() \
            .anonymize_authors(output_dir=output_dir) \
            .add_emoji_feature() \
            .get_cleaned_data()
        
        logger.info("Pipeline completed successfully.")
        
        # Display the resulting DataFrame
        print("\n--- Final Cleaned and Processed DataFrame ---")
        print(cleaned_df.head())
        print(f"\nFinal shape: {cleaned_df.shape}")
        
        # Optional: Save the final output to CSV
        cleaned_csv_path = output_dir / "cleaned_chat_data.csv"
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        logger.info(f"Cleaned data saved to {cleaned_csv_path.resolve()}")
        
        # Save the final output to PARQUET
        cleaned_parquet_path = output_dir / "cleaned_chat_data.parquet"
        # Note: You may need to install 'pyarrow' or 'fastparquet' for pandas to_parquet to work.
        cleaned_df.to_parquet(cleaned_parquet_path, index=False)
        logger.info(f"Cleaned data also saved to {cleaned_parquet_path.resolve()} in Parquet format.")
        
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")

if __name__ == "__main__":
    
    # 1. Define the output directory
    OUTPUT_DIR = Path("./output_data")
    
    # Clean up previous output directory (optional, for fresh runs)
    if OUTPUT_DIR.exists():
        logger.info(f"Removing old output directory: {OUTPUT_DIR}")
        import shutil
        try:
            shutil.rmtree(OUTPUT_DIR)
        except OSError as e:
            logger.error(f"Error removing directory: {e}")

    # 2. Prepare the input data
    raw_df = create_dummy_data()
    
    # 3. Run the data processing pipeline
    run_pipeline(df_input=raw_df, output_dir=OUTPUT_DIR)