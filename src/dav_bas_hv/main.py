# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from preprocess import main as preprocess_main 
from clean_data_v2 import run_data_cleaning as clean_data_main
from add_features_v2 import run_feature_engineering as feature_engineering_main

# Load toml configuration
def load_config(config_path="config.toml"):
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        return None

def main():
    
    # Check if data already exists
    config = load_config()
    if not config:
        return
    
    # ---- Set paths with Pathlib ----

    preprocessed_filename = config["preprocess_csv"]
    cleaned_filename = config["cleaned_csv"]
    feature_engineered_filename = config.get("feature_engineered_csv", "feature_engineered.csv")
    data_folder_str = Path("data/processed").resolve()

    #convert data folder string to a path object
    data_folder = Path(data_folder_str)
    preprocessed_filepath = data_folder / preprocessed_filename
    cleaned_filepath = data_folder / cleaned_filename
    feature_engineered_filepath = data_folder / feature_engineered_filename

    # ---- Preprocessing (Creates preprocess csv) ----

    if preprocessed_filepath.exists():
        print(f"Preprocessed file '{preprocessed_filename} already exists at '{preprocessed_filepath}'. Skipping preprocessing.")
    else:
        print(f"Preprocessed file '{preprocessed_filename}' not found. Running preprocessing...")
        preprocess_main()
        print("Preprocessing completed.")
    
    # ---- Cleaning (Creates cleaned csv) ----

    if cleaned_filepath.exists():
        print(f"Cleaned file '{cleaned_filename}' already exists at '{cleaned_filepath}'. Skipping cleaning.")
    else:
        print(f"Cleaned file '{cleaned_filename}' not found. Running cleaning...")
        cleaned_data_path = clean_data_main(
            input_path=preprocessed_filepath,
            output_path=cleaned_filepath,
            config=config
        )

        print("Cleaning completed. Cleaned data saved to: {cleaned_data_path}")

    # ----- Add Features -----

    if feature_engineered_filepath.exists():
        print(f"Feature engineered file '{feature_engineered_filename}' already exists at '{feature_engineered_filepath}'. Skipping feature engineering.")
    else:
        print(f"Feature engineered file '{feature_engineered_filename}' not found. Running feature engineering...")
        feature_engineered_data_path = feature_engineering_main(
            input_path=cleaned_filepath,
            output_path=feature_engineered_filepath,
            config=config
        )
        print(f"Feature engineering completed. Final data saved to: {feature_engineered_data_path}")

if __name__ == '__main__':
    main()