# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from preprocess import main as preprocess_main 
from clean_data_v2 import run_data_cleaning as clean_data_main

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
    data_folder_str = Path("data/processed").resolve()

    #convert data folder string to a path object
    data_folder = Path(data_folder_str)
    preprocessed_filepath = data_folder / preprocessed_filename
    cleaned_filepath = data_folder / cleaned_filename


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

    clean_data_main()

if __name__ == '__main__':
    main()