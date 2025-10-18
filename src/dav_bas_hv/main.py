# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from test import test_function
from preprocess import main as preprocess_main 
from clean_data_v2 import DataCleaner
# from clean_data_v2 import 

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
    data_folder_str = Path("data/processed").resolve()

    #convert data folder string to a path object
    data_folder = Path(data_folder_str)
    processed_filepath = data_folder / preprocessed_filename

    if processed_filepath.exists():
        print(f"Preprocessed file '{preprocessed_filename} already exists at '{processed_filepath}'. Skipping preprocessing.")
    else:
        print(f"Preprocessed file '{preprocessed_filename}' not found. Running preprocessing...")
    
        preprocess_main()

        print("Preprocessing completed.")
    
    test_function()

if __name__ == '__main__':
    main()