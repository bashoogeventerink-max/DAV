# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from test import test_function
from preprocess import main as preprocess_main 

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
    
    processed_filename = config["inputpath"]
    data_folder_str = Path("data/processed").resolve()

    #convert data folder string to a path object
    data_folder = Path(data_folder_str)
    processed_filepath = data_folder / processed_filename

    if processed_filepath.exists():
        print(f"Processed file '{processed_filename} already exists at '{processed_filepath}'. Skipping preprocessing.")
    else:
        print(f"Processed file '{processed_filename}' not found. Running preprocessing...")
    
        preprocess_main()

        print("Preprocessing completed.")
    
    test_function()

if __name__ == '__main__':
    main()