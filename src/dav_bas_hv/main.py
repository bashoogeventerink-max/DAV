# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from test import test_function
from preprocess import main as preprocess_main 

def load_config(config_path="config.toml"):
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        return None

def main():
    config = load_config()
    if not config:
        return
    
    processed_filename = config[""]
    
    
    
    
    test_function()
    preprocess_main()

if __name__ == '__main__':
    main()