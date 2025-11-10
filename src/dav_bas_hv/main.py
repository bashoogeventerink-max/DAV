# Modules
import tomllib
from pathlib import Path

# Importing necessary functions from other modules
from preprocess import main as preprocess_main 
from clean_data_v2 import run_data_cleaning as clean_data_main
from add_features_v2 import run_feature_engineering as feature_engineering_main

# Importing analysis functions
from categories_graph import run_categories_analysis as categories_analysis_main
from distribution_graph import run_distribution_analysis as distribution_analysis_main
from correlation_graph import run_correlation_analysis as correlation_analysis_main

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
    directories_config = config["directories"]
    preprocessed_filename = directories_config["preprocess_csv"]
    cleaned_filename = directories_config["cleaned_csv"]
    feature_engineered_filename = directories_config["feature_engineered_csv"]
    categories_plot_filename = directories_config["categories_plot_png"]
    distribution_plot_filename = directories_config["distribution_plot_png"]
    correlation_plot_filename = directories_config["correlation_plot_png"]


    # ---- Folder paths with Pathlib ----
    data_folder_str = Path("data/processed").resolve()
    img_folder_str = Path("img/final").resolve()

    #convert data folder string to a path object
    data_folder = Path(data_folder_str)
    img_folder = Path(img_folder_str)

    # ---- Full path names ----
    preprocessed_filepath = data_folder / preprocessed_filename
    cleaned_filepath = data_folder / cleaned_filename
    feature_engineered_filepath = data_folder / feature_engineered_filename
    categories_plot_filepath = img_folder / categories_plot_filename
    distribution_plot_filepath = img_folder / distribution_plot_filename
    correlation_plot_filepath = img_folder / correlation_plot_filename

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

    # ----- Categories Analysis -----
    if categories_plot_filepath.exists():
        print(f"Categories plot '{categories_plot_filename}' already exists at '{categories_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Categories plot '{categories_plot_filename}' not found. Running categories analysis...")
        categories_plot_path = categories_analysis_main(
            input_path=feature_engineered_filepath,
            output_path=categories_plot_filepath,
            config=config
        )
        print(f"Categories analysis completed. Plot saved to: {categories_plot_path}")

    # ----- Distribution Analysis -----

    if distribution_plot_filepath.exists():
        print(f"Distribution plot '{distribution_plot_filename}' already exists at '{distribution_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Distribution plot '{distribution_plot_filename}' not found. Running distribution analysis...")
        distribution_plot_path = distribution_analysis_main(
            input_path=feature_engineered_filepath,
            output_path=distribution_plot_filepath,
            config=config
        )
        print(f"Distribution analysis completed. Plot saved to: {distribution_plot_path}")

    # ----- Correlation Analysis -----
    if correlation_plot_filepath.exists():
        print(f"Correlation plot '{correlation_plot_filename}' already exists at '{correlation_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Correlation plot '{correlation_plot_filename}' not found. Running correlation analysis...")
        correlation_plot_path = correlation_analysis_main(
            input_path=feature_engineered_filepath,
            output_path=correlation_plot_filepath,
            config=config
        )
        print(f"Correlation analysis completed. Plot saved to: {correlation_plot_path}")

if __name__ == '__main__':
    main()