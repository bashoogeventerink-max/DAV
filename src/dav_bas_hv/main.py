# Modules
import tomllib
from pathlib import Path
import os

# Importing necessary functions from other modules
from data_handling.preprocess import run_preprocess as preprocess_main 
from data_handling.clean_data import run_cleaning as clean_data_main
from data_handling.add_features import run_feature_engineering as feature_engineering_main

# Importing analysis functions
from graphs.time_series_graph import run_dual_axis_analysis as time_series_main
from graphs.categories_graph import run_categories_analysis as categories_analysis_main
from graphs.distribution_graph import run_distribution_analysis as distribution_analysis_main
from graphs.correlation_graph import run_correlation_analysis as correlation_analysis_main
from graphs.dimensionality_graph import run_svd_analysis as svd_analysis_main

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
    feature_engineered_filename = config["feature_engineered_csv"]
    time_series_plot_filename = config["time_series_plot_png"]
    categories_plot_filename = config["categories_plot_png"]
    distribution_plot_filename = config["distribution_plot_png"]
    correlation_plot_filename = config["correlation_plot_png"]
    dimensionality_plot_filename = config["dimensionality_plot_png"]

    # ---- Folder paths with Pathlib ----
    data_preprocess_folder_str = Path("data/preprocessed").resolve()
    data_cleaned_folder_str = Path("data/cleaned").resolve()
    data_feature_folder_str = Path("data/feature_added").resolve()
    img_folder_str = Path("img/final").resolve()

    #convert data folder string to a path object
    data_folder_preprocess = Path(data_preprocess_folder_str)
    data_folder_cleaned = Path(data_cleaned_folder_str)
    data_folder_feature = Path(data_feature_folder_str)
    img_folder = Path(img_folder_str)

    # ---- Full path names ----
    preprocessed_filepath = data_folder_preprocess / preprocessed_filename
    cleaned_filepath = data_folder_cleaned / cleaned_filename
    feature_engineered_filepath = data_folder_feature / feature_engineered_filename
    time_series_plot_filepath = img_folder / time_series_plot_filename
    categories_plot_filepath = img_folder / categories_plot_filename
    distribution_plot_filepath = img_folder / distribution_plot_filename
    correlation_plot_filepath = img_folder / correlation_plot_filename
    dimensionality_plot_filepath = img_folder / dimensionality_plot_filename

    # ---- Preprocessing (Creates preprocess csv) ----
    
    if preprocessed_filepath.exists():
        print(f"Preprocessed file '{preprocessed_filename} already exists at '{preprocessed_filepath}'. Skipping preprocessing.")
    else:
        print(f"Preprocessed file '{preprocessed_filename}' not found. Running preprocessing...")
        output_filepath = preprocess_main(device="android")
        print("Preprocessing completed.")
    
    # ---- Cleaning (Creates cleaned csv) ----

    if cleaned_filepath.exists():
        print(f"Cleaned file '{cleaned_filename}' already exists at '{cleaned_filepath}'. Skipping cleaning.")
    else:
        print(f"Cleaned file '{cleaned_filename}' not found. Running cleaning...")
        cleaned_data_path = clean_data_main()
        print("Cleaning completed. Cleaned data saved to: {cleaned_data_path}")

    # ----- Add Features -----

    if feature_engineered_filepath.exists():
        print(f"Feature engineered file '{feature_engineered_filename}' already exists at '{feature_engineered_filepath}'. Skipping feature engineering.")
    else:
        print(f"Feature engineered file '{feature_engineered_filename}' not found. Running feature engineering...")
        feature_engineered_data_path = feature_engineering_main()
        
        print(f"Feature engineering completed. Final data saved to: {feature_engineered_data_path}")

    # ----- Time Series Analysis -----

    if time_series_plot_filepath.exists():
        print(f"Time series plot '{time_series_plot_filename}' already exists at '{time_series_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Time series plot '{time_series_plot_filename}' not found. Running time series analysis...")
        time_series_plot_path = time_series_main(
            output_filename=time_series_plot_filename
        )
        print(f"Time series analysis completed. Plot saved to: {time_series_plot_path}")

    # ----- Categories Analysis -----
    if categories_plot_filepath.exists():
        print(f"Categories plot '{categories_plot_filename}' already exists at '{categories_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Categories plot '{categories_plot_filename}' not found. Running categories analysis...")
        categories_plot_path = categories_analysis_main(
            output_filename=categories_plot_filename
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
            output_filename=correlation_plot_filename
        )
        print(f"Correlation analysis completed. Plot saved to: {correlation_plot_path}")
    
    # ----- Dimensionality Analysis -----
    if dimensionality_plot_filepath.exists():
        print(f"Dimensionality plot '{dimensionality_plot_filename}' already exists at '{dimensionality_plot_filepath}'. Skipping making graph.")
    else:
        print(f"Dimensionality plot '{dimensionality_plot_filename}' not found. Running dimensionality analysis...")
        dimensionality_plot_path = svd_analysis_main(
            input_path=feature_engineered_filepath,
            output_path=dimensionality_plot_filepath,
            config=config
        )
        print(f"Dimensionality analysis completed. Plot saved to: {dimensionality_plot_path}")

if __name__ == '__main__':
    main()