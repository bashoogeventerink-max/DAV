# import packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
# Import the specific SVD implementation from scikit-learn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class SVDAnalyzer:
    """
    A class to perform Single Value Decomposition (SVD) on the 
    feature-engineered DataFrame for dimensionality reduction.
    """
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the feature-engineered CSV/Parquet file.
        :param output_path: Path where the generated plot image (.png) for 
                            explained variance will be saved.
        :param config: The loaded configuration dictionary.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        self.A = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_data(self):
        """
        Loads the feature-engineered data from the specified path.
        """
        logger.info(f"Loading data for SVD analysis from: {self.input_path.name}")
        try:
            # Attempt to load Parquet first, then fallback to CSV
            parquet_path = self.input_path.with_suffix(".parq")
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                # Assuming the primary input is a CSV file
                self.df = pd.read_csv(self.input_path) 
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
            
        # 1. Identify numerical columns for SVD
        # Exclude known non-feature columns (like identifiers or timestamps)
        exclude_cols = self.config.get('svd_exclude_cols', ['author_id', 'message_id', 'timestamp'])
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

        if not feature_cols:
            logger.error("No suitable numerical features found for SVD after exclusion.")
            raise ValueError("No numerical features available for SVD.")
            
        logger.info(f"SVD will use {len(feature_cols)} features.")
        
        # 2. Scale the data
        # It's crucial to standardize features before SVD/PCA
        logger.info("Standardizing feature matrix...")
        scaler = StandardScaler()
        self.A = scaler.fit_transform(self.df[feature_cols].fillna(0)) # Fill NaNs for safety
        
    def _run_svd(self):
        """
        Performs SVD and calculates the cumulative explained variance.
        """
        logger.info("    -> Performing Truncated SVD to calculate explained variance.")
        
        # Use a high number of components (or min(rows, cols) - 1) to capture all variance
        # We use TruncatedSVD as it's efficient for large, potentially sparse data
        n_components = min(self.A.shape) - 1
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(self.A)
        
        # Calculate the cumulative sum of explained variance ratio
        explained_variance_ratio_cumsum = np.cumsum(svd.explained_variance_ratio_)
        
        return explained_variance_ratio_cumsum

    def _generate_plot(self, explained_variance_cumsum: np.ndarray) -> plt.Figure:
        """
        Generates the plot showing cumulative explained variance vs. number of components.
        
        :param explained_variance_cumsum: Cumulative explained variance array.
        :return: The Matplotlib Figure object.
        """
        logger.info("    -> Generating explained variance plot for dimension selection.")
        
        num_components = len(explained_variance_cumsum)
        # Find the number of components needed to explain 90% of the variance
        k_90 = np.argmax(explained_variance_cumsum >= 0.90) + 1
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the cumulative explained variance
        ax.plot(range(1, num_components + 1), explained_variance_cumsum, marker='o', linestyle='-', color='purple')

        # Highlight the 90% cutoff
        ax.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
        ax.axvline(x=k_90, color='r', linestyle='--', alpha=0.6)
        
        # Annotation for the 90% cutoff point
        ax.text(k_90 + 1, 0.90, f'  k={k_90}', color='red', va='bottom')

        # Apply titles and labels
        ax.set_title('SVD: Cumulative Explained Variance', fontsize=16)
        ax.set_xlabel('Number of Latent Factors (k)', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        ax.grid(axis='both', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        
        # Set x-axis ticks to be integers
        max_tick = min(num_components, 20) # Only show up to 20 ticks for readability
        ax.set_xticks(range(1, max_tick + 1, max(1, max_tick // 10)))
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout() 
        
        return fig

    def run(self) -> Path:
        """
        Runs the SVD analysis pipeline: loads data, performs SVD, generates 
        plot, and saves the final image.
        
        :return: Path to the final saved plot image.
        """
        self._load_data()
        
        if self.A is None:
            logger.error("Feature matrix A was not created. Aborting SVD analysis.")
            return self.output_path

        logger.info("Starting SVD analysis and visualization...")
        
        # 1. Perform SVD
        explained_variance_cumsum = self._run_svd()
        
        # 2. Generate the plot
        fig = self._generate_plot(explained_variance_cumsum)
        
        # 3. Save the figure
        logger.info(f"Saving SVD explained variance plot to: {self.output_path.name}")
        fig.savefig(self.output_path, dpi=300)
        
        logger.info("SVD analysis complete.")
        # Close the plot to free up memory
        plt.close(fig) 
        return self.output_path

# Public function to be used in main.py

def run_svd_analysis(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the SVD analysis and visualization process.
    
    :param input_path: Path to the feature-engineered CSV/Parquet.
    :param output_path: Path to save the final explained variance plot image (.png).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = SVDAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()