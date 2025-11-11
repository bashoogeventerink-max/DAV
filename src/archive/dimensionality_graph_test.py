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
    feature-engineered DataFrame for dimensionality reduction and factor interpretation.
    """
    def __init__(self, input_path: Path, output_path: Path, config: dict):
        """
        :param input_path: Path to the feature-engineered CSV/Parquet file.
        :param output_path: Base Path for output. The plot is saved as a PNG, 
                            and the factor data is saved as a Parquet file.
        :param config: The loaded configuration dictionary.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.df = None
        self.A = None
        self.feature_cols = None
        
        # Ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_data(self):
        """
        Loads the feature-engineered data from the specified path and standardizes features.
        """
        logger.info(f"Loading data for SVD analysis from: {self.input_path.name}")
        try:
            # Attempt to load Parquet first, then fallback to CSV
            parquet_path = self.input_path.with_suffix(".parq")
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                self.df = pd.read_csv(self.input_path) 
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.input_path}: {e}")
            raise
            
        # 1. Identify numerical columns for SVD
        exclude_cols = self.config.get('svd_exclude_cols', ['author_id', 'message_id', 'timestamp'])
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        self.feature_cols = [col for col in numerical_cols if col not in exclude_cols]

        if not self.feature_cols:
            logger.error("No suitable numerical features found for SVD after exclusion.")
            raise ValueError("No numerical features available for SVD.")
            
        logger.info(f"SVD will use {len(self.feature_cols)} features.")
        
        # 2. Scale the data
        logger.info("Standardizing feature matrix...")
        scaler = StandardScaler()
        # Use a copy of the original feature matrix for safe scaling
        self.A = scaler.fit_transform(self.df[self.feature_cols].fillna(0)) 
        
    def _run_svd(self, n_components: int = None):
        """
        Performs SVD and returns the model. If n_components is None, 
        it calculates the full explained variance.
        """
        n_comp = n_components if n_components else min(self.A.shape) - 1
        
        logger.info(f"    -> Performing Truncated SVD with {n_comp} components.")
        
        svd = TruncatedSVD(n_components=n_comp)
        svd.fit(self.A)
        
        return svd

    def _generate_plot(self, svd_model: TruncatedSVD) -> Path:
        """
        Generates and saves the plot showing cumulative explained variance.
        
        :param svd_model: The fitted TruncatedSVD model.
        :return: Path to the final saved plot image.
        """
        explained_variance_cumsum = np.cumsum(svd_model.explained_variance_ratio_)
        num_components = len(explained_variance_cumsum)
        plot_path = self.output_path.with_suffix(".png")

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
        
        max_tick = min(num_components, 20) 
        ax.set_xticks(range(1, max_tick + 1, max(1, max_tick // 10)))
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout() 
        
        logger.info(f"Saving SVD explained variance plot to: {plot_path.name}")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig) 
        return plot_path
        
    def _perform_reduction_and_interpret(self, k: int = 15) -> pd.DataFrame:
        """
        Performs the final SVD reduction with a fixed k and creates the 
        factor interpretation matrix (V_T) and the reduced data matrix (U * Sigma).
        
        :param k: The number of components to keep (based on plot analysis).
        :return: The full DataFrame with factor scores appended.
        """
        logger.info(f"--- Finalizing SVD Reduction with selected k={k} ---")
        
        # 1. Rerun SVD with the selected k=15
        svd_model = self._run_svd(n_components=k)
        
        # A_reduced: U * Sigma (the row scores for the new latent factors)
        A_reduced = svd_model.transform(self.A)
        
        # Vt: The V_transpose matrix (the factor loadings)
        Vt = svd_model.components_
        
        # 2. Create the Interpretation DataFrame (V_T)
        factor_names = [f'Factor_{i+1}' for i in range(k)]
        
        # V_T: Features as rows, Factors as columns (easier to read)
        factor_loadings_df = pd.DataFrame(
            Vt.T, # Transpose Vt to get V, then transpose again for the desired layout (Features x Factors)
            index=self.feature_cols,
            columns=factor_names
        )
        
        # Save the interpretation matrix for external analysis
        interpret_path = self.output_path.with_name(f"{self.output_path.stem}_loadings.parq")
        factor_loadings_df.to_parquet(interpret_path)
        logger.info(f"Factor Loadings (V_T) saved to: {interpret_path.name}")

        # 3. Create the Reduced Data DataFrame (U * Sigma)
        # Append the factor scores (A_reduced) to the original DataFrame
        factor_scores_df = pd.DataFrame(A_reduced, columns=factor_names, index=self.df.index)
        
        # Remove the original high-dimensional features (optional, but good practice)
        df_reduced = self.df.drop(columns=self.feature_cols, errors='ignore').copy()
        
        # Concatenate the original metadata with the new factor scores
        df_reduced = pd.concat([df_reduced, factor_scores_df], axis=1)

        # Save the final reduced data
        reduced_path = self.output_path.with_name(f"{self.output_path.stem}_reduced.parq")
        df_reduced.to_parquet(reduced_path)
        logger.info(f"Reduced data (U*Sigma) saved to: {reduced_path.name}")

        return df_reduced
        
    def run(self) -> Path:
        """
        Runs the full SVD analysis pipeline: loads data, generates plot to select k, 
        and performs final reduction and interpretation.
        
        :return: Path to the final saved plot image.
        """
        self._load_data()
        
        if self.A is None:
            logger.error("Feature matrix A was not created. Aborting SVD analysis.")
            return self.output_path

        logger.info("Starting SVD analysis and visualization...")
        
        # 1. Perform initial SVD for variance analysis
        # Find the max number of components
        n_comp_full = min(self.A.shape) - 1
        svd_full = self._run_svd(n_components=n_comp_full)
        
        # 2. Generate the plot and save the figure
        plot_path = self._generate_plot(svd_full)
        
        # 3. Determine k based on the plot (90% threshold)
        explained_variance_cumsum = np.cumsum(svd_full.explained_variance_ratio_)
        k_optimal = np.argmax(explained_variance_cumsum >= 0.90) + 1
        
        # 4. Perform final reduction and save the results
        self._perform_reduction_and_interpret(k=k_optimal)
        
        logger.info("SVD analysis complete: Plot generated and data/loadings saved.")
        return plot_path

# Public function to be used in main.py

def run_svd_analysis(input_path: Path, output_path: Path, config: dict) -> Path:
    """
    Main entry point for the SVD analysis and visualization process.
    
    :param input_path: Path to the feature-engineered CSV.
    :param output_path: Base path for output files (plot and data).
    :param config: The loaded application configuration.
    :return: Path to the final plot image file.
    """
    analyzer = SVDAnalyzer(
        input_path=input_path,
        output_path=output_path,
        config=config
    )
    return analyzer.run()