# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Union, Literal
import warnings


class HeavyIonPCA:
    """
    Flexible PCA analysis for heavy-ion collision observables.
    Supports both pt-integrated and pt-differential observables.
    """
    
    def __init__(
        self,
        kernel: Optional[str] = None,
        kernel_params: Optional[Dict] = None,
        standardize: bool = True,
        method: Literal['svd', 'sklearn'] = 'svd'
    ):
        """
        Initialize PCA analyzer.
        
        Parameters:
        -----------
        kernel : str, optional
            Kernel type for kernel PCA: 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
            If None, performs standard PCA
        kernel_params : dict, optional
            Parameters for kernel PCA (e.g., {'gamma': 0.1, 'degree': 3})
        standardize : bool
            Whether to standardize data (zero mean, unit variance)
        method : str
            'svd' for manual SVD implementation or 'sklearn' for sklearn PCA
        """
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.standardize = standardize
        self.method = method
        
        # Storage for fitted components
        self.scaler = None
        self.u = None
        self.s = None
        self.vh = None
        self.pca_model = None
        self.n_samples = None
        self.n_features = None
        self.is_fitted = False
        
    def validate_input(self, Y: np.ndarray) -> None:
        """Validate input data shape and type."""
        if not isinstance(Y, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if Y.ndim != 2:
            raise ValueError(f"Input must be 2D array, got shape {Y.shape}")
        
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("Input contains NaN or Inf values")
        
        print(f"✓ Input validation passed: shape {Y.shape}")
        
    def fit(self, Y: np.ndarray) -> 'HeavyIonPCA':
        """
        Fit PCA to data.
        
        Parameters:
        -----------
        Y : np.ndarray
            Input data with shape (n_design_points, n_features)
            For pt-differential: features = [obs1_pt1, obs1_pt2, ..., obs2_pt1, ...]
            For pt-integrated: features = [obs1, obs2, obs3, ...]
        """
        self.validate_input(Y)
        self.n_samples, self.n_features = Y.shape
        
        # Standardize data
        if self.standardize:
            self.scaler = StandardScaler()
            Y_scaled = self.scaler.fit_transform(Y)
        else:
            self.scaler = None
            Y_scaled = Y.copy()
        
        # Perform PCA based on method and kernel
        if self.kernel is not None:
            self._fit_kernel_pca(Y_scaled)
        elif self.method == 'svd':
            self._fit_svd(Y_scaled)
        else:
            self._fit_sklearn(Y_scaled)
        
        self.is_fitted = True
        return self
    
    def _fit_svd(self, Y_scaled: np.ndarray) -> None:
        """Fit using manual SVD decomposition."""
        self.u, self.s, self.vh = np.linalg.svd(Y_scaled, full_matrices=False)
        
        # Validate SVD shapes
        expected_shapes = {
            'u': (self.n_samples, min(self.n_samples, self.n_features)),
            's': (min(self.n_samples, self.n_features),),
            'vh': (min(self.n_samples, self.n_features), self.n_features)
        }
        
        assert self.u.shape == expected_shapes['u'], \
            f"u shape mismatch: expected {expected_shapes['u']}, got {self.u.shape}"
        assert self.s.shape == expected_shapes['s'], \
            f"s shape mismatch: expected {expected_shapes['s']}, got {self.s.shape}"
        assert self.vh.shape == expected_shapes['vh'], \
            f"vh shape mismatch: expected {expected_shapes['vh']}, got {self.vh.shape}"
        
        print(f"✓ SVD shapes validated: u{self.u.shape}, s{self.s.shape}, vh{self.vh.shape}")
    
    def _fit_sklearn(self, Y_scaled: np.ndarray) -> None:
        """Fit using sklearn PCA."""
        self.pca_model = PCA()
        self.pca_model.fit(Y_scaled)
        
        # Extract SVD components for compatibility
        self.s = np.sqrt(self.pca_model.explained_variance_ * (self.n_samples - 1))
        self.vh = self.pca_model.components_
        # u can be computed if needed: u = Y_scaled @ vh.T / s
        
        print(f"✓ sklearn PCA fitted with {len(self.s)} components")
    
    def _fit_kernel_pca(self, Y_scaled: np.ndarray) -> None:
        """Fit using kernel PCA."""
        self.pca_model = KernelPCA(
            kernel=self.kernel,
            fit_inverse_transform=True,
            **self.kernel_params
        )
        self.pca_model.fit(Y_scaled)
        print(f"✓ Kernel PCA fitted with {self.kernel} kernel")
    
    def explained_variance_ratio(self, n_components: Optional[int] = None) -> np.ndarray:
        """Calculate explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("Must fit model before computing explained variance")
        
        if self.kernel is not None:
            warnings.warn("Explained variance not directly available for kernel PCA")
            return None
        
        if self.method == 'sklearn':
            var_ratio = self.pca_model.explained_variance_ratio_
        else:
            variance = np.square(self.s) / (self.n_samples - 1)
            var_ratio = variance / np.sum(variance)
        
        if n_components is not None:
            return var_ratio[:n_components]
        return var_ratio
    
    def plot_variance(
        self,
        n_components: int = 10,
        threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Plot explained variance and cumulative variance.
        
        Parameters:
        -----------
        n_components : int
            Number of components to plot
        threshold : float, optional
            Threshold line for cumulative variance (e.g., 0.95)
        figsize : tuple
            Figure size
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before plotting variance")
        
        var_ratio = self.explained_variance_ratio(n_components)
        if var_ratio is None:
            print("Cannot plot variance for kernel PCA")
            return None
        
        cumulative_var = np.cumsum(var_ratio)
        idx = np.arange(1, len(var_ratio) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        ax1.bar(idx, var_ratio, alpha=0.7, color='steelblue')
        ax1.set_xlabel("Principal Component", fontsize=11)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=11)
        ax1.set_title("Variance per Component", fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Cumulative variance
        ax2.plot(idx, cumulative_var, 'o-', color='darkred', linewidth=2, markersize=6)
        ax2.set_xlabel("Number of Components", fontsize=11)
        ax2.set_ylabel("Cumulative Variance Ratio", fontsize=11)
        ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        if threshold is not None:
            ax2.axhline(y=threshold, color='green', linestyle='--', 
                       label=f'{threshold:.1%} threshold', linewidth=2)
            # Find number of components needed
            n_needed = np.argmax(cumulative_var >= threshold) + 1
            ax2.axvline(x=n_needed, color='orange', linestyle='--', 
                       label=f'{n_needed} components', linewidth=2)
            ax2.legend()
            print(f"Components needed for {threshold:.1%} variance: {n_needed}")
        
        plt.tight_layout()
        return fig
    
    def transform(
        self,
        Y: Optional[np.ndarray] = None,
        n_components: int = 10,
        whiten: bool = True
    ) -> np.ndarray:
        """
        Transform data to PC space.
        
        Parameters:
        -----------
        Y : np.ndarray, optional
            Data to transform. If None, transforms training data
        n_components : int
            Number of PCs to keep
        whiten : bool
            Whether to whiten (scale by sqrt(n_samples - 1))
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before transforming")
        
        if Y is not None:
            self.validate_input(Y)
            if self.scaler is not None:
                Y_scaled = self.scaler.transform(Y)
            else:
                Y_scaled = Y.copy()
        else:
            # Transform training data
            Y_scaled = self.scaler.transform(Y) if self.scaler and Y is not None else None
        
        if self.kernel is not None:
            pc_data = self.pca_model.transform(Y_scaled)[:, :n_components]
        elif self.method == 'sklearn':
            pc_data = self.pca_model.transform(Y_scaled)[:, :n_components]
        else:
            pc_data = self.u[:, :n_components]
            if whiten:
                pc_data = pc_data * np.sqrt(self.n_samples - 1)
        
        print(f"✓ Transformed to {n_components} principal components: shape {pc_data.shape}")
        return pc_data
    
    def get_inverse_transform_matrix(self, n_components: int = 10) -> np.ndarray:
        """
        Get transformation matrix from PC space back to original space.
        
        Returns:
        --------
        inv_matrix : np.ndarray
            Matrix to transform from PC space to scaled original space
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before getting inverse transform")
        
        if self.kernel is not None:
            raise NotImplementedError("Inverse transform matrix not available for kernel PCA")
        
        if self.method == 'sklearn':
            # For sklearn: inv_matrix = components.T * std
            inv_matrix = self.pca_model.components_[:n_components, :].T
            if self.scaler is not None:
                inv_matrix = inv_matrix * self.scaler.scale_.reshape(-1, 1)
        else:
            # For SVD: (S @ V) / sqrt(n-1) * scale
            inv_matrix = (np.diag(self.s[:n_components]) @ self.vh[:n_components, :]) / \
                        np.sqrt(self.n_samples - 1)
            if self.scaler is not None:
                inv_matrix = inv_matrix * self.scaler.scale_.reshape(1, -1)
        
        print(f"✓ Inverse transform matrix shape: {inv_matrix.shape}")
        return inv_matrix
    
    def inverse_transform(
        self,
        pc_data: np.ndarray,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Transform PC data back to original space.
        
        Parameters:
        -----------
        pc_data : np.ndarray
            Data in PC space
        n_components : int, optional
            Number of components used (inferred from pc_data if None)
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before inverse transforming")
        
        if n_components is None:
            n_components = pc_data.shape[1]
        
        if self.kernel is not None:
            Y_reconstructed = self.pca_model.inverse_transform(pc_data)
        else:
            inv_matrix = self.get_inverse_transform_matrix(n_components)
            # Unwhiten if needed
            pc_unwhitened = pc_data / np.sqrt(self.n_samples - 1)
            Y_reconstructed = pc_unwhitened @ inv_matrix
        
        # Inverse standardization
        if self.scaler is not None:
            Y_reconstructed = self.scaler.inverse_transform(Y_reconstructed)
        
        return Y_reconstructed
    
    def reconstruction_error(
        self,
        Y: np.ndarray,
        n_components: int
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate reconstruction error for given number of components.
        
        Returns:
        --------
        mse : float
            Mean squared error
        errors_per_sample : np.ndarray
            Error for each sample
        """
        pc_data = self.transform(Y, n_components)
        Y_reconstructed = self.inverse_transform(pc_data, n_components)
        
        errors = np.mean((Y - Y_reconstructed)**2, axis=1)
        mse = np.mean(errors)
        
        return mse, errors
    
    def select_n_components(
        self,
        variance_threshold: float = 0.95,
        max_components: Optional[int] = None
    ) -> int:
        """
        Automatically select number of components based on variance threshold.
        
        Parameters:
        -----------
        variance_threshold : float
            Cumulative variance ratio to achieve (e.g., 0.95 for 95%)
        max_components : int, optional
            Maximum number of components to consider
        """
        if self.kernel is not None:
            raise NotImplementedError("Automatic selection not available for kernel PCA")
        
        var_ratio = self.explained_variance_ratio()
        cumulative_var = np.cumsum(var_ratio)
        
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1
        
        if max_components is not None:
            n_components = min(n_components, max_components)
        
        print(f"✓ Selected {n_components} components for {variance_threshold:.1%} variance")
        return n_components
    
    def get_summary(self) -> Dict:
        """Get summary of fitted PCA."""
        if not self.is_fitted:
            raise RuntimeError("Must fit model first")
        
        summary = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'kernel': self.kernel or 'None (standard PCA)',
            'standardized': self.standardize,
            'method': self.method
        }
        
        if self.kernel is None:
            var_ratio = self.explained_variance_ratio()
            summary['total_components'] = len(var_ratio)
            summary['variance_first_10'] = np.sum(var_ratio[:10])
        
        return summary


# Example usage function
def example_usage():
    """Example of how to use the HeavyIonPCA class."""
    
    # Generate synthetic heavy-ion collision data
    np.random.seed(42)
    n_design_points = 100
    n_observables = 5
    n_pt_bins = 10
    
    # Example 1: pt-differential data
    n_features_diff = n_observables * n_pt_bins
    Y_diff = np.random.randn(n_design_points, n_features_diff) * 0.5 + \
             np.random.randn(1, n_features_diff)
    
    print("=" * 60)
    print("EXAMPLE 1: Standard PCA on pt-differential data")
    print("=" * 60)
    
    pca_diff = HeavyIonPCA(standardize=True, method='svd')
    pca_diff.fit(Y_diff)
    
    # Select components automatically
    n_comp = pca_diff.select_n_components(variance_threshold=0.95)
    
    # Transform data
    pc_data = pca_diff.transform(n_components=n_comp)
    
    # Plot variance
    fig = pca_diff.plot_variance(n_components=20, threshold=0.95)
    plt.show()
    
    # Get inverse transform matrix for emulator
    inv_matrix = pca_diff.get_inverse_transform_matrix(n_components=n_comp)
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Kernel PCA with RBF kernel")
    print("=" * 60)
    
    # Example 2: Kernel PCA
    kpca = HeavyIonPCA(kernel='rbf', kernel_params={'gamma': 0.1}, standardize=True)
    kpca.fit(Y_diff)
    pc_data_kernel = kpca.transform(n_components=10)
    
    # Calculate reconstruction error
    mse, _ = pca_diff.reconstruction_error(Y_diff, n_components=n_comp)
    print(f"\nReconstruction MSE with {n_comp} components: {mse:.6f}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    summary = pca_diff.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()
