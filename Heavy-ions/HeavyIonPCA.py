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
      standardize: bool
          Whether to standardize data (zero mean, unit variance)
      method: str
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
      """
      Validate input data shape and type.
      """
      if not isinstance(Y, np.ndarray):
          raise TypeError("Input must be a numpy array")
    
      if Y.ndim != 2:
          raise ValueError(f"Input must be 2D array, got shape {Y.shape}")
        
      if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
          raise ValueError("Input constains NaN or Inf values")
    
      print(f" ✓ Input validation passed: shape {Y.shape}")
    
  def fit(self, Y: np.ndarray) -> 'HeavyIonPCA':
      """
      Fit PCA to data.
    
      Parameters:
      ----------
      Y : np.ndarray
          Input data with shape (n_design_points, n_features)
          For pt-differential: features = [obs1_pt1, obs1_pt2, ..., obs2_pt1, ...]
          For pt-integrated: features = [obs1, obs2, obs3, ...]
      """
      self.validate_input()
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
    
# Example usage function
def example_usage():
    """Example of how to use the HeavyIonPCA class""".
  
    
      
if __name__ == "__main__":
    example_usage() 



      
   
      
      
  


