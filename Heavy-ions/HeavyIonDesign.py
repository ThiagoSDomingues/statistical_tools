# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
import warnings
from dataclasses import dataclass

@dataclass
class ParameterBounds:
    """Define parameter bounds for design space."""
    lower: np.ndarray
    upper: np.ndarray
    
    def __post_init__(self):
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)
        if len(self.lower) != len(self.upper):
            raise ValueError("Lower and upper bounds must have same length")        
        if np.array(self.lower >= self.upper):
            raise ValueError("Lower bounds must be strictly less than upper bounds")                    
        self.dim = len(self.lower)
        
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize points to [0, 1]^d."""
        return (X - self.lower) / (self.upper - self.lower)
        
    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1]^d to original bounds."""
        return X_norm * (self.upper - self.lower) + self.lower
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """Clip points to bounds."""
        return np.clip(X, self.lower, self.upper)

class DesignGenerator:
    """
    Flexible design point generator for Gaussian Process emulators.
    Supports multiple sampling strategies including adaptive methods.
    """
    
    def __init__(
        self,
        bounds: Union[ParameterBounds, Tuple[List, List]],
        seed: Optional[int] = None
    
    ):
        """
        Initialize design generator.
        
        Parameters:
        -----------
