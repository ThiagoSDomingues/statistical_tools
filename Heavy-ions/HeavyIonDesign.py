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
        bounds : ParameterBounds or tuple of (lower, upper)
            Parameter space bounds
        seed : int, optional
            Random seed for reproducibility
        """
        if isinstance(bound, tuple):
            self.bounds = ParameterBounds(bounds[0], bounds[1])
        else:
            self.bounds = bounds

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.dim = self.bounds.dim

        print(f"✓ Design generator initialized: {self.dim}D parameter space")

    def generate(
        self,
        n_points: int,
        method: Literal[
            'random', 'lhs', 'sobol', 'halton', 'maximin', 'minimax', 'uniform_grid', 'sphere_packing'
        ] = 'lhs', 
        **kwargs
    ) -> np.ndarray:
    """
    Generate design points.
    
    Parameters:
    -----------
    n_points : int
        Number of design points
    method : str
        Sampling method
    **kwargs : additional method-specific parameters
    
    Returns:
    --------
    X : np.ndarray, shape (n_points, dim)
        Design points in original parameter space
    """
    method_map = {
        'random': self._random_sampling,
        'lhs': self._latin_hypercube, 
        'sobol': self._sobol_sequence, 
        'halton': self._halton_sequence, 
        'maximin': self._maximin_lhs,
        'minimax': self._minimax_distance,
        'uniform_grid': self._uniform_grid,
        'sphere_packing': self._sphere_packing
    }
    
    if method no in method_map:
        raise ValueError(f"Unknown method '{method}'. Avilable: {List(method_map.keys())}")
    print(f"Generating {n_points} design points using '{method}' method...")
    X = method_map[method](n_points, **kwargs)
    
    # Validate output.
    assert X.shape == (n_points, self.dim), f"Shape mismatch: {X.shape}"
    assert np.all(X >= self.bounds.lower) and np.all(X <= self.bounds.upper), \ 
        "Points outside bounds"
    
    print(f" ✓ Generated {n_points} design points)
    return X
                       

