# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gaussian_kde
from scipy.integrate import trapezoid
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass
import warnings
from itertools import combinations

@dataclass
class SensitivityResults:
    """Container for sensitivity analysis results."""
    method: str
    first_order: Optional[Dict[str, float]] = None
    total_order: Optional[Dict[str, float]] = None
    second_order: Optional[Dict[Tuple[str,str],float]] = None
    local_sensitivity: Optional[Dict[str, np.ndarray]] = None
    parameter_names: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    
    def summary(self) -> str:
    """Generate summary string."""
    lines = [f"Sensitivity Analysis Results ({self.method})"] 
    lines.append("=" * 60)
    
    if self.first_order:
        lines.append("\nFirst-order indices (main effects):")
        for param, value in self.first_order.items():
            lines.append(f" {param:20s}: {value:8.4f}")
            
    if self.total_order:
        lines.append("\nTotal-order indices (main + interactions):")
        for param, value in self.total_order.items():
            lines.append(f" {param:20s}: {value:8.4f}")
                
    if self.second_order:
        lines.append("\nSecond-order indices (interactions):")
        for (p1, p2), value in list(self.second_order.items())[:10]:
            lines.append(f" {p1}-{p2:15s}: {value:8.4f}")       
    
    return "\n".join(lines)        

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for Gaussian Process emulators.
    Supports both global and local sensitivity analysis methods.
    """
    
    def __init__(
        self,
        emulator: Callable,
        bounds: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        seed: Optional[int] = None
       
    ):
        """
        Initialize sensitivity analyzer.
        
        Parameters
        ----------
        emulator : callable
            Trained emulator that takes X (n_samples, n_params) and returns predictions
            Should return (mean, variance) or just mean
        bounds : np.ndarray, shape (n_params, 2)   
            Parameter bounds [[lower1, upper1], [lower2, upper2], ...]
        parameter_names : list of str, optional    
            Names of parameters 
        seed : int, optional
            Random seed
        """
        self.emulator = emulator
        self.bounds = np.asarray(bounds)
        self.n_params = len(bounds)
        
        if parameter_names is None:
            self.parameter_names = [f"param_{i}" for i in range(self.n_params)]
        else:
            self.parameter_names = parameter_names
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        print(f"✓ Sensitivity analyzer initialized for {self.n_params} parameters")
