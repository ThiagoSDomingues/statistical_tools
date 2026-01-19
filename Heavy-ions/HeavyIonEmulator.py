### Author: OptimusThi
"""
Script to train Gaussian process emulators for heavy-ion collisions.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared,
    ConstantKernel, WhiteKernel, DotProduct
)
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
import warnings
from dataclasses import dataclass
import time


class KernelComparison:
    """Compare different kernels for GP emulation."""
    
    def __init__(
        self,
        X_train: np.ndarray
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        random_state: Optional[int] = None 
    ):
        """
        Initialize kernel comparison.
        """

# Unfinished!
def predict(self, X, return_cov=False, extra_std=0):
    """
    Predict model output at a point in parameter space `X`. 
    """
    if return_cov:
        return mean, _Covariance(cov, self._slices)
    else:
        return mean

if __name__ == '__main__:
  
