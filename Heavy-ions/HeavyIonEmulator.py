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
  
