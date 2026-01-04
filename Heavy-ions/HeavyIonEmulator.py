### Author: OptimusThi
"""
Script to train Gaussian emualators.
"""
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels


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
  
