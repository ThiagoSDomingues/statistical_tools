### Author: OptimusThi
"""
Script to specify and compare different kernels for GPR
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

X_train, y_train = load_training_data

kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
gpr.fit(X_train, y_train)
y_mean, y_std = gpr.predict(X, return_std=True)

### Matern kernel example
# Define a kernel with initial hyperparameters
kernel = Matern(length_scale=1.0, nu=1.5)

gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, normalize_y=True)

gp.fit(X_train, y_train)

