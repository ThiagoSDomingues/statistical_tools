### Author: OptimusThi

"""
script to use a Gaussian process regression to fit heavy-ion collisions experimental data
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# load experimental data (without uncertainties)

# fitting our model: initialization of hyperparameters
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process
gaussian_process.fit()
gaussian_process.kernel_ 


