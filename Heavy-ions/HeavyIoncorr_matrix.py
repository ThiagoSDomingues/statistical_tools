# Author: OptimusThi

### Modeling a correlation matrix for experimental data is a common challenge 
# in heavy-ion physics when the full covariance matrix isn't published ### 

# Toy model: Gaussian correlation matrix
# This script generates a toy pT spectrum, 
# builds a correlation  matrix, and visualizes the difference btw corr and uncorr error bands.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Binning
pt_bins = np.linspace(0.5, 10, 20)
n_bins = len(pt_bins)

# Toy observable (power law pT spectra)
# Simplified model: dN/dpt ~ pt^-6
yields = 1000 * (pt_bins)**-6
rel_error = 0.1 # 10% relative uncertainty
sigma = yields * rel_error

# Building the correlation matrix
correlation_length = 2.0
corr_matrix = np.zeros((n_bins, n_bins))

for i in range(n_bins):
    for j in range(n_bins):
        dist = np.abs(pt_bins[i] - pt_bins[j])
        corr_matrix[i,j] = np.exp(-.5 * (dist / correlation_length)**2)

# Construct Covariance matrix: C_ij = rho_ij * sigma_i * sigma_j
cov_matrix = np.outer(sigma, sigma) * corr_matrix

# Generate Toy samples (correlated vs. uncorrelated)

