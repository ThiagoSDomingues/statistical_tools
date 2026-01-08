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
n_samples = 100 
uncorrelated_samples = np.random.multivariate_normal(yields, np.diag(sigma**2), n_samples)
correlated_samples = np.random.multivariate_normal(yields, cov_matrix, n_samples)

# ---- Visualization ---- # 
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: correlation matrix
sns.heatmap(corr_matrix, annot=False, cmap='viridis', ax=ax[0])
ax[0].set_title("Modeled Correlation Matrix ($p_T$ bins)")
ax[0].set_xlabel("Bin index")
ax[0].set_ylabel("Bind index")

# Plot B: effect on observable
ax[1].errorbar(pt_bins, yields, yerr=sigma, fmt='ko', label='Central Value', capsize=4)

# Plot a few samples
for i in range(5):
    ax[1].plot(pt_bins, correlated_samples[i], color='blue', alpha=0.3, label='Correlated Sample' if i==0 else "")
    ax[1].plot(pt_bins, uncorrelated_samples[i], color='red', linestyle='--', alpha=0.3, label='Uncorrelated Sample' if i==0 else "")

ax[1].set_yscale('log')
ax[1].set_xlabel('$p_T$ [GeV/c]')
ax[1].set_ylabel('$d^2N/dp_T dy$')
ax[1].set_title("Impact of Correlations on Spectra Shape")
ax[1].legend()

plt.tight_layout()
plt.show()
