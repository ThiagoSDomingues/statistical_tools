#!/usr/bin/env python3

"""
This script generates plots / figures for Bayesian analysis results in heavy-ion collisions
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from HeavyIonPCA import inverse_tf_matrix, scaler
from HeavyIonEmulators import predict_observables 
from HeavyIonMCMC import Chain
### Load: Emulators, inverse_tf_matrix, scaler

@plot
def plot_prior_CI(
    n_pr_samples, 
    show_exp=False,
    bound_min=5,
    bound_max=95,
    save_folder="prior"
):
    """
    Plot prior and Bayesian design point predictions overlaid with 90% credible intervals.
    """
  
    # Collect scaled spectra predictions for each prior sample
    pr_predictions = []

    n_params = 17 # number of parameters (parameter space dimension)
    
    # Looping over all prior samples: supposing a uniform prior
    # Use the emulator surrogate to predict observables for each prior sample
    for params in np.random.uniform(design_min, design_max, (n_pr_samples, n_params)):
        y_pred, cov_pred = predict_observables(params, Emulators, inverse_tf_matrix, scaler)
        pr_predictions.append(y_pred.flatten())
    
    pr_predictions = np.array(pr_predictions) # shape (n_samples, n_obs)
     
    
