# Importing all necessary libraries 

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

# Flattening simulation data: (n_design_points, n_features)
Y_flat = 

# Input matrix
Y = Y_flat.copy()
print("Y.shape: "+ str(Y.shape)) 
### improvement: validation here! ###

# Standardization: zero mean and unit variance
SS  =  StandardScaler(copy=True) # from sklearn module

# Singular Value Decomposition (SVD)
u, s, vh = np.linalg.svd(SS.fit_transform(Y), full_matrices=True)

print(f'shape of u: {u.shape}, shape of s: {s.shape}, shape of vh: {vh.shape})
### improvement: create a verification here! the shape of each component here must be right! ### 
### must be -- u: (n_design_points, n_design_points); s: (n_features,); v:(n_features, n_features) ###  

# print the explained ratio of variance
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))

# number of principal components to plot
n_pc_to_plot = 10 # make it more general or make 10 the standard number of pc's to plot

importance = np.square(s[:n_pc_to_plot]/math.sqrt(u.shape[0]-1))
cumulative_importance = np.cumsum(importance)/np.sum(importance)

idx = np.arange(1, 1+len(importance)) # PC index

# barplots 
ax1.bar(idx, importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")

ax2.bar(idx, cumulative_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
plt.tight_layout(True)
plt.show()

n_pcs = 10 # example. This must be chose by after the explained variance analysis by some criteria, or some threshold.  

# After check the explained variance we need to whiten and project data to principal component axis, only keeping the first n_pcs PCs.
pc_tf_data = u[:,0:n_pcs] * math.sqrt(u.shape[0]-1)
print(f'Shape of PC transformed data {pc_tf_data.shape}')

# Scale Transformation from PC space to original data space
inverse_tf_matrix = np.diag(s[0:n_pcs]) @ vh[0:n_pcs,:] * SS.scale_.reshape(1,Y_flat.shape[1])/ math.sqrt(u.shape[0]-1)

#### what's necessary for each emulator? ####  
# pc_tf_data, inverse_tf_matrix, SS, anything else?
