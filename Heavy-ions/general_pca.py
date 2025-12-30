# Importing all necessary libraries 

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
### must be -- u: (n_design_points, n_design_points); s: (n_features,); v:(n_features, n_features)
