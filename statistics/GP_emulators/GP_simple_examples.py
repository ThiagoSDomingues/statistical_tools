### Author: OptimusThi

"""
Script to use a Gaussian process regression to fit heavy-ion collisions experimental data
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# load experimental data (without uncertainties): training dataset
X_train, y_train  = load_experimental_data

# fitting our model: initialization of hyperparameters
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_ 

# emulator's predictions: mean prediction and a confidence interval
mean_prediciton, std_prediction = gaussian_process.predict(X_new, return_std=True)

# plotting predictions
plt.plot()
plt.scatter()
plt.plot()
plt.fill_between()
plt.legend()
plt.xlabel("kinematic variable")
plt.ylabel("observable")
_ = plt.title("Gaussian process regression on noise-free dataset")

### Example: noisy targets
 
X_train, y_train = load_experimental_data
