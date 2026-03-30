### Author: OptimusThi
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


@dataclass
class EmulatorMetrics:
    """Container for emulator validation metrics."""
    rmse: float
    mae: float
    r2: float
    max_error: float
    mean_std: Optional[float] = None
    coverage_95: Optional[float] = None
    log_likelihood: Optional[float] = None
    
    def __str__(self):
        lines = [
            f"RMSE:          {self.rmse:.6f}",
            f"MAE:           {self.mae:.6f}",
            f"R²:            {self.r2:.6f}",
            f"Max Error:     {self.max_error:.6f}",
        ]
        if self.mean_std is not None:
            lines.append(f"Mean Std:      {self.mean_std:.6f}")
        if self.coverage_95 is not None:
            lines.append(f"95% Coverage:  {self.coverage_95:.2%}")
        if self.log_likelihood is not None:
            lines.append(f"Log Likelihood: {self.log_likelihood:.2f}")
        return "\n".join(lines)


class GPEmulator:
    """
    Flexible Gaussian Process emulator for heavy-ion collision observables.
    Supports multiple kernels, optimization strategies, and validation methods.
    """
    
    def __init__(
        self,
        kernel_type: Literal[
            'rbf', 'matern', 'rq', 'rbf+matern', 'rbf+rq', 'custom'
        ] = 'rbf',
        kernel_params: Optional[Dict] = None,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-10,
        random_state: Optional[int] = None
    ):
        """
        Initialize GP emulator.
        
        Parameters:
        -----------
        kernel_type : str
            Type of kernel to use
        kernel_params : dict, optional
            Kernel-specific parameters
        normalize_y : bool
            Whether to normalize target values
        n_restarts_optimizer : int
            Number of restarts for hyperparameter optimization
        alpha : float
            Noise level (nugget term)
        random_state : int, optional
            Random seed
        """
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.random_state = random_state
        
        self.gp = None
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        self.training_time = None
        
    def _create_kernel(self, n_dims: int):
        """Create kernel based on kernel_type."""
        
        # Default length scales
        length_scale = self.kernel_params.get('length_scale', [1.0] * n_dims)
        length_scale_bounds = self.kernel_params.get(
            'length_scale_bounds', 
            (1e-2, 1e2)
        )
        
        if self.kernel_type == 'rbf':
            # Squared Exponential kernel (RBF)
            # k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
            kernel = ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-3, 1e3)
            ) * RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            )
            
        elif self.kernel_type == 'matern':
            # Matérn kernel (more flexible, controls smoothness)
            # nu controls smoothness: 1/2, 3/2, 5/2, or inf
            nu = self.kernel_params.get('nu', 2.5)
            kernel = ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-3, 1e3)
            ) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=nu
            )
            
        elif self.kernel_type == 'rq':
            # Rational Quadratic kernel (mixture of RBF kernels)
            # k(x, x') = σ² (1 + ||x - x'||² / (2αℓ²))^(-α)
            alpha_rq = self.kernel_params.get('alpha', 1.0)
            kernel = ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-3, 1e3)
            ) * RationalQuadratic(
                length_scale=length_scale,
                alpha=alpha_rq,
                length_scale_bounds=length_scale_bounds
            )
            
        elif self.kernel_type == 'rbf+matern':
            # Combination: RBF for smooth trends + Matérn for local variations
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds) +
                ConstantKernel(0.1, (1e-4, 1e2)) * Matern(length_scale, length_scale_bounds, nu=1.5)
            )
            
        elif self.kernel_type == 'rbf+rq':
            # Combination: RBF + Rational Quadratic
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds) +
                ConstantKernel(0.1, (1e-4, 1e2)) * RationalQuadratic(length_scale, 1.0, length_scale_bounds)
            )
            
        elif self.kernel_type == 'custom':
            # User-provided kernel
            kernel = self.kernel_params.get('kernel')
            if kernel is None:
                raise ValueError("Must provide 'kernel' in kernel_params for custom kernel")
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Add white noise kernel
        if self.kernel_params.get('add_white_noise', True):
            noise_level = self.kernel_params.get('noise_level', 1e-5)
            kernel += WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=(1e-10, 1e-1)
            )
        
        return kernel
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimize: bool = True,
        verbose: bool = True
    ):
        """
        Fit GP emulator to training data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training input (design points)
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Training output (observables)
        optimize : bool
            Whether to optimize hyperparameters
        verbose : bool
            Print fitting information
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        if self.y_train.ndim == 1:
            self.y_train = self.y_train.reshape(-1, 1)
        
        n_samples, n_dims = self.X_train.shape
        n_outputs = self.y_train.shape[1]
        
        if verbose:
            print(f"Fitting GP emulator:")
            print(f"  Training samples: {n_samples}")
            print(f"  Input dimensions: {n_dims}")
            print(f"  Output dimensions: {n_outputs}")
            print(f"  Kernel type: {self.kernel_type}")
        
        # Create kernel
        kernel = self._create_kernel(n_dims)
        
        # Create GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer if optimize else 0,
            alpha=self.alpha,
            random_state=self.random_state
        )
        
        # Fit
        start_time = time.time()
        
        if n_outputs == 1:
            self.gp.fit(self.X_train, self.y_train.ravel())
        else:
            # For multiple outputs, fit separate GP for each
            # (Can be extended to use MultiOutputGP)
            warnings.warn(
                "Multiple outputs detected. Fitting first output only. "
                "For multiple outputs, fit separate emulators."
            )
            self.gp.fit(self.X_train, self.y_train[:, 0])
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        if verbose:
            print(f"  Training time: {self.training_time:.2f}s")
            print(f"  Optimized kernel: {self.gp.kernel_}")
            print(f"  Log-likelihood: {self.gp.log_marginal_likelihood_value_:.2f}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with GP emulator.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Test points
        return_std : bool
            Whether to return standard deviation
        return_cov : bool
            Whether to return covariance matrix
        
        Returns:
        --------
        y_pred : ndarray
            Predicted mean
        y_std : ndarray (if return_std=True)
            Predicted standard deviation
        y_cov : ndarray (if return_cov=True)
            Predicted covariance matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit emulator before prediction")
        
        X = np.asarray(X)
        
        if return_cov:
            y_pred, y_cov = self.gp.predict(X, return_cov=True)
            return y_pred, y_cov
        elif return_std:
            y_pred, y_std = self.gp.predict(X, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.gp.predict(X, return_std=False)
            return y_pred
    
    def validate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_predictions: bool = False
    ) -> EmulatorMetrics:
        """
        Validate emulator on test data.
        
        Parameters:
        -----------
        X_test : ndarray
            Test inputs
        y_test : ndarray
            True test outputs
        return_predictions : bool
            Whether to return predictions
        
        Returns:
        --------
        metrics : EmulatorMetrics
            Validation metrics
        predictions : tuple (if return_predictions=True)
            (y_pred, y_std)
        """
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        max_error = np.max(np.abs(y_test - y_pred))
        mean_std = np.mean(y_std)
        
        # Calculate 95% coverage (calibration)
        errors = np.abs(y_test.ravel() - y_pred.ravel())
        within_2std = np.sum(errors < 2 * y_std) / len(errors)
        
        # Log likelihood
        log_likelihood = self.gp.log_marginal_likelihood_value_
        
        metrics = EmulatorMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            max_error=max_error,
            mean_std=mean_std,
            coverage_95=within_2std,
            log_likelihood=log_likelihood
        )
        
        if return_predictions:
            return metrics, (y_pred, y_std)
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Perform k-fold cross-validation.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        y : ndarray
            Output data
        cv : int
            Number of folds
        verbose : bool
            Print progress
        
        Returns:
        --------
        cv_results : dict
            Cross-validation metrics for each fold
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'max_error': [],
            'coverage_95': []
        }
        
        if verbose:
            print(f"\nPerforming {cv}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            
            # Create and fit new GP for this fold
            fold_emulator = GPEmulator(
                kernel_type=self.kernel_type,
                kernel_params=self.kernel_params,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=self.alpha,
                random_state=self.random_state
            )
            
            fold_emulator.fit(X_train_fold, y_train_fold, verbose=False)
            metrics = fold_emulator.validate(X_test_fold, y_test_fold)
            
            cv_scores['rmse'].append(metrics.rmse)
            cv_scores['mae'].append(metrics.mae)
            cv_scores['r2'].append(metrics.r2)
            cv_scores['max_error'].append(metrics.max_error)
            cv_scores['coverage_95'].append(metrics.coverage_95)
            
            if verbose:
                print(f"  Fold {fold+1}/{cv}: RMSE={metrics.rmse:.6f}, R²={metrics.r2:.4f}")
        
        # Convert to arrays
        for key in cv_scores:
            cv_scores[key] = np.array(cv_scores[key])
        
        if verbose:
            print(f"\nCross-validation results:")
            print(f"  RMSE: {np.mean(cv_scores['rmse']):.6f} ± {np.std(cv_scores['rmse']):.6f}")
            print(f"  MAE:  {np.mean(cv_scores['mae']):.6f} ± {np.std(cv_scores['mae']):.6f}")
            print(f"  R²:   {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
            print(f"  95% Coverage: {np.mean(cv_scores['coverage_95']):.2%} ± {np.std(cv_scores['coverage_95']):.2%}")
        
        return cv_scores
    
    def leave_one_out_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Leave-one-out cross-validation (LOO-CV).
        More expensive but provides unbiased error estimate.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples = len(X)
        y_pred_loo = np.zeros(n_samples)
        y_std_loo = np.zeros(n_samples)
        
        if verbose:
            print(f"\nPerforming Leave-One-Out CV ({n_samples} iterations)...")
        
        for i in range(n_samples):
            # Leave out sample i
            train_idx = np.delete(np.arange(n_samples), i)
            X_train_loo = X[train_idx]
            y_train_loo = y[train_idx]
            X_test_loo = X[i:i+1]
            
            # Fit and predict
            loo_emulator = GPEmulator(
                kernel_type=self.kernel_type,
                kernel_params=self.kernel_params,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=0,  # Don't optimize for each fold
                alpha=self.alpha,
                random_state=self.random_state
            )
            
            loo_emulator.fit(X_train_loo, y_train_loo, verbose=False)
            y_pred_loo[i], y_std_loo[i] = loo_emulator.predict(X_test_loo, return_std=True)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_samples}")
        
        # Calculate metrics
        rmse_loo = np.sqrt(mean_squared_error(y.ravel(), y_pred_loo))
        mae_loo = mean_absolute_error(y.ravel(), y_pred_loo)
        r2_loo = r2_score(y.ravel(), y_pred_loo)
        
        errors = np.abs(y.ravel() - y_pred_loo)
        coverage_95_loo = np.sum(errors < 2 * y_std_loo) / n_samples
        
        loo_results = {
            'rmse': rmse_loo,
            'mae': mae_loo,
            'r2': r2_loo,
            'coverage_95': coverage_95_loo,
            'predictions': y_pred_loo,
            'std': y_std_loo
        }
        
        if verbose:
            print(f"\nLOO-CV results:")
            print(f"  RMSE: {rmse_loo:.6f}")
            print(f"  MAE:  {mae_loo:.6f}")
            print(f"  R²:   {r2_loo:.4f}")
            print(f"  95% Coverage: {coverage_95_loo:.2%}")
        
        return loo_results
    
    def plot_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """Plot prediction quality."""
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Predicted vs True
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        lim_min = min(y_test.min(), y_pred.min())
        lim_max = max(y_test.max(), y_pred.max())
        axes[0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
        axes[0].set_xlabel('True Values', fontsize=11)
        axes[0].set_ylabel('Predicted Values', fontsize=11)
        axes[0].set_title('Predicted vs True', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. Residuals
        residuals = y_test.ravel() - y_pred.ravel()
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].fill_between([y_pred.min(), y_pred.max()], -2*y_std.mean(), 2*y_std.mean(), 
                            alpha=0.2, color='red', label='±2σ (mean)')
        axes[1].set_xlabel('Predicted Values', fontsize=11)
        axes[1].set_ylabel('Residuals', fontsize=11)
        axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # 3. Uncertainty calibration
        standardized_errors = np.abs(residuals) / y_std
        axes[2].hist(standardized_errors, bins=20, alpha=0.7, edgecolor='black', density=True)
        
        # Overlay expected distribution (half-normal for |error|/std)
        x_theory = np.linspace(0, standardized_errors.max(), 100)
        axes[2].plot(x_theory, 2*norm.pdf(x_theory), 'r-', linewidth=2, label='Expected (|N(0,1)|)')
        axes[2].axvline(x=2, color='orange', linestyle='--', linewidth=2, label='2σ threshold')
        axes[2].set_xlabel('|Error| / Predicted Std', fontsize=11)
        axes[2].set_ylabel('Density', fontsize=11)
        axes[2].set_title('Uncertainty Calibration', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


class KernelComparison:
    """Compare different kernels for GP emulation."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        random_state: Optional[int] = None
    ):
        """
        Initialize kernel comparison.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        random_state : Random seed
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.results = {}
    
    def compare_kernels(
        self,
        kernel_types: List[str] = None,
        n_restarts: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Compare different kernel types.
        
        Parameters:
        -----------
        kernel_types : list of str
            Kernels to compare
        n_restarts : int
            Optimization restarts
        verbose : bool
            Print progress
        
        Returns:
        --------
        results : dict
            Comparison results
        """
        if kernel_types is None:
            kernel_types = ['rbf', 'matern', 'rq', 'rbf+matern']
        
        if verbose:
            print("=" * 70)
            print("KERNEL COMPARISON")
            print("=" * 70)
        
        for kernel_type in kernel_types:
            if verbose:
                print(f"\n--- Testing {kernel_type.upper()} kernel ---")
            
            emulator = GPEmulator(
                kernel_type=kernel_type,
                normalize_y=True,
                n_restarts_optimizer=n_restarts,
                random_state=self.random_state
            )
            
            # Fit
            emulator.fit(self.X_train, self.y_train, verbose=verbose)
            
            # Validate
            metrics = emulator.validate(self.X_test, self.y_test)
            
            self.results[kernel_type] = {
                'emulator': emulator,
                'metrics': metrics,
                'kernel': str(emulator.gp.kernel_),
                'training_time': emulator.training_time
            }
            
            if verbose:
                print(f"\nValidation Metrics:")
                print(metrics)
        
        return self.results
    
    def plot_comparison(self, figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """Plot kernel comparison results."""
        if not self.results:
            raise RuntimeError("Must run compare_kernels first")
        
        n_kernels = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        kernel_names = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, n_kernels))
        
        # Metrics to plot
        metrics_to_plot = ['rmse', 'mae', 'r2', 'coverage_95']
        titles = ['RMSE (lower is better)', 'MAE (lower is better)', 
                 'R² (higher is better)', '95% Coverage (closer to 0.95 is better)']
        
        for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
            values = [self.results[k]['metrics'].__dict__[metric] for k in kernel_names]
            
            bars = ax.bar(range(n_kernels), values, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            
            # Highlight best
            if metric in ['rmse', 'mae']:
                best_idx = np.argmin(values)
            elif metric == 'coverage_95':
                best_idx = np.argmin(np.abs(np.array(values) - 0.95))
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.set_xticks(range(n_kernels))
            ax.set_xticklabels(kernel_names, rotation=45, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            
            # Add reference line for coverage
            if metric == 'coverage_95':
                ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='Target (0.95)')
                ax.legend()
        
        plt.suptitle('Kernel Comparison', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig


# Example usage and testing
def example_usage():
    """Comprehensive example of GP emulator usage."""
    
    print("=" * 70)
    print("GP EMULATOR FRAMEWORK - COMPREHENSIVE EXAMPLE")
    print("=" * 70)
    
    # Generate synthetic HIC-like data
    np.random.seed(42)
    n_train = 100
    n_test = 30
    n_dims = 5
    
    # Training data
    X_train = np.random.uniform(0, 1, size=(n_train, n_dims))
    
    # Create non-linear response (mimicking v2 dependence on eta/s)
    def true_function(X):
        # Complex nonlinear function
        y = (0.1 * (1 - np.exp(-5 * X[:, 0])) *  # eta/s effect
             np.sqrt(X[:, 2]) *                    # tau_0 effect
             (1 + 0.3 * X[:, 1]) +                 # zeta/s effect
             0.05 * np.sin(3 * X[:, 3]) +          # periodic effect
             0.02 * X[:, 4])                       # linear effect
        return y + np.random.normal(0, 0.01, len(X))
    
    y_train = true_function(X_train)
    
    # Test data
    X_test = np.random.uniform(0, 1, size=(n_test, n_dims))
    y_test = true_function(X_test)
    
    print(f"\nDataset:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Example 1: Basic GP emulator
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic GP Emulator with RBF Kernel")
    print("=" * 70)
    
    emulator_rbf = GPEmulator(kernel_type='rbf', n_restarts_optimizer=10)
    emulator_rbf.fit(X_train, y_train)
    
    metrics_rbf = emulator_rbf.validate(X_test, y_test)
    print(f"\nValidation metrics:")
    print(metrics_rbf)
    
    fig1 = emulator_rbf.plot_predictions(X_test, y_test)
    plt.show()
    
    # Example 2: Cross-validation
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Cross-Validation")
    print("=" * 70)
    
    cv_results = emulator_rbf.cross_validate(X_train, y_train, cv=5)
    
    # Plot CV results
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_cv = ['rmse', 'r2', 'coverage_95']
    titles = ['RMSE per Fold', 'R² per Fold', '95% Coverage per Fold']
    
    for ax, metric, title in zip(axes, metrics_cv, titles):
        values = cv_results[metric]
        ax.bar(range(len(values)), values, alpha=0.7, edgecolor='black')
        ax.axhline(y=values.mean(), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {values.mean():.4f}')
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 3: Kernel comparison
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Kernel Comparison")
    print("=" * 70)
    
    kernel_comp = KernelComparison(X_train, y_train, X_test, y_test, random_state=42)
    comparison_results = kernel_comp.compare_kernels(
        kernel_types=['rbf', 'matern', 'rq', 'rbf+matern'],
        n_restarts=5,
        verbose=True
    )
    
    fig3 = kernel_comp.plot_comparison()
    plt.show()
    
    # Example 4: Different Matérn smoothness parameters
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Matérn Kernel with Different Smoothness")
    print("=" * 70)
    
    nu_values = [0.5, 1.5, 2.5]
    matern_results = {}
    
    for nu in nu_values:
        print(f"\n--- Matérn with nu={nu} ---")
        emulator = GPEmulator(
            kernel_type='matern',
            kernel_params={'nu': nu},
            n_restarts_optimizer=5,
            random_state=42
        )
        emulator.fit(X_train, y_train, verbose=False)
        metrics = emulator.validate(X_test, y_test)
        matern_results[f'nu={nu}'] = metrics
        print(metrics)
    
    # Example 5: Leave-One-Out CV (for small datasets)
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Leave-One-Out Cross-Validation")
    print("=" * 70)
    
    # Use smaller dataset for LOO-CV demonstration
    X_small = X_train[:30]
    y_small = y_train[:30]
    
    emulator_loo = GPEmulator(kernel_type='rbf', n_restarts_optimizer=5, random_state=42)
    emulator_loo.fit(X_small, y_small, verbose=False)
    
    loo_results = emulator_loo.leave_one_out_cv(X_small, y_small, verbose=True)
    
    # Plot LOO-CV results
    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs True
    axes[0].scatter(y_small, loo_results['predictions'], alpha=0.6, s=50, 
                   edgecolors='black', linewidths=0.5)
    lim = [y_small.min(), y_small.max()]
    axes[0].plot(lim, lim, 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True Values', fontsize=11)
    axes[0].set_ylabel('LOO-CV Predictions', fontsize=11)
    axes[0].set_title('Leave-One-Out CV: Predicted vs True', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Standardized errors
    errors = np.abs(y_small.ravel() - loo_results['predictions'])
    std_errors = errors / loo_results['std']
    axes[1].hist(std_errors, bins=15, alpha=0.7, edgecolor='black', density=True)
    x_theory = np.linspace(0, std_errors.max(), 100)
    axes[1].plot(x_theory, 2*norm.pdf(x_theory), 'r-', linewidth=2, 
                label='Expected |N(0,1)|')
    axes[1].axvline(x=2, color='orange', linestyle='--', linewidth=2, label='2σ')
    axes[1].set_xlabel('|Error| / Std', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('LOO-CV: Uncertainty Calibration', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 6: Hyperparameter optimization comparison
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Impact of Hyperparameter Optimization Restarts")
    print("=" * 70)
    
    n_restarts_list = [0, 5, 10, 20]
    restart_comparison = {}
    
    for n_restarts in n_restarts_list:
        print(f"\nTesting with {n_restarts} restarts...")
        emulator = GPEmulator(
            kernel_type='rbf',
            n_restarts_optimizer=n_restarts,
            random_state=42
        )
        start = time.time()
        emulator.fit(X_train, y_train, verbose=False)
        train_time = time.time() - start
        
        metrics = emulator.validate(X_test, y_test)
        restart_comparison[n_restarts] = {
            'metrics': metrics,
            'train_time': train_time,
            'log_likelihood': emulator.gp.log_marginal_likelihood_value_
        }
        
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Log-likelihood: {emulator.gp.log_marginal_likelihood_value_:.2f}")
        print(f"  RMSE: {metrics.rmse:.6f}")
    
    # Plot restart comparison
    fig5, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_restarts_arr = list(restart_comparison.keys())
    train_times = [restart_comparison[n]['train_time'] for n in n_restarts_arr]
    log_liks = [restart_comparison[n]['log_likelihood'] for n in n_restarts_arr]
    rmses = [restart_comparison[n]['metrics'].rmse for n in n_restarts_arr]
    
    axes[0].plot(n_restarts_arr, train_times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Restarts', fontsize=11)
    axes[0].set_ylabel('Training Time (s)', fontsize=11)
    axes[0].set_title('Training Time vs Restarts', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(n_restarts_arr, log_liks, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Number of Restarts', fontsize=11)
    axes[1].set_ylabel('Log Marginal Likelihood', fontsize=11)
    axes[1].set_title('Log-Likelihood vs Restarts', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(n_restarts_arr, rmses, 'o-', linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('Number of Restarts', fontsize=11)
    axes[2].set_ylabel('RMSE', fontsize=11)
    axes[2].set_title('Test RMSE vs Restarts', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)


class MultiOutputGPEmulator:
    """
    Multi-output GP emulator for multiple observables.
    Fits separate GP for each output dimension.
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        kernel_params: Optional[Dict] = None,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-10,
        random_state: Optional[int] = None
    ):
        """Initialize multi-output GP emulator."""
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.random_state = random_state
        
        self.emulators = []
        self.n_outputs = None
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        verbose: bool = True
    ):
        """
        Fit separate GP for each output.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Input design points
        Y : ndarray, shape (n_samples, n_outputs)
            Multiple outputs
        verbose : bool
            Print fitting progress
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        self.n_outputs = Y.shape[1]
        
        if verbose:
            print(f"Fitting {self.n_outputs} GP emulators...")
        
        self.emulators = []
        for i in range(self.n_outputs):
            if verbose:
                print(f"\n  Output {i+1}/{self.n_outputs}")
            
            emulator = GPEmulator(
                kernel_type=self.kernel_type,
                kernel_params=self.kernel_params,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=self.alpha,
                random_state=self.random_state
            )
            
            emulator.fit(X, Y[:, i], verbose=verbose)
            self.emulators.append(emulator)
        
        self.is_fitted = True
        
        if verbose:
            print(f"\n✓ All {self.n_outputs} emulators fitted successfully")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict all outputs.
        
        Returns:
        --------
        Y_pred : ndarray, shape (n_samples, n_outputs)
            Predictions for all outputs
        Y_std : ndarray, shape (n_samples, n_outputs)
            Standard deviations (if return_std=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit emulator before prediction")
        
        n_samples = len(X)
        Y_pred = np.zeros((n_samples, self.n_outputs))
        
        if return_std:
            Y_std = np.zeros((n_samples, self.n_outputs))
            
            for i, emulator in enumerate(self.emulators):
                Y_pred[:, i], Y_std[:, i] = emulator.predict(X, return_std=True)
            
            return Y_pred, Y_std
        else:
            for i, emulator in enumerate(self.emulators):
                Y_pred[:, i] = emulator.predict(X, return_std=False)
            
            return Y_pred
    
    def validate(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[int, EmulatorMetrics]:
        """
        Validate each output emulator.
        
        Returns:
        --------
        metrics_dict : dict
            Metrics for each output
        """
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        
        metrics_dict = {}
        
        if verbose:
            print("\nValidating multi-output emulator:")
            print("-" * 50)
        
        for i, emulator in enumerate(self.emulators):
            metrics = emulator.validate(X_test, Y_test[:, i])
            metrics_dict[i] = metrics
            
            if verbose:
                print(f"\nOutput {i+1}:")
                print(f"  RMSE: {metrics.rmse:.6f}")
                print(f"  R²:   {metrics.r2:.4f}")
                print(f"  Coverage: {metrics.coverage_95:.2%}")
        
        return metrics_dict
    
    def cross_validate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cv: int = 5,
        verbose: bool = True
    ) -> Dict[int, Dict]:
        """
        Cross-validate each output emulator.
        
        Returns:
        --------
        cv_results : dict
            CV results for each output
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        cv_results = {}
        
        if verbose:
            print(f"\nPerforming {cv}-fold CV for {self.n_outputs} outputs...")
        
        for i in range(self.n_outputs):
            if verbose:
                print(f"\n--- Output {i+1}/{self.n_outputs} ---")
            
            emulator = GPEmulator(
                kernel_type=self.kernel_type,
                kernel_params=self.kernel_params,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=self.alpha,
                random_state=self.random_state
            )
            
            cv_scores = emulator.cross_validate(X, Y[:, i], cv=cv, verbose=verbose)
            cv_results[i] = cv_scores
        
        return cv_results


def advanced_example():
    """Advanced example with PCA integration."""
    
    print("=" * 70)
    print("ADVANCED EXAMPLE: GP EMULATOR WITH PCA")
    print("=" * 70)
    
    # Generate high-dimensional output data
    np.random.seed(42)
    n_train = 80
    n_test = 20
    n_params = 5
    n_observables = 50  # High-dimensional output
    
    X_train = np.random.uniform(0, 1, size=(n_train, n_params))
    X_test = np.random.uniform(0, 1, size=(n_test, n_params))
    
    # Generate correlated outputs (simulating pt-differential observables)
    pt_bins = np.linspace(0.5, 4.0, n_observables)
    Y_train = np.zeros((n_train, n_observables))
    Y_test = np.zeros((n_test, n_observables))
    
    for i in range(n_train):
        base_amplitude = 0.1 * (1 - np.exp(-5 * X_train[i, 0]))
        Y_train[i] = base_amplitude * np.sqrt(pt_bins) * (1 + 0.1 * X_train[i, 1]) + \
                     np.random.normal(0, 0.01, n_observables)
    
    for i in range(n_test):
        base_amplitude = 0.1 * (1 - np.exp(-5 * X_test[i, 0]))
        Y_test[i] = base_amplitude * np.sqrt(pt_bins) * (1 + 0.1 * X_test[i, 1]) + \
                    np.random.normal(0, 0.01, n_observables)
    
    print(f"\nDataset:")
    print(f"  Training: {X_train.shape} → {Y_train.shape}")
    print(f"  Test: {X_test.shape} → {Y_test.shape}")
    
    # Apply PCA (simplified, assumes HeavyIonPCA is available)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=10)
    Y_train_pc = pca.fit_transform(Y_train)
    Y_test_pc = pca.transform(Y_test)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"\nPCA: {pca.n_components} PCs explain {explained_var:.2%} variance")
    
    # Fit multi-output GP emulator on PC scores
    print("\nFitting multi-output GP emulator on PC scores...")
    mo_emulator = MultiOutputGPEmulator(
        kernel_type='rbf',
        n_restarts_optimizer=5,
        random_state=42
    )
    
    mo_emulator.fit(X_train, Y_train_pc, verbose=True)
    
    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION ON PC SPACE")
    print("=" * 70)
    
    metrics_pc = mo_emulator.validate(X_test, Y_test_pc, verbose=True)
    
    # Transform back to observable space
    Y_pred_pc, Y_std_pc = mo_emulator.predict(X_test, return_std=True)
    Y_pred = pca.inverse_transform(Y_pred_pc)
    
    # Calculate metrics in observable space
    rmse_obs = np.sqrt(np.mean((Y_test - Y_pred)**2))
    r2_obs = r2_score(Y_test.ravel(), Y_pred.ravel())
    
    print("\n" + "=" * 70)
    print("VALIDATION ON OBSERVABLE SPACE")
    print("=" * 70)
    print(f"RMSE: {rmse_obs:.6f}")
    print(f"R²:   {r2_obs:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. PCA variance
    axes[0, 0].bar(range(1, 11), pca.explained_variance_ratio_, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Principal Component', fontsize=11)
    axes[0, 0].set_ylabel('Variance Ratio', fontsize=11)
    axes[0, 0].set_title('PCA: Explained Variance', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Predicted vs True (first PC)
    axes[0, 1].scatter(Y_test_pc[:, 0], Y_pred_pc[:, 0], alpha=0.6, s=50, 
                      edgecolors='black', linewidths=0.5)
    lim = [min(Y_test_pc[:, 0].min(), Y_pred_pc[:, 0].min()),
           max(Y_test_pc[:, 0].max(), Y_pred_pc[:, 0].max())]
    axes[0, 1].plot(lim, lim, 'r--', linewidth=2, label='Perfect')
    axes[0, 1].set_xlabel('True PC1', fontsize=11)
    axes[0, 1].set_ylabel('Predicted PC1', fontsize=11)
    axes[0, 1].set_title('First Principal Component', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Observable space: example profiles
    sample_idx = 0
    axes[1, 0].plot(pt_bins, Y_test[sample_idx], 'ko-', label='True', linewidth=2, markersize=6)
    axes[1, 0].plot(pt_bins, Y_pred[sample_idx], 'r^--', label='Predicted', linewidth=2, markersize=6)
    axes[1, 0].fill_between(pt_bins, 
                            Y_pred[sample_idx] - 2*np.mean(Y_std_pc[sample_idx]),
                            Y_pred[sample_idx] + 2*np.mean(Y_std_pc[sample_idx]),
                            alpha=0.3, color='red', label='±2σ')
    axes[1, 0].set_xlabel('pₜ [GeV]', fontsize=11)
    axes[1, 0].set_ylabel('Observable', fontsize=11)
    axes[1, 0].set_title(f'Example Profile (Test Sample {sample_idx})', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. RMSE per PC
    rmse_per_pc = [metrics_pc[i].rmse for i in range(10)]
    axes[1, 1].bar(range(1, 11), rmse_per_pc, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Principal Component', fontsize=11)
    axes[1, 1].set_ylabel('RMSE', fontsize=11)
    axes[1, 1].set_title('RMSE per Principal Component', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run basic examples
    example_usage()
    
    # Run advanced example
    print("\n\n")
    advanced_example()
