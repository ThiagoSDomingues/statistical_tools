# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gaussian_kde
from scipy.integrate import trapezoid
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass
import warnings
from itertools import combinations


@dataclass
class SensitivityResults:
    """Container for sensitivity analysis results."""
    method: str
    first_order: Optional[Dict[str, float]] = None
    total_order: Optional[Dict[str, float]] = None
    second_order: Optional[Dict[Tuple[str, str], float]] = None
    local_sensitivity: Optional[Dict[str, np.ndarray]] = None
    parameter_names: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"Sensitivity Analysis Results ({self.method})"]
        lines.append("=" * 60)
        
        if self.first_order:
            lines.append("\nFirst-order indices (main effects):")
            for param, value in self.first_order.items():
                lines.append(f"  {param:20s}: {value:8.4f}")
        
        if self.total_order:
            lines.append("\nTotal-order indices (main + interactions):")
            for param, value in self.total_order.items():
                lines.append(f"  {param:20s}: {value:8.4f}")
        
        if self.second_order:
            lines.append("\nSecond-order indices (interactions):")
            for (p1, p2), value in list(self.second_order.items())[:10]:
                lines.append(f"  {p1}-{p2:15s}: {value:8.4f}")
        
        return "\n".join(lines)


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for Gaussian Process emulators.
    Supports both global and local sensitivity analysis methods.
    """
    
    def __init__(
        self,
        emulator: Callable,
        bounds: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize sensitivity analyzer.
        
        Parameters:
        -----------
        emulator : callable
            Trained emulator that takes X (n_samples, n_params) and returns predictions
            Should return (mean, variance) or just mean
        bounds : np.ndarray, shape (n_params, 2)
            Parameter bounds [[lower1, upper1], [lower2, upper2], ...]
        parameter_names : list of str, optional
            Names of parameters
        seed : int, optional
            Random seed
        """
        self.emulator = emulator
        self.bounds = np.asarray(bounds)
        self.n_params = len(bounds)
        
        if parameter_names is None:
            self.parameter_names = [f"param_{i}" for i in range(self.n_params)]
        else:
            self.parameter_names = parameter_names
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        print(f"✓ Sensitivity analyzer initialized for {self.n_params} parameters")
    
    def _validate_emulator_output(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and standardize emulator output."""
        output = self.emulator(X_test)
        
        if isinstance(output, tuple):
            mean, var = output
            return np.asarray(mean), np.asarray(var)
        else:
            return np.asarray(output), None
    
    def _normalize_params(self, X: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _denormalize_params(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] to original bounds."""
        return X_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def sobol_indices(
        self,
        n_samples: int = 10000,
        calc_second_order: bool = True,
        confidence_level: float = 0.95
    ) -> SensitivityResults:
        """
        Calculate Sobol sensitivity indices (global variance-based SA).
        
        Uses Saltelli sampling scheme for efficient estimation.
        
        Parameters:
        -----------
        n_samples : int
            Number of base samples (total samples will be N*(2+n_params))
        calc_second_order : bool
            Whether to calculate second-order interaction indices
        confidence_level : float
            Confidence level for bootstrap intervals
        
        Returns:
        --------
        results : SensitivityResults
            First-order (Si), total-order (STi), and second-order (Sij) indices
        """
        print(f"Computing Sobol indices with {n_samples} samples...")
        
        # Generate Saltelli sample matrices
        # A: base sample matrix
        # B: resampled matrix
        # AB_i: matrix where column i is from A, others from B
        
        A = self._generate_samples(n_samples)
        B = self._generate_samples(n_samples)
        
        # Evaluate model at sample points
        f_A, _ = self._validate_emulator_output(A)
        f_B, _ = self._validate_emulator_output(B)
        
        # Handle multi-output case
        if f_A.ndim == 1:
            f_A = f_A.reshape(-1, 1)
            f_B = f_B.reshape(-1, 1)
        
        n_outputs = f_A.shape[1]
        
        # Initialize indices
        S1 = np.zeros((self.n_params, n_outputs))  # First-order
        ST = np.zeros((self.n_params, n_outputs))  # Total-order
        
        if calc_second_order:
            S2 = np.zeros((self.n_params, self.n_params, n_outputs))
        
        # Evaluate at AB_i matrices
        f_AB = []
        for i in range(self.n_params):
            AB_i = B.copy()
            AB_i[:, i] = A[:, i]
            f_ABi, _ = self._validate_emulator_output(AB_i)
            if f_ABi.ndim == 1:
                f_ABi = f_ABi.reshape(-1, 1)
            f_AB.append(f_ABi)
        
        # Calculate variance
        f_all = np.vstack([f_A, f_B] + f_AB)
        V = np.var(f_all, axis=0)
        
        # Calculate first-order and total-order indices
        for i in range(self.n_params):
            # First-order: V_i / V
            Vi = np.mean(f_B * (f_AB[i] - f_A), axis=0)
            S1[i] = Vi / (V + 1e-10)
            
            # Total-order: 1 - V_~i / V
            VTi = np.mean((f_A - f_AB[i])**2, axis=0) / 2
            ST[i] = VTi / (V + 1e-10)
        
        # Calculate second-order indices
        if calc_second_order:
            print("Computing second-order indices...")
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    # Create AB_ij matrix
                    AB_ij = B.copy()
                    AB_ij[:, i] = A[:, i]
                    AB_ij[:, j] = A[:, j]
                    f_ABij, _ = self._validate_emulator_output(AB_ij)
                    if f_ABij.ndim == 1:
                        f_ABij = f_ABij.reshape(-1, 1)
                    
                    # S_ij = V_ij / V - S_i - S_j
                    Vij = np.mean(f_B * (f_ABij - f_A), axis=0)
                    S2[i, j] = Vij / (V + 1e-10) - S1[i] - S1[j]
                    S2[j, i] = S2[i, j]
        
        # Average over outputs if multiple
        S1_mean = np.mean(S1, axis=1)
        ST_mean = np.mean(ST, axis=1)
        
        # Create result dictionaries
        first_order = {self.parameter_names[i]: S1_mean[i] for i in range(self.n_params)}
        total_order = {self.parameter_names[i]: ST_mean[i] for i in range(self.n_params)}
        
        second_order = None
        if calc_second_order:
            S2_mean = np.mean(S2, axis=2)
            second_order = {}
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    key = (self.parameter_names[i], self.parameter_names[j])
                    second_order[key] = S2_mean[i, j]
        
        results = SensitivityResults(
            method="Sobol",
            first_order=first_order,
            total_order=total_order,
            second_order=second_order,
            parameter_names=self.parameter_names,
            metadata={'n_samples': n_samples, 'n_outputs': n_outputs}
        )
        
        print("✓ Sobol indices computed")
        return results
    
    def morris_screening(
        self,
        n_trajectories: int = 100,
        n_levels: int = 4,
        optimal_trajectories: bool = True
    ) -> SensitivityResults:
        """
        Morris screening method (Elementary Effects).
        Efficient global SA for identifying important vs unimportant parameters.
        
        Parameters:
        -----------
        n_trajectories : int
            Number of Morris trajectories
        n_levels : int
            Number of levels for grid
        optimal_trajectories : bool
            Use optimal trajectory selection for better space coverage
        
        Returns:
        --------
        results : SensitivityResults
            Morris mu* (mean of absolute effects) and sigma (std of effects)
        """
        print(f"Computing Morris screening with {n_trajectories} trajectories...")
        
        delta = n_levels / (2 * (n_levels - 1))
        
        # Storage for elementary effects
        all_effects = [[] for _ in range(self.n_params)]
        
        for traj in range(n_trajectories):
            # Generate trajectory
            trajectory = self._generate_morris_trajectory(n_levels, delta)
            
            # Evaluate at each point in trajectory
            for i in range(self.n_params):
                X_before = trajectory[i]
                X_after = trajectory[i + 1]
                
                f_before, _ = self._validate_emulator_output(X_before.reshape(1, -1))
                f_after, _ = self._validate_emulator_output(X_after.reshape(1, -1))
                
                # Calculate elementary effect
                changed_param = np.where(X_before != X_after)[0][0]
                
                if f_before.ndim > 1:
                    f_before = f_before.mean()
                    f_after = f_after.mean()
                
                ee = (f_after - f_before) / delta
                all_effects[changed_param].append(ee)
        
        # Calculate Morris statistics
        mu_star = {}  # Mean of absolute effects
        sigma = {}    # Standard deviation of effects
        mu = {}       # Mean of effects (signed)
        
        for i in range(self.n_params):
            effects = np.array(all_effects[i])
            mu_star[self.parameter_names[i]] = np.mean(np.abs(effects))
            sigma[self.parameter_names[i]] = np.std(effects)
            mu[self.parameter_names[i]] = np.mean(effects)
        
        results = SensitivityResults(
            method="Morris",
            first_order=mu_star,  # Using mu_star as measure of importance
            total_order=None,
            parameter_names=self.parameter_names,
            metadata={
                'sigma': sigma,
                'mu': mu,
                'n_trajectories': n_trajectories
            }
        )
        
        print("✓ Morris screening completed")
        return results
    
    def delta_indices(
        self,
        n_samples: int = 5000,
        n_bootstrap: int = 100
    ) -> SensitivityResults:
        """
        Delta moment-independent sensitivity indices.
        Works well when variance-based methods fail (e.g., non-monotonic models).
        
        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples
        n_bootstrap : int
            Number of bootstrap resamples for uncertainty estimation
        
        Returns:
        --------
        results : SensitivityResults
            Delta indices for each parameter
        """
        print(f"Computing Delta indices with {n_samples} samples...")
        
        # Generate samples
        X = self._generate_samples(n_samples)
        f, _ = self._validate_emulator_output(X)
        
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        
        n_outputs = f.shape[1]
        
        delta = np.zeros((self.n_params, n_outputs))
        
        for i in range(self.n_params):
            # For each parameter, compute Delta index
            # Delta_i = 0.5 * E[|f(x) - f(x')|] where x'_i is resampled
            
            # Generate conditional samples
            X_cond = X.copy()
            X_cond[:, i] = self.rng.uniform(
                self.bounds[i, 0],
                self.bounds[i, 1],
                size=n_samples
            )
            
            f_cond, _ = self._validate_emulator_output(X_cond)
            if f_cond.ndim == 1:
                f_cond = f_cond.reshape(-1, 1)
            
            # Compute Delta
            delta[i] = 0.5 * np.mean(np.abs(f - f_cond), axis=0)
        
        # Average over outputs
        delta_mean = np.mean(delta, axis=1)
        
        # Normalize
        delta_normalized = delta_mean / (delta_mean.sum() + 1e-10)
        
        delta_dict = {
            self.parameter_names[i]: delta_normalized[i]
            for i in range(self.n_params)
        }
        
        results = SensitivityResults(
            method="Delta",
            first_order=delta_dict,
            parameter_names=self.parameter_names,
            metadata={'n_samples': n_samples}
        )
        
        print("✓ Delta indices computed")
        return results
    
    def local_sensitivity(
        self,
        X_base: np.ndarray,
        method: Literal['finite_diff', 'gradient', 'derivative'] = 'finite_diff',
        epsilon: float = 1e-4,
        normalize: bool = True
    ) -> SensitivityResults:
        """
        Local sensitivity analysis at specific point(s).
        Computes derivatives ∂f/∂xi at given locations.
        
        Parameters:
        -----------
        X_base : np.ndarray, shape (n_points, n_params) or (n_params,)
            Base point(s) for local SA
        method : str
            Method for computing derivatives
        epsilon : float
            Step size for finite differences
        normalize : bool
            Whether to normalize by parameter range
        
        Returns:
        --------
        results : SensitivityResults
            Local sensitivities at each point
        """
        print(f"Computing local sensitivity at {len(X_base) if X_base.ndim > 1 else 1} point(s)...")
        
        if X_base.ndim == 1:
            X_base = X_base.reshape(1, -1)
        
        n_points = len(X_base)
        
        # Storage for sensitivities
        local_sens = np.zeros((n_points, self.n_params))
        
        for point_idx in range(n_points):
            x_base = X_base[point_idx]
            f_base, _ = self._validate_emulator_output(x_base.reshape(1, -1))
            
            if f_base.ndim > 1:
                f_base = f_base.mean()
            
            for i in range(self.n_params):
                # Perturb parameter i
                x_perturb = x_base.copy()
                
                if method == 'finite_diff':
                    # Central difference
                    delta = epsilon * (self.bounds[i, 1] - self.bounds[i, 0])
                    
                    x_plus = x_base.copy()
                    x_plus[i] = min(x_base[i] + delta, self.bounds[i, 1])
                    
                    x_minus = x_base.copy()
                    x_minus[i] = max(x_base[i] - delta, self.bounds[i, 0])
                    
                    f_plus, _ = self._validate_emulator_output(x_plus.reshape(1, -1))
                    f_minus, _ = self._validate_emulator_output(x_minus.reshape(1, -1))
                    
                    if f_plus.ndim > 1:
                        f_plus = f_plus.mean()
                        f_minus = f_minus.mean()
                    
                    gradient = (f_plus - f_minus) / (2 * delta)
                
                else:  # forward difference
                    delta = epsilon * (self.bounds[i, 1] - self.bounds[i, 0])
                    x_perturb[i] = min(x_base[i] + delta, self.bounds[i, 1])
                    
                    f_perturb, _ = self._validate_emulator_output(x_perturb.reshape(1, -1))
                    if f_perturb.ndim > 1:
                        f_perturb = f_perturb.mean()
                    
                    gradient = (f_perturb - f_base) / delta
                
                if normalize:
                    # Normalized sensitivity: (∂f/∂xi) * (range_i / |f|)
                    param_range = self.bounds[i, 1] - self.bounds[i, 0]
                    gradient = gradient * param_range / (np.abs(f_base) + 1e-10)
                
                local_sens[point_idx, i] = gradient
        
        # Create result dictionary
        local_dict = {
            self.parameter_names[i]: local_sens[:, i]
            for i in range(self.n_params)
        }
        
        results = SensitivityResults(
            method="Local",
            local_sensitivity=local_dict,
            parameter_names=self.parameter_names,
            metadata={
                'n_points': n_points,
                'epsilon': epsilon,
                'normalized': normalize
            }
        )
        
        print("✓ Local sensitivity computed")
        return results
    
    def regional_sensitivity(
        self,
        regions: List[Dict[str, Tuple[float, float]]],
        n_samples_per_region: int = 1000,
        method: str = 'variance'
    ) -> Dict[str, SensitivityResults]:
        """
        Regional sensitivity analysis.
        Computes SA separately for different regions of parameter space.
        Useful for understanding how sensitivity varies across space.
        
        Parameters:
        -----------
        regions : list of dict
            Each dict defines a region: {param_name: (lower, upper), ...}
        n_samples_per_region : int
            Samples per region
        method : str
            SA method to use ('sobol', 'morris', 'delta')
        
        Returns:
        --------
        results : dict
            Dictionary mapping region index to SensitivityResults
        """
        print(f"Computing regional sensitivity for {len(regions)} regions...")
        
        results = {}
        
        for region_idx, region_bounds in enumerate(regions):
            print(f"\n  Region {region_idx + 1}/{len(regions)}...")
            
            # Create temporary bounds for this region
            temp_bounds = self.bounds.copy()
            for param_name, (lower, upper) in region_bounds.items():
                param_idx = self.parameter_names.index(param_name)
                temp_bounds[param_idx] = [lower, upper]
            
            # Temporarily modify bounds
            original_bounds = self.bounds.copy()
            self.bounds = temp_bounds
            
            # Compute SA for this region
            if method == 'sobol':
                sa_result = self.sobol_indices(
                    n_samples=n_samples_per_region,
                    calc_second_order=False
                )
            elif method == 'morris':
                sa_result = self.morris_screening(
                    n_trajectories=n_samples_per_region // 10
                )
            elif method == 'delta':
                sa_result = self.delta_indices(n_samples=n_samples_per_region)
            
            results[f"region_{region_idx}"] = sa_result
            
            # Restore original bounds
            self.bounds = original_bounds
        
        print("\n✓ Regional sensitivity analysis completed")
        return results
    
    def _generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate random samples from parameter space."""
        samples = self.rng.uniform(0, 1, size=(n_samples, self.n_params))
        return self._denormalize_params(samples)
    
    def _generate_morris_trajectory(self, n_levels: int, delta: float) -> np.ndarray:
        """Generate a Morris trajectory through parameter space."""
        grid = np.linspace(0, 1, n_levels)
        
        # Random starting point
        trajectory = [self.rng.choice(grid, size=self.n_params)]
        
        # Random order of parameters to vary
        param_order = self.rng.permutation(self.n_params)
        
        for param_idx in param_order:
            point = trajectory[-1].copy()
            
            # Change parameter by +/- delta
            current_val = point[param_idx]
            direction = self.rng.choice([-1, 1])
            new_val = current_val + direction * delta
            
            # Keep in bounds
            new_val = np.clip(new_val, 0, 1)
            point[param_idx] = new_val
            
            trajectory.append(point)
        
        trajectory = np.array(trajectory)
        return self._denormalize_params(trajectory)
    
    def plot_sobol_indices(
        self,
        results: SensitivityResults,
        figsize: Tuple[int, int] = (12, 6),
        show_second_order: bool = True
    ) -> plt.Figure:
        """Plot Sobol indices."""
        if results.method != "Sobol":
            raise ValueError("This method is for Sobol results only")
        
        fig, axes = plt.subplots(1, 2 if show_second_order and results.second_order else 1,
                                figsize=figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # First and total order indices
        params = list(results.first_order.keys())
        s1 = [results.first_order[p] for p in params]
        st = [results.total_order[p] for p in params]
        
        x = np.arange(len(params))
        width = 0.35
        
        axes[0].bar(x - width/2, s1, width, label='First-order (S₁)', alpha=0.8, color='steelblue')
        axes[0].bar(x + width/2, st, width, label='Total-order (Sₜ)', alpha=0.8, color='coral')
        axes[0].set_xlabel('Parameters', fontsize=11)
        axes[0].set_ylabel('Sensitivity Index', fontsize=11)
        axes[0].set_title('Sobol Indices', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(params, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].axhline(y=0, color='black', linewidth=0.8)
        
        # Second-order indices
        if show_second_order and results.second_order and len(axes) > 1:
            interactions = list(results.second_order.items())
            interactions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Top 10 interactions
            top_interactions = interactions[:min(10, len(interactions))]
            labels = [f"{p1}-{p2}" for (p1, p2), _ in top_interactions]
            values = [v for _, v in top_interactions]
            
            y_pos = np.arange(len(labels))
            axes[1].barh(y_pos, values, alpha=0.8, color='mediumpurple')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(labels)
            axes[1].set_xlabel('Interaction Index (S₁₂)', fontsize=11)
            axes[1].set_title('Parameter Interactions', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3, axis='x')
            axes[1].axvline(x=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def plot_morris_screening(
        self,
        results: SensitivityResults,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot Morris screening results (μ* vs σ).
        
        Parameters with high μ* are important.
        Parameters with high σ have non-linear or interactive effects.
        """
        if results.method != "Morris":
            raise ValueError("This method is for Morris results only")
        
        params = results.parameter_names
        mu_star = [results.first_order[p] for p in params]
        sigma = [results.metadata['sigma'][p] for p in params]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        scatter = ax.scatter(mu_star, sigma, s=100, alpha=0.6, c=range(len(params)),
                           cmap='viridis', edgecolors='black', linewidth=1.5)
        
        # Label points
        for i, param in enumerate(params):
            ax.annotate(param, (mu_star[i], sigma[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        ax.set_xlabel('μ* (Mean of Absolute Effects)', fontsize=12, fontweight='bold')
        ax.set_ylabel('σ (Standard Deviation of Effects)', fontsize=12, fontweight='bold')
        ax.set_title('Morris Screening: Parameter Importance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add interpretation regions
        max_mu = max(mu_star)
        max_sigma = max(sigma)
        
        ax.axvline(x=max_mu*0.3, color='red', linestyle='--', alpha=0.3, linewidth=2)
        ax.axhline(y=max_sigma*0.3, color='red', linestyle='--', alpha=0.3, linewidth=2)
        
        # Add text annotations for regions
        ax.text(0.98, 0.98, 'High importance\n+ interactions',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_local_sensitivity(
        self,
        results: SensitivityResults,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Plot local sensitivity analysis results."""
        if results.method != "Local":
            raise ValueError("This method is for local sensitivity results only")
        
        n_points = results.metadata['n_points']
        params = results.parameter_names
        
        # Prepare data
        sens_matrix = np.array([results.local_sensitivity[p] for p in params])
        
        if n_points == 1:
            # Bar plot for single point
            fig, ax = plt.subplots(figsize=(10, 6))
            values = sens_matrix[:, 0]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            y_pos = np.arange(len(params))
            ax.barh(y_pos, values, alpha=0.7, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('Normalized Local Sensitivity', fontsize=11)
            ax.set_title('Local Sensitivity Analysis', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=1)
            ax.grid(alpha=0.3, axis='x')
        else:
            # Heatmap for multiple points
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(sens_matrix, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(sens_matrix)),
                          vmax=np.max(np.abs(sens_matrix)))
            
            ax.set_yticks(np.arange(len(params)))
            ax.set_yticklabels(params)
            ax.set_xlabel('Evaluation Point', fontsize=11)
            ax.set_ylabel('Parameter', fontsize=11)
            ax.set_title('Local Sensitivity Across Points', fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Normalized Sensitivity')
        
        plt.tight_layout()
        return fig
    
    def compare_methods(
        self,
        methods: List[str] = ['sobol', 'morris', 'delta'],
        n_samples: int = 5000,
        figsize: Tuple[int, int] = (14, 6)
    ) -> Dict[str, SensitivityResults]:
        """
        Compare different sensitivity analysis methods.
        
        Parameters:
        -----------
        methods : list of str
            SA methods to compare ('sobol', 'morris', 'delta')
        n_samples : int
            Number of samples for each method
        figsize : tuple
            Figure size for comparison plot
        
        Returns:
        --------
        results : dict
            Dictionary mapping method names to results
        """
        print("Comparing sensitivity analysis methods...")
        print("=" * 60)
        
        all_results = {}
        
        for method in methods:
            print(f"\nRunning {method.upper()} method...")
            if method == 'sobol':
                result = self.sobol_indices(n_samples=n_samples, calc_second_order=False)
            elif method == 'morris':
                result = self.morris_screening(n_trajectories=n_samples // 10)
            elif method == 'delta':
                result = self.delta_indices(n_samples=n_samples)
            else:
                warnings.warn(f"Unknown method '{method}', skipping...")
                continue
            
            all_results[method] = result
        
        # Plot comparison
        self._plot_method_comparison(all_results, figsize)
        
        return all_results
    
    def _plot_method_comparison(
        self,
        results: Dict[str, SensitivityResults],
        figsize: Tuple[int, int]
    ) -> plt.Figure:
        """Plot comparison of different SA methods."""
        n_methods = len(results)
        params = self.parameter_names
        
        fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, result) in enumerate(results.items()):
            indices = [result.first_order[p] for p in params]
            
            y_pos = np.arange(len(params))
            axes[idx].barh(y_pos, indices, alpha=0.7)
            axes[idx].set_xlabel('Sensitivity Index', fontsize=10)
            axes[idx].set_title(method_name.upper(), fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='x')
            
            if idx == 0:
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(params)
        
        plt.suptitle('Sensitivity Analysis Method Comparison', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig


class TimeEvolvingSensitivity:
    """
    Sensitivity analysis for time-evolving systems.
    Useful for heavy-ion collisions where observables evolve over time.
    """
    
    def __init__(
        self,
        emulator: Callable,
        bounds: np.ndarray,
        time_points: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize time-evolving sensitivity analyzer.
        
        Parameters:
        -----------
        emulator : callable
            Emulator that returns time series: f(X) -> (n_samples, n_time_points)
        bounds : np.ndarray
            Parameter bounds
        time_points : np.ndarray
            Time points where outputs are evaluated
        parameter_names : list, optional
            Parameter names
        seed : int, optional
            Random seed
        """
        self.base_analyzer = SensitivityAnalyzer(emulator, bounds, parameter_names, seed)
        self.time_points = np.asarray(time_points)
        self.n_time = len(time_points)
        
        print(f"✓ Time-evolving SA initialized with {self.n_time} time points")
    
    def temporal_sobol(
        self,
        n_samples: int = 5000
    ) -> Dict[str, np.ndarray]:
        """
        Compute Sobol indices at each time point.
        
        Returns:
        --------
        results : dict
            Dictionary with 'first_order' and 'total_order' containing
            arrays of shape (n_params, n_time)
        """
        print("Computing time-evolving Sobol indices...")
        
        n_params = self.base_analyzer.n_params
        S1_temporal = np.zeros((n_params, self.n_time))
        ST_temporal = np.zeros((n_params, self.n_time))
        
        # Generate sample matrices once
        A = self.base_analyzer._generate_samples(n_samples)
        B = self.base_analyzer._generate_samples(n_samples)
        
        # Evaluate emulator (returns time series)
        f_A = self.base_analyzer.emulator(A)  # (n_samples, n_time)
        f_B = self.base_analyzer.emulator(B)
        
        # For each time point
        for t_idx in range(self.n_time):
            f_A_t = f_A[:, t_idx]
            f_B_t = f_B[:, t_idx]
            
            V = np.var(np.concatenate([f_A_t, f_B_t]))
            
            for i in range(n_params):
                # Create AB_i matrix
                AB_i = B.copy()
                AB_i[:, i] = A[:, i]
                f_ABi = self.base_analyzer.emulator(AB_i)[:, t_idx]
                
                # First-order
                Vi = np.mean(f_B_t * (f_ABi - f_A_t))
                S1_temporal[i, t_idx] = Vi / (V + 1e-10)
                
                # Total-order
                VTi = np.mean((f_A_t - f_ABi)**2) / 2
                ST_temporal[i, t_idx] = VTi / (V + 1e-10)
        
        results = {
            'first_order': S1_temporal,
            'total_order': ST_temporal,
            'time_points': self.time_points,
            'parameter_names': self.base_analyzer.parameter_names
        }
        
        print("✓ Temporal Sobol indices computed")
        return results
    
    def plot_temporal_sensitivity(
        self,
        results: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """Plot time-evolving sensitivity indices."""
        S1 = results['first_order']
        ST = results['total_order']
        time = results['time_points']
        params = results['parameter_names']
        
        n_params = len(params)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # First-order indices
        for i in range(n_params):
            axes[0].plot(time, S1[i], label=params[i], linewidth=2, marker='o', markersize=4)
        axes[0].set_ylabel('First-order Index (S₁)', fontsize=11)
        axes[0].set_title('Time Evolution of Sensitivity Indices', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best', ncol=2)
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim([-0.05, 1.05])
        
        # Total-order indices
        for i in range(n_params):
            axes[1].plot(time, ST[i], label=params[i], linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Time', fontsize=11)
        axes[1].set_ylabel('Total-order Index (Sₜ)', fontsize=11)
        axes[1].legend(loc='best', ncol=2)
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        return fig


# Example usage for heavy-ion collisions
def example_usage():
    """Comprehensive example of sensitivity analysis for HIC emulators."""
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS FOR HEAVY-ION COLLISION EMULATORS")
    print("=" * 70)
    
    # Define mock emulator (replace with your trained GP emulator)
    def mock_emulator(X):
        """
        Mock emulator for demonstration.
        X: (n_samples, n_params)
        Returns: (n_samples,) or (n_samples, n_outputs)
        """
        # Nonlinear function mimicking physical response
        f = (np.sin(X[:, 0] * 3) * 0.5 + 
             X[:, 1]**2 * 0.3 + 
             X[:, 2] * X[:, 3] * 0.4 +
             np.exp(-X[:, 4]) * 0.2)
        return f
    
    # Parameter bounds (typical HIC parameters)
    # [eta/s, zeta/s, tau_0, T_switch, normalization]
    bounds = np.array([
        [0.05, 0.25],   # eta/s
        [0.01, 0.10],   # zeta/s  
        [0.2, 1.0],     # tau_0 [fm/c]
        [0.135, 0.165], # T_switch [GeV]
        [0.8, 1.2]      # normalization
    ])
    
    param_names = ['η/s', 'ζ/s', 'τ₀', 'T_sw', 'norm']
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(
        emulator=mock_emulator,
        bounds=bounds,
        parameter_names=param_names,
        seed=42
    )
    
    print("\n" + "=" * 70)
    print("1. SOBOL INDICES (Global Variance-Based SA)")
    print("=" * 70)
    
    sobol_results = analyzer.sobol_indices(n_samples=5000, calc_second_order=True)
    print(sobol_results.summary())
    
    fig1 = analyzer.plot_sobol_indices(sobol_results)
    plt.show()
    
    print("\n" + "=" * 70)
    print("2. MORRIS SCREENING (Efficient Global SA)")
    print("=" * 70)
    
    morris_results = analyzer.morris_screening(n_trajectories=100)
    print(morris_results.summary())
    
    fig2 = analyzer.plot_morris_screening(morris_results)
    plt.show()
    
    print("\n" + "=" * 70)
    print("3. DELTA INDICES (Moment-Independent SA)")
    print("=" * 70)
    
    delta_results = analyzer.delta_indices(n_samples=3000)
    print(delta_results.summary())
    
    print("\n" + "=" * 70)
    print("4. LOCAL SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Evaluate at central point
    X_center = np.mean(bounds, axis=1)
    local_results = analyzer.local_sensitivity(
        X_base=X_center,
        method='finite_diff',
        normalize=True
    )
    
    fig3 = analyzer.plot_local_sensitivity(local_results)
    plt.show()
    
    print("\n" + "=" * 70)
    print("5. COMPARING MULTIPLE METHODS")
    print("=" * 70)
    
    comparison = analyzer.compare_methods(
        methods=['sobol', 'morris', 'delta'],
        n_samples=3000
    )
    
    print("\n" + "=" * 70)
    print("6. TIME-EVOLVING SENSITIVITY (for temporal observables)")
    print("=" * 70)
    
    # Mock time-evolving emulator
    def time_emulator(X):
        """Returns time series for each sample."""
        n_samples = len(X)
        time = np.linspace(0, 10, 20)
        output = np.zeros((n_samples, len(time)))
        
        for i in range(n_samples):
            output[i] = np.sin(X[i, 0] * time) * np.exp(-time * X[i, 1])
        
        return output
    
    time_analyzer = TimeEvolvingSensitivity(
        emulator=time_emulator,
        bounds=bounds,
        time_points=np.linspace(0, 10, 20),
        parameter_names=param_names,
        seed=42
    )
    
    temporal_results = time_analyzer.temporal_sobol(n_samples=2000)
    fig4 = time_analyzer.plot_temporal_sensitivity(temporal_results)
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
