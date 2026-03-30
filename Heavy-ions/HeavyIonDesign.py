# Author: OptimusThi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Dict, List, Tuple, Callable, Union, Literal
import warnings
from dataclasses import dataclass


@dataclass
class ParameterBounds:
    """Define parameter bounds for design space."""
    lower: np.ndarray
    upper: np.ndarray
    
    def __post_init__(self):
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)
        if len(self.lower) != len(self.upper):
            raise ValueError("Lower and upper bounds must have same length")
        if np.any(self.lower >= self.upper):
            raise ValueError("Lower bounds must be strictly less than upper bounds")
        self.dim = len(self.lower)
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize points to [0, 1]^d."""
        return (X - self.lower) / (self.upper - self.lower)
    
    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1]^d to original bounds."""
        return X_norm * (self.upper - self.lower) + self.lower
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """Clip points to bounds."""
        return np.clip(X, self.lower, self.upper)


class DesignGenerator:
    """
    Flexible design point generator for Gaussian Process emulators.
    Supports multiple sampling strategies including adaptive methods.
    """
    
    def __init__(
        self,
        bounds: Union[ParameterBounds, Tuple[List, List]],
        seed: Optional[int] = None
    ):
        """
        Initialize design generator.
        
        Parameters:
        -----------
        bounds : ParameterBounds or tuple of (lower, upper)
            Parameter space bounds
        seed : int, optional
            Random seed for reproducibility
        """
        if isinstance(bounds, tuple):
            self.bounds = ParameterBounds(bounds[0], bounds[1])
        else:
            self.bounds = bounds
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.dim = self.bounds.dim
        
        print(f"✓ Design generator initialized: {self.dim}D parameter space")
    
    def generate(
        self,
        n_points: int,
        method: Literal[
            'random', 'lhs', 'sobol', 'halton', 'maximin', 
            'minimax', 'uniform_grid', 'sphere_packing'
        ] = 'lhs',
        **kwargs
    ) -> np.ndarray:
        """
        Generate design points.
        
        Parameters:
        -----------
        n_points : int
            Number of design points
        method : str
            Sampling method
        **kwargs : additional method-specific parameters
        
        Returns:
        --------
        X : np.ndarray, shape (n_points, dim)
            Design points in original parameter space
        """
        method_map = {
            'random': self._random_sampling,
            'lhs': self._latin_hypercube,
            'sobol': self._sobol_sequence,
            'halton': self._halton_sequence,
            'maximin': self._maximin_lhs,
            'minimax': self._minimax_distance,
            'uniform_grid': self._uniform_grid,
            'sphere_packing': self._sphere_packing
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method '{method}'. Available: {list(method_map.keys())}")
        
        print(f"Generating {n_points} design points using '{method}' method...")
        X = method_map[method](n_points, **kwargs)
        
        # Validate output
        assert X.shape == (n_points, self.dim), f"Shape mismatch: {X.shape}"
        assert np.all(X >= self.bounds.lower) and np.all(X <= self.bounds.upper), \
            "Points outside bounds"
        
        print(f"✓ Generated {n_points} design points")
        return X
    
    def _random_sampling(self, n_points: int, **kwargs) -> np.ndarray:
        """Pure random sampling (Monte Carlo)."""
        X_norm = self.rng.rand(n_points, self.dim)
        return self.bounds.denormalize(X_norm)
    
    def _latin_hypercube(
        self,
        n_points: int,
        criterion: str = 'maximin',
        iterations: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Latin Hypercube Sampling with optimization.
        
        Parameters:
        -----------
        criterion : str
            'center', 'maximin', 'correlation', 'ratio'
        iterations : int
            Number of optimization iterations
        """
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
        
        if criterion == 'center':
            X_norm = sampler.random(n=n_points)
        else:
            # Generate multiple candidates and select best
            n_candidates = min(10, iterations // 100 + 1)
            best_score = -np.inf
            best_X = None
            
            for _ in range(n_candidates):
                X_candidate = sampler.random(n=n_points)
                
                if criterion == 'maximin':
                    score = self._maximin_score(X_candidate)
                elif criterion == 'correlation':
                    score = -np.abs(np.corrcoef(X_candidate.T)).sum()
                elif criterion == 'ratio':
                    score = self._coverage_score(X_candidate)
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                if score > best_score:
                    best_score = score
                    best_X = X_candidate
            
            X_norm = best_X
        
        return self.bounds.denormalize(X_norm)
    
    def _sobol_sequence(self, n_points: int, scramble: bool = True, **kwargs) -> np.ndarray:
        """Sobol quasi-random sequence."""
        sampler = qmc.Sobol(d=self.dim, scramble=scramble, seed=self.seed)
        X_norm = sampler.random(n=n_points)
        return self.bounds.denormalize(X_norm)
    
    def _halton_sequence(self, n_points: int, scramble: bool = True, **kwargs) -> np.ndarray:
        """Halton quasi-random sequence."""
        sampler = qmc.Halton(d=self.dim, scramble=scramble, seed=self.seed)
        X_norm = sampler.random(n=n_points)
        return self.bounds.denormalize(X_norm)
    
    def _maximin_lhs(
        self,
        n_points: int,
        iterations: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Latin Hypercube with Maximin distance criterion.
        Maximizes minimum distance between points.
        """
        # Start with LHS
        X_norm = qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=n_points)
        
        # Optimize using coordinate exchange
        best_score = self._maximin_score(X_norm)
        
        for _ in range(iterations):
            # Random coordinate exchange
            i = self.rng.randint(n_points)
            j = self.rng.randint(self.dim)
            old_val = X_norm[i, j]
            X_norm[i, j] = self.rng.rand()
            
            new_score = self._maximin_score(X_norm)
            
            if new_score > best_score:
                best_score = new_score
            else:
                X_norm[i, j] = old_val  # Revert
        
        return self.bounds.denormalize(X_norm)
    
    def _minimax_distance(
        self,
        n_points: int,
        iterations: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Minimax design: minimizes maximum distance from any point in space
        to nearest design point. Good for space-filling.
        """
        def objective(X_flat):
            X = X_flat.reshape(n_points, self.dim)
            X = np.clip(X, 0, 1)
            
            # Sample test points
            n_test = 1000
            X_test = self.rng.rand(n_test, self.dim)
            
            # Find minimum distance from each test point to design points
            min_dists = cdist(X_test, X).min(axis=1)
            
            # Minimize the maximum of these minimum distances
            return -min_dists.min()
        
        # Initialize with LHS
        X0 = qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=n_points)
        
        result = minimize(
            objective,
            x0=X0.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * (n_points * self.dim),
            options={'maxiter': iterations}
        )
        
        X_norm = result.x.reshape(n_points, self.dim)
        return self.bounds.denormalize(X_norm)
    
    def _uniform_grid(self, n_points: int, **kwargs) -> np.ndarray:
        """Uniform grid sampling."""
        # Calculate points per dimension
        points_per_dim = int(np.ceil(n_points ** (1 / self.dim)))
        actual_points = points_per_dim ** self.dim
        
        if actual_points != n_points:
            warnings.warn(
                f"Adjusted n_points from {n_points} to {actual_points} "
                f"({points_per_dim}^{self.dim}) for uniform grid"
            )
        
        # Create grid
        axes = [np.linspace(0, 1, points_per_dim) for _ in range(self.dim)]
        grid = np.meshgrid(*axes, indexing='ij')
        X_norm = np.column_stack([g.ravel() for g in grid])
        
        return self.bounds.denormalize(X_norm)
    
    def _sphere_packing(
        self,
        n_points: int,
        iterations: int = 500,
        alpha: float = 0.1,
        **kwargs
    ) -> np.ndarray:
        """
        Sphere packing algorithm for space-filling design.
        Points repel each other like charged particles.
        """
        # Initialize with LHS
        X_norm = qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=n_points)
        
        for iteration in range(iterations):
            # Calculate pairwise distances
            dists = squareform(pdist(X_norm))
            np.fill_diagonal(dists, np.inf)
            
            # Calculate repulsion forces
            forces = np.zeros_like(X_norm)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    diff = X_norm[i] - X_norm[j]
                    dist = dists[i, j]
                    force = diff / (dist ** 3 + 1e-10)
                    forces[i] += force
                    forces[j] -= force
            
            # Update positions
            step_size = alpha * (1 - iteration / iterations)
            X_norm += step_size * forces
            X_norm = np.clip(X_norm, 0, 1)
        
        return self.bounds.denormalize(X_norm)
    
    def _maximin_score(self, X: np.ndarray) -> float:
        """Calculate maximin score (minimum pairwise distance)."""
        if len(X) < 2:
            return 0.0
        dists = pdist(X)
        return np.min(dists) if len(dists) > 0 else 0.0
    
    def _coverage_score(self, X: np.ndarray) -> float:
        """Calculate space-filling coverage score."""
        # Ratio of filled volume to total volume
        dists = pdist(X)
        return np.mean(dists)
    
    def plot_design(
        self,
        X: np.ndarray,
        dims: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (12, 10),
        show_projections: bool = True
    ) -> plt.Figure:
        """
        Visualize design points.
        
        Parameters:
        -----------
        X : np.ndarray
            Design points to visualize
        dims : list, optional
            Dimensions to plot (defaults to first 3 or all if dim <= 3)
        figsize : tuple
            Figure size
        show_projections : bool
            Show 2D projections
        """
        if dims is None:
            dims = list(range(min(3, self.dim)))
        
        n_dims = len(dims)
        
        if n_dims == 1:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.scatter(X[:, dims[0]], np.zeros_like(X[:, dims[0]]), alpha=0.6, s=50)
            ax.set_xlabel(f"Parameter {dims[0]}")
            ax.set_yticks([])
            
        elif n_dims == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(X[:, dims[0]], X[:, dims[1]], alpha=0.6, s=50)
            ax.set_xlabel(f"Parameter {dims[0]}")
            ax.set_ylabel(f"Parameter {dims[1]}")
            ax.grid(alpha=0.3)
            
        elif n_dims >= 3:
            if show_projections:
                fig = plt.figure(figsize=figsize)
                
                # 3D plot
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                ax1.scatter(
                    X[:, dims[0]], X[:, dims[1]], X[:, dims[2]],
                    alpha=0.6, s=30
                )
                ax1.set_xlabel(f"Param {dims[0]}")
                ax1.set_ylabel(f"Param {dims[1]}")
                ax1.set_zlabel(f"Param {dims[2]}")
                ax1.set_title("3D View")
                
                # 2D projections
                projections = [
                    (dims[0], dims[1], (2, 2, 2)),
                    (dims[0], dims[2], (2, 2, 3)),
                    (dims[1], dims[2], (2, 2, 4))
                ]
                
                for d1, d2, pos in projections:
                    ax = fig.add_subplot(*pos)
                    ax.scatter(X[:, d1], X[:, d2], alpha=0.6, s=30)
                    ax.set_xlabel(f"Param {d1}")
                    ax.set_ylabel(f"Param {d2}")
                    ax.grid(alpha=0.3)
                    ax.set_title(f"Projection {d1}-{d2}")
            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    X[:, dims[0]], X[:, dims[1]], X[:, dims[2]],
                    alpha=0.6, s=50
                )
                ax.set_xlabel(f"Parameter {dims[0]}")
                ax.set_ylabel(f"Parameter {dims[1]}")
                ax.set_zlabel(f"Parameter {dims[2]}")
        
        plt.tight_layout()
        return fig
    
    def compare_designs(
        self,
        n_points: int,
        methods: List[str],
        metric: Literal['maximin', 'coverage', 'discrepancy'] = 'maximin'
    ) -> Dict:
        """
        Compare different design methods.
        
        Returns:
        --------
        results : dict
            Dictionary with methods as keys and scores as values
        """
        results = {}
        designs = {}
        
        for method in methods:
            X = self.generate(n_points, method=method)
            X_norm = self.bounds.normalize(X)
            
            if metric == 'maximin':
                score = self._maximin_score(X_norm)
            elif metric == 'coverage':
                score = self._coverage_score(X_norm)
            elif metric == 'discrepancy':
                score = -qmc.discrepancy(X_norm)  # Lower is better
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results[method] = score
            designs[method] = X
            print(f"{method:20s}: {metric} = {score:.6f}")
        
        return results, designs


class AdaptiveDesignGenerator(DesignGenerator):
    """
    Adaptive sampling strategies that use information from existing data
    to select new design points intelligently.
    """
    
    def adaptive_sampling(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: Optional[np.ndarray] = None,
        strategy: Literal[
            'uncertainty', 'variance', 'expected_improvement',
            'distance_based', 'gradient_based'
        ] = 'distance_based',
        batch_size: int = 10,
        **kwargs
    ) -> np.ndarray:
        """
        Generate new design points adaptively based on existing data.
        
        Parameters:
        -----------
        n_points : int
            Total number of new points to generate
        X_existing : np.ndarray
            Existing design points
        y_existing : np.ndarray, optional
            Observed outputs at existing points (required for some strategies)
        strategy : str
            Adaptive sampling strategy
        batch_size : int
            Number of points to add per iteration
        
        Returns:
        --------
        X_new : np.ndarray
            New design points
        """
        strategy_map = {
            'distance_based': self._distance_based_adaptive,
            'variance': self._variance_based_adaptive,
            'uncertainty': self._uncertainty_sampling,
            'expected_improvement': self._expected_improvement_sampling,
            'gradient_based': self._gradient_based_adaptive
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(strategy_map.keys())}")
        
        print(f"Adaptive sampling: {strategy} strategy, {n_points} new points")
        
        X_new = strategy_map[strategy](
            n_points, X_existing, y_existing, batch_size, **kwargs
        )
        
        return X_new
    
    def _distance_based_adaptive(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: Optional[np.ndarray],
        batch_size: int,
        **kwargs
    ) -> np.ndarray:
        """
        Select points that maximize minimum distance to existing points.
        Good for space-filling when no output information available.
        """
        X_norm_existing = self.bounds.normalize(X_existing)
        X_new = []
        
        n_batches = int(np.ceil(n_points / batch_size))
        
        for batch in range(n_batches):
            points_this_batch = min(batch_size, n_points - len(X_new))
            
            # Generate candidates
            n_candidates = 500
            X_candidates = self.rng.rand(n_candidates, self.dim)
            
            # Calculate minimum distance to existing points
            X_all = np.vstack([X_norm_existing, X_new]) if X_new else X_norm_existing
            min_dists = cdist(X_candidates, X_all).min(axis=1)
            
            # Select points with largest minimum distance
            best_indices = np.argsort(min_dists)[-points_this_batch:]
            X_new.extend(X_candidates[best_indices])
        
        X_new = np.array(X_new)[:n_points]
        return self.bounds.denormalize(X_new)
    
    def _variance_based_adaptive(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: np.ndarray,
        batch_size: int,
        length_scale: float = 0.1,
        **kwargs
    ) -> np.ndarray:
        """
        Select points with highest predicted variance using GP approximation.
        Requires output data.
        """
        if y_existing is None:
            raise ValueError("variance_based strategy requires y_existing")
        
        X_norm_existing = self.bounds.normalize(X_existing)
        X_new = []
        
        n_batches = int(np.ceil(n_points / batch_size))
        
        for batch in range(n_batches):
            points_this_batch = min(batch_size, n_points - len(X_new))
            
            # Generate candidates
            n_candidates = 500
            X_candidates = self.rng.rand(n_candidates, self.dim)
            
            # Simple GP variance approximation (kernel-based)
            X_all = np.vstack([X_norm_existing, X_new]) if X_new else X_norm_existing
            
            # Calculate predicted variance (inversely proportional to proximity)
            dists = cdist(X_candidates, X_all)
            kernel_vals = np.exp(-dists ** 2 / (2 * length_scale ** 2))
            variance = 1 - kernel_vals.max(axis=1)
            
            # Select high variance points
            best_indices = np.argsort(variance)[-points_this_batch:]
            X_new.extend(X_candidates[best_indices])
        
        X_new = np.array(X_new)[:n_points]
        return self.bounds.denormalize(X_new)
    
    def _uncertainty_sampling(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: np.ndarray,
        batch_size: int,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from regions with high uncertainty in output.
        Similar to variance-based but considers output magnitude.
        """
        if y_existing is None:
            raise ValueError("uncertainty_sampling requires y_existing")
        
        # Check if y is multi-dimensional (e.g., from PCA)
        if y_existing.ndim == 1:
            y_existing = y_existing.reshape(-1, 1)
        
        y_std = np.std(y_existing, axis=0)
        
        return self._variance_based_adaptive(
            n_points, X_existing, y_existing, batch_size,
            length_scale=0.1 * np.mean(y_std), **kwargs
        )
    
    def _expected_improvement_sampling(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: np.ndarray,
        batch_size: int,
        **kwargs
    ) -> np.ndarray:
        """
        Sample points with high expected improvement (Bayesian optimization style).
        """
        if y_existing is None:
            raise ValueError("expected_improvement requires y_existing")
        
        # For multi-output, use mean of outputs
        if y_existing.ndim > 1:
            y_flat = y_existing.mean(axis=1)
        else:
            y_flat = y_existing
        
        y_best = np.max(y_flat)
        
        X_norm_existing = self.bounds.normalize(X_existing)
        X_new = []
        
        n_batches = int(np.ceil(n_points / batch_size))
        
        for batch in range(n_batches):
            points_this_batch = min(batch_size, n_points - len(X_new))
            
            # Generate candidates
            n_candidates = 500
            X_candidates = self.rng.rand(n_candidates, self.dim)
            
            # Simple EI approximation
            X_all = np.vstack([X_norm_existing, X_new]) if X_new else X_norm_existing
            dists = cdist(X_candidates, X_all)
            
            # Predict mean and variance
            weights = np.exp(-dists ** 2 / 0.02)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
            y_pred = weights @ y_flat
            
            variance = 1 - weights.max(axis=1)
            
            # Expected improvement
            improvement = np.maximum(y_pred - y_best, 0)
            ei = improvement * variance
            
            best_indices = np.argsort(ei)[-points_this_batch:]
            X_new.extend(X_candidates[best_indices])
        
        X_new = np.array(X_new)[:n_points]
        return self.bounds.denormalize(X_new)
    
    def _gradient_based_adaptive(
        self,
        n_points: int,
        X_existing: np.ndarray,
        y_existing: np.ndarray,
        batch_size: int,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from regions with high gradient (rapid output changes).
        """
        if y_existing is None:
            raise ValueError("gradient_based requires y_existing")
        
        X_norm_existing = self.bounds.normalize(X_existing)
        
        # Estimate gradients using finite differences
        if y_existing.ndim == 1:
            y_existing = y_existing.reshape(-1, 1)
        
        # Find nearest neighbors and estimate local gradient
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(X_existing)))
        nn.fit(X_norm_existing)
        
        # Sample candidates and evaluate gradient
        n_candidates = 500
        X_candidates = self.rng.rand(n_candidates, self.dim)
        
        distances, indices = nn.kneighbors(X_candidates)
        
        # Estimate gradient magnitude
        gradient_mag = np.zeros(n_candidates)
        for i in range(n_candidates):
            neighbors = indices[i]
            y_neighbors = y_existing[neighbors]
            gradient_mag[i] = np.std(y_neighbors)
        
        # Select high gradient regions
        best_indices = np.argsort(gradient_mag)[-n_points:]
        X_new = X_candidates[best_indices]
        
        return self.bounds.denormalize(X_new)


def example_usage():
    """Comprehensive examples of design generation."""
    
    # Define parameter space (e.g., for heavy-ion collision model)
    lower_bounds = [0.01, 0.5, 0.1, 1.0, 0.0]
    upper_bounds = [0.5, 2.0, 1.0, 5.0, 1.0]
    bounds = ParameterBounds(lower_bounds, upper_bounds)
    
    print("=" * 70)
    print("EXAMPLE 1: Comparing Different Design Methods")
    print("=" * 70)
    
    generator = DesignGenerator(bounds, seed=42)
    
    # Compare methods
    methods = ['random', 'lhs', 'sobol', 'halton', 'maximin']
    results, designs = generator.compare_designs(
        n_points=50,
        methods=methods,
        metric='maximin'
    )
    
    # Visualize best designs
    best_method = max(results, key=results.get)
    print(f"\nBest method: {best_method}")
    
    fig = generator.plot_design(designs[best_method], dims=[0, 1, 2])
    plt.suptitle(f"Best Design: {best_method} (maximin={results[best_method]:.4f})")
    plt.show()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Adaptive Sampling")
    print("=" * 70)
    
    # Initial design
    X_initial = generator.generate(30, method='lhs')
    
    # Simulate some outputs (e.g., from expensive simulation)
    y_initial = np.sum(np.sin(X_initial * 2 * np.pi), axis=1)
    
    # Adaptive generator
    adaptive_gen = AdaptiveDesignGenerator(bounds, seed=42)
    
    # Generate adaptive points
    X_adaptive = adaptive_gen.adaptive_sampling(
        n_points=20,
        X_existing=X_initial,
        y_existing=y_initial,
        strategy='distance_based'
    )
    
    # Visualize
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(
        X_initial[:, 0], X_initial[:, 1], X_initial[:, 2],
        c='blue', label='Initial', alpha=0.6, s=50
    )
    ax1.set_title("Initial Design (LHS)")
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        X_initial[:, 0], X_initial[:, 1], X_initial[:, 2],
        c='blue', label='Initial', alpha=0.4, s=30
    )
    ax2.scatter(
        X_adaptive[:, 0], X_adaptive[:, 1], X_adaptive[:, 2],
        c='red', label='Adaptive', alpha=0.8, s=50, marker='^'
    )
    ax2.set_title("With Adaptive Points")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Space-Filling Designs")
    print("=" * 70)
    
    # Generate different space-filling designs
    X_maximin = generator.generate(40, method='maximin', iterations=500)
    X_sphere = generator.generate(40, method='sphere_packing', iterations=300)
    
    # Calculate fill quality
    X_maximin_norm = bounds.normalize(X_maximin)
    X_sphere_norm = bounds.normalize(X_sphere)
    
    print(f"Maximin score: {generator._maximin_score(X_maximin_norm)}")
