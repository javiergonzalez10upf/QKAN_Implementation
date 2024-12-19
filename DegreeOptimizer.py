import numpy as np
from typing import List, Dict, Tuple
import polars as pl
from cpp_pyqubo import Constraint
from pyqubo import Array
import neal

from BaseOptimizer import BaseOptimizer
from QKAN_Steps.ChebyshevStep import ChebyshevStep


class DegreeOptimizer(BaseOptimizer):
    def __init__(self,
                 network_shape: List[int],
                 max_degree: int,
                 complexity_weight: float = 0.1,
                 significance_threshold: float = 0.05):
        """
        Initialize degree optimizer using QUBO formulation with collapsed combinations.
        Args:
            network_shape: Shape of the network
            max_degree: Maximum polynomial degree to consider
            complexity_weight: Weight for degree complexity penalty
            significance_threshold: Minimum relative improvement needed to prefer higher degree
        """
        super().__init__()
        self.network_shape = network_shape
        self.num_layers = len(network_shape) - 1
        self.max_degree = max_degree
        self.complexity_weight = complexity_weight
        self.significance_threshold = significance_threshold
        self.transform_cache = {}
        self.degree_scores = {}
        self.data_same = True


    def _compute_transforms(self, feature_data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Implementation of abstract method from BaseOptimizer.
        Compute Chebyshev transforms for all degrees up to max_degree.
        """
        transforms = {}
        n_samples = len(feature_data)

        for d in range(self.max_degree + 1):
            cheb_step = ChebyshevStep(degree=d)
            # Transform each feature separately and maintain sample dimension
            transformed_features = []
            for feature_idx in range(feature_data.shape[1]):
                feature_transform = cheb_step.transform_diagonal(feature_data[:, feature_idx])
                transformed_features.append(feature_transform)

            transforms[d] = np.stack(transformed_features, axis=1)

        return transforms

    def evaluate_degree(self, x_data: pl.DataFrame, y_data: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
            Calculate R² scores for each degree directly without cross-validation.
            """
        cache_key = str(x_data.schema)

        if cache_key in self.degree_scores and self.data_same:
            print("Using cached degree scores...")
            return self.degree_scores[cache_key]

        scores = np.zeros(self.max_degree + 1)

        feature_data = x_data.to_numpy()

        for d in range(self.max_degree + 1):
            # Get transforms up to this degree
            transforms = []
            for degree in range(d + 1):
                degree_transform = self._compute_transforms(feature_data)[degree]
                transforms.append(degree_transform.reshape(len(feature_data), -1))

            # Stack features
            X = np.hstack(transforms) if transforms else np.zeros((len(y_data), 0))

            # Fit and predict
            coeffs = np.linalg.lstsq(X, y_data, rcond=None)[0]
            y_pred = X @ coeffs

            # Calculate metrics
            metrics = self._compute_metrics(y_data, y_pred, weights)
            scores[d] = metrics['mse']  # or r2 depending on what we want to optimize

            print(f"\nDegree {d}:")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"R²:  {metrics['r2']:.4f}")
            self.degree_scores[cache_key] = scores
        return scores
    def is_degree_definitive(self, scores: np.ndarray) -> tuple[bool, int]:
        """
        Determine if there's a definitively best degree based on R² scores.
        Args:
            scores: Array of R² scores for each degree
        Returns:
            Tuple of (is_definitive, best_degree)
        """
        best_degree = int(np.argmin(scores))
        best_score = float(scores[best_degree])

        is_definitive = True
        for d in range(len(scores)):
            if d != best_degree:
                score = float(scores[d])
                # Changed relative improvement calculation for MSE
                relative_improvement = (score - best_score) / (score + 1e-10)
                if relative_improvement < self.significance_threshold:
                    is_definitive = False
                    break


        return is_definitive, best_degree

    def optimize_layer(self, layer_idx: int, x_data: pl.DataFrame, y_data: np.ndarray, 
                       weights:np.ndarray, num_reads: int = 1000) -> List[List[int]]:
        """
        Optimize degrees for a single layer.
        Args:
            layer_idx: Which layer to optimize
            x_data: Input data
            y_data: Target data
            time_data: Time data
            weights: Optional sample weights
            num_reads: Number of annealing reads
        Returns:
            List of optimal degrees for this layer's functions

        """
        input_dim = self.network_shape[layer_idx]
        output_dim = self.network_shape[layer_idx + 1]
        num_functions = input_dim * output_dim

        q = Array.create('q', shape=(num_functions, self.max_degree + 1), vartype='BINARY')

        scores = self.evaluate_degree(x_data, y_data, weights)

        # Print metrics per degree for monitoring if needed
        print("Optimize_layer using annealer")

        is_definitive, definitive_degree = self.is_degree_definitive(scores)
        
        # Build QUBO
        H = 0.0
        
        if is_definitive:
            for i in range(num_functions):
                H += -100.0 * q[i, definitive_degree]
                for d in range(self.max_degree + 1):
                    if d != definitive_degree:
                        H += 100.0 * q[i, d]
        else:
            for i in range(num_functions):
                for d in range(self.max_degree + 1):
                    improvement = scores[d] - scores[d-1] if d > 0 else scores[d]
                    H += -1.0 * improvement * q[i,d]
                    H += self.complexity_weight * (d**2) * q[i,d]

        # Constraint: exactly one degree per function
        for i in range(num_functions):
            constraint = (sum(q[i,d] for d in range(self.max_degree + 1)) - 1)**2
            H += 10.0 * Constraint(constraint, label=f'one_degree_{i}')

        # Compile and solve
        model = H.compile()
        bqm = model.to_bqm()

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)

        decoded = model.decode_sampleset(sampleset)
        best_sample = min(decoded, key=lambda x: x.energy)

        # Extract optimal degrees
        optimal_degrees = []
        for out_idx in range(output_dim):
            output_connections = []
            for in_idx in range(input_dim):
                qubo_idx = out_idx * input_dim + in_idx
                for d in range(self.max_degree + 1):
                    if best_sample.sample[f'q[{qubo_idx}][{d}]'] == 1:
                        output_connections.append(d)
                        break
            optimal_degrees.append(output_connections)
        
        return optimal_degrees

    def optimize_network(self, training_data: Dict[str, np.ndarray], 
                        num_reads: int = 1000) -> List[List[List[int]]]:
        """
        Optimize degrees for entire network layer by layer.
        Args:
            training_data: Dictionary containing layer-wise training data
            num_reads: Number of annealing reads
        Returns:
            List of optimal degrees for each layer
        """
        network_degrees = []
        for layer in range(self.num_layers):
            layer_degrees = self.optimize_layer(
                layer_idx=layer,
                x_data=training_data[f'layer_{layer}_input'],
                y_data=training_data[f'layer_{layer}_output'],
                num_reads=num_reads
            )
            network_degrees.append(layer_degrees)
        return network_degrees

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Compute both MSE and R² scores.
        :param y_true: True values
        :param y_pred: Predicted values
        :param weights: Optional sample weights
        :return: Dictionary with both metrics
        """
        # MSE calculation (weighted if weights provided)
        squared_errors = (y_true - y_pred) ** 2
        if weights is not None:
            mse = np.average(squared_errors, weights=weights)
        else:
            mse = np.mean(squared_errors)
        # R² calculation
        if weights is not None:
            # Weighted R² using competition formula
            weighted_squared_errors_sum = np.sum(weights * squared_errors)
            weighted_y_squared_sum = np.sum(weights * y_true ** 2)
            r2 = 1 - weighted_squared_errors_sum / weighted_y_squared_sum if weighted_y_squared_sum > 0 else 0.0
        else:
            # Unweighted R²
            ss_res = np.sum(squared_errors)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'mse': mse,
            'r2': r2
        }
    def save_state(self, filename: str, query_params: Dict = None) -> None:
        """Save optimizer state"""
        if query_params is None:
            query_params = {
                'n_rows': 100000,
                'columns': ['date_id', 'responder_6', 'weight'] + [f'feature_{i:02d}' for i in range(79)],
                'sort_by': 'date_id',
            }
        state = {
            'network_shape': self.network_shape,
            'max_degree': self.max_degree,
            'complexity_weight': self.complexity_weight,
            'significance_threshold': self.significance_threshold,
            'transform_cache': self.transform_cache,
            'degree_scores': self.degree_scores,
            'query_params' : query_params,
        }
        np.save(filename, state)

    def load_state(self, filename: str, current_query_params:dict) -> None:
        """Load optimizer state and validate query matches"""
        state = np.load(filename, allow_pickle=True).item()

        # Load basic parameters
        self.network_shape = state['network_shape']
        self.max_degree = state['max_degree']
        self.complexity_weight = state['complexity_weight']
        self.significance_threshold = state['significance_threshold']

        # Validate query matches
        if self._validate_query(state['query_params'], current_query_params):
            print("Loading cached computations")
            self.transform_cache = state['transform_cache']
            self.degree_scores = state['degree_scores']
        else:
            print("Query changed, clearing caches")
            self.data_same = False
            self.transform_cache = {}
            self.degree_scores = {}
    def _validate_query(self, saved_params: dict, current_query_params: dict) -> bool:
        """Validate query matches"""
        return (saved_params['n_rows'] == current_query_params['n_rows'] and
                saved_params['columns'] == current_query_params['columns'] and
                saved_params['sort_by'] == current_query_params['sort_by'])
