import numpy as np
from typing import List, Dict
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


    def _compute_transforms(self, feature_data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Implementation of abstract method from BaseOptimizer.
        Compute Chebyshev transforms for all degrees up to max_degree.
        """
        transforms = {}
        for d in range(self.max_degree + 1):
            cheb_step = ChebyshevStep(degree=d)
            arr = cheb_step.transform_diagonal(feature_data)
            transforms[d] = arr
        return transforms
    def evaluate_expressiveness(self, x_data: pl.DataFrame, y_data: np.ndarray, cv_strategy: str = 'expanding_window') -> np.ndarray:
        """
        Calculate R^2 scores using chosen cross-validation strategy.
        :param x_data: INput data as DataFrame
        :param y_data: Target data
        :param cv_strategy: Cross-validation strategy
        :return: Array of R^2 scores for each degree
        """

        scores = np.zeros(self.max_degree + 1)

        # Get folds based on strategy
        if cv_strategy == 'expanding_window':
            folds = self._get_expanding_window_folds(x_data)
        else:  # time_based
            folds = self._get_time_based_folds(x_data)

        for d in range(self.max_degree + 1):
            cv_scores = []
            for i, (train_mask, val_mask) in enumerate(folds):
                # Get train/val data using masks
                train_data = x_data.filter(train_mask)
                val_data = x_data.filter(val_mask)
                
                # Get corresponding y data
                train_y = y_data[train_mask]
                val_y = y_data[val_mask]

                # Rest remains the same
                fold_id = f"fold_{i}"

                train_features = []
                val_features = []
                for degree in range(d + 1):
                    train_collapsed = self._compute_collapsed_combinations(train_data, f"{fold_id}_train")[degree]
                    val_collapsed = self._compute_collapsed_combinations(val_data, f"{fold_id}_val")[degree]
                    train_features.append(train_collapsed.reshape(-1,1))
                    val_features.append(val_collapsed.reshape(-1,1))
                X_train = np.hstack(train_features) if train_features else np.zeros((len(train_y), 0))
                X_val = np.hstack(val_features) if val_features else np.zeros((len(val_y), 0))
                # Compute condition number for debugging
                if X_train.size > 0:
                    u, s, vh = np.linalg.svd(X_train, full_matrices=False)
                    cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
                    print("Condition number of X_train:", cond_num)
                coeffs = np.linalg.lstsq(X_train, train_y, rcond=None)[0]
                val_pred = X_val @ coeffs
                cv_scores.append(self._compute_validation_score(val_pred, val_y))
            scores[d] = np.mean(cv_scores)
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
                      num_reads: int = 1000) -> List[List[int]]:
        """
        Optimize degrees for a single layer.
        Args:
            layer_idx: Which layer to optimize
            x_data: Input data
            y_data: Target data
            num_reads: Number of annealing reads
        Returns:
            List of optimal degrees for this layer's functions
        """
        input_dim = self.network_shape[layer_idx]
        output_dim = self.network_shape[layer_idx + 1]
        num_functions = input_dim * output_dim

        q = Array.create('q', shape=(num_functions, self.max_degree + 1), vartype='BINARY')

        scores = self.evaluate_expressiveness(x_data, y_data)
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
    def save_state(self, filename: str) -> None:
        """Save optimizer state"""
        state = {
            'network_shape': self.network_shape,
            'max_degree': self.max_degree,
            'complexity_weight': self.complexity_weight,
            'significance_threshold': self.significance_threshold,
            'fold_caches': self._fold_caches
        }
        np.save(filename, state)

    def load_state(self, filename: str) -> None:
        """Load optimizer state"""
        state = np.load(filename, allow_pickle=True).item()
        self.network_shape = state['network_shape']
        self.max_degree = state['max_degree']
        self.complexity_weight = state['complexity_weight']
        self.significance_threshold = state['significance_threshold']
        self._fold_caches = state['fold_caches']