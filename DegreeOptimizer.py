import warnings
import numpy as np
from typing import List, Dict, Tuple
import polars as pl
from cpp_pyqubo import Placeholder, Constraint
from pyqubo import Array
import neal
from ChebyshevStep import ChebyshevStep


class DegreeOptimizer:
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
        self.network_shape = network_shape
        self.num_layers = len(network_shape) - 1
        self.max_degree = max_degree
        self.complexity_weight = complexity_weight
        self.significance_threshold = significance_threshold
        self._fold_caches = {}  # Cache per fold

    def _compute_collapsed_combinations(self, x_data: pl.DataFrame, fold_id: str = None) -> Dict[int, np.ndarray]:
        """
        Precompute and collapse Chebyshev polynomial combinations for a specific fold.
        Based on Troy's paper section 4.5, collapsing input combinations.
        Args:
            x_data: Input data
            fold_id: Optional identifier for the fold (for cross-validation)
        Returns:
            Dictionary mapping degrees to collapsed combinations
        """
        cache_key = f"{fold_id}_{hash(str(x_data))}" if fold_id else hash(str(x_data))
        if cache_key in self._fold_caches:
            return self._fold_caches[cache_key]
            
        # Convert feature data to numpy array
        feature_data = x_data.select(pl.col('^feature_.*$')).to_numpy()
            
        transforms = {}
        for d in range(self.max_degree + 1):
            cheb_step = ChebyshevStep(degree=d)
            arr = cheb_step.transform_diagonal(feature_data)
            transforms[d] = arr

        self._fold_caches[cache_key] = transforms
        return transforms

    def _compute_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score with validation checks"""
        if len(y_true) < 2:
            raise ValueError("Need at least 2 samples for R² score")

        # ss_res = np.sum((y_true - y_pred) ** 2)
        # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        #
        # score = 1 - (ss_res / (ss_tot + 1e-10))
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)
        score = 1 - (rss / tss)
        if not (-1 <= score <= 1):
            warnings.warn(f"Invalid R² score: {score}")
            
        return score
    def _get_time_based_folds(self, data: pl.DataFrame, n_splits:int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-based cross-validation folds."""
        timestamps = data.get_column('date_id').unique().sort()
        n_times = len(timestamps)

        folds = []
        for i in range(n_splits):
            split_idx = int((i + 1) * n_times // (n_splits + 1))
            val_end_idx = int((i + 2) * n_times // (n_splits + 1))

            #Training: up to split point
            train_times = timestamps[:split_idx]
            #Validation: next chunk
            val_times = timestamps[split_idx:val_end_idx]

            train_mask = data.get_column('date_id').is_in(train_times)
            val_mask = data.get_column('date_id').is_in(val_times)

            folds.append((train_mask.to_numpy(), val_mask.to_numpy()))
        return folds

    def _get_expanding_window_folds(self, data: pl.DataFrame, n_splits:int = 5, initial_ratio:float = 0.6) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create expanding window cross-validation folds.
        :param data: Input DataFrame with time column
        :param n_splits: Number of validation periods
        :param initial_ratio: Initial training set size as ratio of total data
        :return: List of (train_mask, val_mask) tuples
        """
        timestamps = data.get_column('date_id').unique().sort()
        n_times = len(timestamps)

        initial_train_size = int(n_times * initial_ratio)

        val_size = int((n_times - initial_train_size) / n_splits)

        folds = []
        for i in range(n_splits):
            train_end_idx = initial_train_size + i * val_size
            train_times = timestamps[:train_end_idx]

            val_times = timestamps[train_end_idx:min(train_end_idx + val_size, n_times)]

            train_mask = data.get_column('date_id').is_in(train_times)
            val_mask = data.get_column('date_id').is_in(val_times)

            folds.append((train_mask.to_numpy(), val_mask.to_numpy()))
        return folds

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
                mse = np.mean((val_y - val_pred) ** 2)
                cv_scores.append(mse)
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