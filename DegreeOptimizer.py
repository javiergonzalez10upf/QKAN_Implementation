import unittest
from typing import List, Dict

import neal
import numpy as np
from cpp_pyqubo import Placeholder, Constraint
from pyqubo import Array

from ChebyshevStep import ChebyshevStep


class DegreeOptimizer:
    def __init__(self,
                 network_shape: List[int],
                 max_degree: int,
                 complexity_weight: float = 0.1):
        """
        Initializes degree optimizer using QUBO formulation.
        :param network_shape: Shape of the network
        :param num_layers: Number of layers to optimize degrees for
        :param max_degree: Maximum polynomial degree to consider
        :param complexity_weight: Weight for degree complexity penalty
        """
        self.network_shape = network_shape
        self.num_layers = len(network_shape)  - 1
        self.max_degree = max_degree
        self.complexity_weight = complexity_weight

    def optimize_layer(self,
                       layer_idx: int,
                       x_data: np.ndarray,
                       y_data: np.ndarray,
                       num_reads: int = 1000,
                       feed_dict: dict[str, float] = None) -> List[List[int]]:
        """
        Optimize degrees for a single layer.
        :param layer_idx: Which layer to optimize
        :param x_data: Input data for this layer
        :param y_data: Target data for this layer
        :param num_reads: Number of annealing reads
        :param feed_dict: Optional dictionary for feed-forward optimization
        :return: optimal degrees: List of optimal degrees for this layer's functions
        """
        if feed_dict is None:
            feed_dict = {'complexity_weight': 0.1}  # Default value

        input_dim = self.network_shape[layer_idx]
        output_dim = self.network_shape[layer_idx + 1]
        num_functions = input_dim * output_dim

        q = Array.create('q', shape=(num_functions, self.max_degree + 1), vartype='BINARY')

        scores = self.evaluate_expressiveness(x_data, y_data)

        # Build QUBO for this layer
        H = 0.0

        # 1. Expressiveness reward term
        for i in range(num_functions):
            for d in range(self.max_degree + 1):
                H += -1.0 * scores[d] * q[i,d]

        # 2. Degree complexity penalty
        for i in range(num_functions):
            for d in range(self.max_degree + 1):
                H += self.complexity_weight * (d**2) * q[i,d]

        # 3. Constraint: exactly one degree per function
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

    def evaluate_expressiveness(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """
        Calculate R^2 scores for each degree [max_degree+1]
        :param x_data: Input data [N samples]
        :param y_data: Target data [N samples]
        :return: scores: R^2 scores for each degree [max_degree+1]
        """
        scores = np.zeros(self.max_degree+1)
        for d in range(self.max_degree+1):
            cheb_step = ChebyshevStep(degree=d)
            y_pred = cheb_step.transform_diagonal(x_data)
            # R^2 score calculation
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            scores[d] = 1 - (ss_res / (ss_tot + 1e-10))
        return scores


    def optimize_network(self,
                         training_data: Dict[str, np.ndarray],
                         num_reads: int = 1000) -> List[List[List[int]]]:
        """
        Optimize degrees for entier network layer by layer
        :param training_data: Dictionary containing layer-wise training data
        :param num_reads: Number of annealing reads
        :return: network_degrees: List of optimal degrees for each layer
        """

        network_degrees = []

        for layer in range(self.num_layers):
            layer_degrees = self.optimize_layer(
                layer_idx=layer,
                x_data = training_data[f'layer_{layer}_input'],
                y_data = training_data[f'layer_{layer}_output'],
                num_reads = num_reads
            )
            network_degrees.append(layer_degrees)

        return network_degrees


class TestDegreeOptimizer(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.max_degree = 8
        self.network_shape = [1, 1]  # Start with simplest case
        self.optimizer = DegreeOptimizer(
            network_shape=self.network_shape,
            max_degree=self.max_degree,
            complexity_weight=0.1
        )

    def test_single_chebyshev_degrees(self):
        """Test degree selection for pure Chebyshev polynomials"""
        x = np.linspace(-1, 1, 50)
        test_cases = [
            # T_0(x) = 1
            (lambda x: np.ones_like(x), 0, "T_0(x) = 1"),
            # T_1(x) = x
            (lambda x: x, 1, "T_1(x) = x"),
            # T_2(x) = 2x² - 1
            (lambda x: 2*x**2 - 1, 2, "T_2(x) = 2x² - 1"),
            # T_3(x) = 4x³ - 3x
            (lambda x: 4*x**3 - 3*x, 3, "T_3(x) = 4x³ - 3x")
        ]

        for func, expected_degree, name in test_cases:
            y = func(x)

            layer_degrees = self.optimizer.optimize_layer(
                layer_idx=0,
                x_data=x[:, None],
                y_data=y[:, None],
                num_reads=100
            )

            self.assertEqual(layer_degrees[0][0], expected_degree,
                             f"Should select degree {expected_degree} for {name}")

    def test_chebyshev_combinations(self):
        """Test degree selection for combinations of Chebyshev polynomials"""
        x = np.linspace(-1, 1, 50)
        test_cases = [
            # T_1(x) + T_2(x)
            (lambda x: x + (2*x**2 - 1), 2, "T_1(x) + T_2(x)"),
            # T_2(x) + T_3(x)
            (lambda x: (2*x**2 - 1) + (4*x**3 - 3*x), 3, "T_2(x) + T_3(x)"),
            # 0.5*T_1(x) + 0.5*T_3(x)
            (lambda x: 0.5*x + 0.5*(4*x**3 - 3*x), 3, "0.5*T_1(x) + 0.5*T_3(x)")
        ]

        for func, expected_degree, name in test_cases:
            y = func(x)

            layer_degrees = self.optimizer.optimize_layer(
                layer_idx=0,
                x_data=x[:, None],
                y_data=y[:, None],
                num_reads=100
            )

            self.assertEqual(layer_degrees[0][0], expected_degree,
                             f"Should select degree {expected_degree} for {name}")

    def test_noisy_chebyshev(self):
        """Test degree selection with noisy Chebyshev polynomials"""
        x = np.linspace(-1, 1, 50)
        np.random.seed(42)  # For reproducibility

        test_cases = [
            # T_2(x) + small noise
            (lambda x: (2*x**2 - 1) + np.random.normal(0, 0.1, x.shape), 2,
             "T_2(x) + small noise"),
            # T_3(x) + medium noise
            (lambda x: (4*x**3 - 3*x) + np.random.normal(0, 0.2, x.shape), 3,
             "T_3(x) + medium noise")
        ]

        for func, expected_degree, name in test_cases:
            y = func(x)

            layer_degrees = self.optimizer.optimize_layer(
                layer_idx=0,
                x_data=x[:, None],
                y_data=y[:, None],
                num_reads=100
            )

            self.assertEqual(layer_degrees[0][0], expected_degree,
                             f"Should select degree {expected_degree} for {name}")

    def test_degree_preference(self):
        """Test optimizer prefers lower degrees when possible"""
        x = np.linspace(-1, 1, 50)
        # T_1(x) with very small high degree terms
        y = x + 0.01*(4*x**3 - 3*x) + 0.001*(8*x**4 - 8*x**2 + 1)

        layer_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=x[:, None],
            y_data=y[:, None],
            num_reads=100
        )

        self.assertEqual(layer_degrees[0][0], 1,
                         "Should prefer degree 1 when higher degrees have negligible coefficients")

    def test_complexity_weight_impact(self):
        """Test impact of complexity weight on degree selection"""
        x = np.linspace(-1, 1, 50)
        y = 4*x**3 - 3*x  # T_3(x)

        # With very high complexity penalty
        high_penalty_optimizer = DegreeOptimizer(
            network_shape=self.network_shape,
            max_degree=self.max_degree,
            complexity_weight=1.0  # High penalty
        )

        degrees_high_penalty = high_penalty_optimizer.optimize_layer(
            layer_idx=0,
            x_data=x[:, None],
            y_data=y[:, None],
            num_reads=100
        )

        # Should select lower degree due to high complexity penalty
        self.assertLess(degrees_high_penalty[0][0], 3,
                        "High complexity penalty should force lower degree")