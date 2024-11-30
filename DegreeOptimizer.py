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
        self.complexity_weight = Placeholder("complexity_weight")

    def optimize_layer(self,
                       layer_idx: int,
                       x_data: np.ndarray,
                       y_data: np.ndarray,
                       num_reads: int = 1000) -> List[List[int]]:
        """
        Optimize degrees for a single layer.
        :param layer_idx: Which layer to optimize
        :param x_data: Input data for this layer
        :param y_data: Target data for this layer
        :param num_reads: Number of annealing reads
        :return: optimal degrees: List of optimal degrees for this layer's functions
        """

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
        bqm = model.to_bqm(feed_dict={'complexity_weight': self.complexity_weight})

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
                    if best_sample.sample[f'q[{qubo_idx},{d}]'] == 1:
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
        """Set up test fixtures"""
        self.network_shape = [2, 2, 1]
        self.max_degree = 3
        self.optimizer = DegreeOptimizer(
            network_shape=self.network_shape,
            max_degree=self.max_degree,
            complexity_weight=0.1
        )
    def test_evaluate_expressiveness(self):
        """Test R² score calculation for Chebyshev polynomials"""
        # Create test data that follows T_2(x) = 2x² - 1
        x_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y_data = 2*x_data**2 - 1  # Second Chebyshev polynomial T_2(x)

        scores = self.optimizer.evaluate_expressiveness(x_data, y_data)

        # Test scores shape
        self.assertEqual(len(scores), self.max_degree + 1,
                         "Should have score for each possible degree")

        # Test scores values - degree 2 should be best for T_2(x)
        self.assertGreater(scores[2], scores[1],
                           "T_2(x) should score better with degree 2 than 1")
        self.assertGreater(scores[2], scores[0],
                           "T_2(x) should score better with degree 2 than 0")
        if self.max_degree > 2:
            self.assertGreater(scores[2], scores[3],
                               "T_2(x) should score better with degree 2 than 3")
    def test_optimize_layer(self):
        """Test layer degree optimization with Chebyshev polynomials"""
        # First layer has 2 input nodes -> 2 output nodes
        x = np.linspace(-1, 1, 5)  # 5 samples
        x_data = np.array([
            [x1, x2] for x1, x2 in zip(x, x[::-1])  # Different values for each input
        ])

        # Each output node computes a different Chebyshev polynomial
        # Output node 1: T_2(x1) for first input
        # Output node 2: T_1(x2) for second input
        y_data = np.array([
            [2*x1**2 - 1, x2]  # T_2(x1), T_1(x2)
            for x1, x2 in x_data
        ])

        layer_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=x_data,  # Shape: [5, 2]
            y_data=y_data,  # Shape: [5, 2]
            num_reads=100
        )

        # Check degrees matrix shape: [output_dim, input_dim] = [2, 2]
        self.assertEqual(len(layer_degrees), 2, "Should have degrees for 2 output nodes")
        self.assertEqual(len(layer_degrees[0]), 2, "Each output should have 2 input degrees")

        # First output node should prefer degree 2 for T_2(x) relationship
        self.assertEqual(layer_degrees[0][0], 2, "Should select degree 2 for T_2(x)")
        # Second output node should prefer degree 1 for T_1(x) relationship
        self.assertEqual(layer_degrees[1][1], 1, "Should select degree 1 for T_1(x)")

    def test_optimize_network(self):
        """Test full network optimization with Chebyshev polynomials"""
        # Layer 0: 2 inputs -> 2 outputs
        x = np.linspace(-1, 1, 5)
        layer0_input = np.array([
            [x1, x2] for x1, x2 in zip(x, x[::-1])
        ])
        layer0_output = np.array([
            [2*x1**2 - 1, x2]  # T_2(x1), T_1(x2)
            for x1, x2 in layer0_input
        ])

        # Layer 1: 2 inputs -> 1 output
        layer1_input = layer0_output  # Use output from layer 0
        layer1_output = np.array([
            [y1 + y2]  # Simple sum of inputs
            for y1, y2 in layer1_input
        ])

        training_data = {
            'layer_0_input': layer0_input,   # Shape: [5, 2]
            'layer_0_output': layer0_output, # Shape: [5, 2]
            'layer_1_input': layer1_input,   # Shape: [5, 2]
            'layer_1_output': layer1_output  # Shape: [5, 1]
        }

        network_degrees = self.optimizer.optimize_network(
            training_data=training_data,
            num_reads=100
        )

        # Test layer 0 degrees matrix: [2, 2]
        self.assertEqual(len(network_degrees[0]), 2, "Layer 0 should have 2 output nodes")
        self.assertEqual(len(network_degrees[0][0]), 2, "Layer 0 outputs should have 2 input degrees")

        # Test layer 1 degrees matrix: [1, 2]
        self.assertEqual(len(network_degrees[1]), 1, "Layer 1 should have 1 output node")
        self.assertEqual(len(network_degrees[1][0]), 2, "Layer 1 output should have 2 input degrees")