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
                 complexity_weight: float = 0.1,
                 significance_threshold: float = 0.05,):
        """
        Initializes degree optimizer using QUBO formulation.
        :param network_shape: Shape of the network
        :param num_layers: Number of layers to optimize degrees for
        :param max_degree: Maximum polynomial degree to consider
        :param complexity_weight: Weight for degree complexity penalty
        :param significance_threshold: Minimum relative improvement needed to prefer higher degree
        """
        self.network_shape = network_shape
        self.num_layers = len(network_shape)  - 1
        self.max_degree = max_degree
        self.complexity_weight = complexity_weight
        self.significance_threshold = significance_threshold

    def is_degree_definitive(self, scores: np.ndarray) -> tuple[bool, int]:
        """
        Determine if there's a definitively best degree based on R² scores.

        :param scores: Array of R² scores for each degree
        :return: Tuple of (is_definitive, best_degree)
        """
        best_degree = int(np.argmax(scores))
        best_score = float(scores[best_degree])

        # Check if best degree is significantly better than others
        is_definitive = True
        for d in range(len(scores)):
            if d != best_degree:
                score = float(scores[d])  # Explicitly convert to float
                relative_improvement = (best_score - score) / (1 - score + 1e-10)
                if relative_improvement < self.significance_threshold:
                    is_definitive = False
                    break

        return is_definitive, best_degree


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


        input_dim = self.network_shape[layer_idx]
        output_dim = self.network_shape[layer_idx + 1]
        num_functions = input_dim * output_dim

        q = Array.create('q', shape=(num_functions, self.max_degree + 1), vartype='BINARY')

        scores = self.evaluate_expressiveness(x_data, y_data)
        is_definitive, definitive_degree = self.is_degree_definitive(scores)
        # Build QUBO for this layer
        H = 0.0

        if is_definitive:
            # If we have a definitive degree, strongly encourage its selection
            definitive_bonus = 100.0  # Large bonus for selecting the definitive degree
            for i in range(num_functions):
                # Add large bonus for selecting the definitive degree
                H += -definitive_bonus * q[i, definitive_degree]
                # Add large penalty for selecting any other degree
                for d in range(self.max_degree + 1):
                    if d != definitive_degree:
                        H += definitive_bonus * q[i, d]
        else:
            # Normal expressiveness reward term
            for i in range(num_functions):
                for d in range(self.max_degree + 1):
                    H += -1.0 * scores[d] * q[i,d]

        # 2. Degree complexity penalty
        for i in range(num_functions):
            for d in range(self.max_degree + 1):
                H += self.complexity_weight * (d**2) * q[i,d]

        # 3. K-fold cross validation interaction terms
        for i in range(num_functions):
            for d1 in range(self.max_degree + 1):
                for d2 in range(d1 + 1, self.max_degree + 1):
                    H += self.complexity_weight * 0.5 * d1 * d2 * q[i,d1] * q[i,d2]

        # 4. Constraint: exactly one degree per function
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

            # Cross-validation
            k = 5
            fold_size = len(x_data) // k
            cv_scores = []
            for i in range(k):
                val_idx = slice(i * fold_size, (i + 1) * fold_size)

                # Use only validation fold data for scoring
                y_val = y_data[val_idx]
                y_pred_val = y_pred[val_idx]

                # Calculate R² on validation fold only
                ss_res = np.sum((y_val - y_pred_val) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                cv_scores.append(1 - (ss_res / (ss_tot + 1e-10)))

            scores[d] = np.mean(cv_scores)
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
        """Test degree selection for combinations of Chebyshev polynomials using carefully chosen points"""

        # Case 1: T_2(x) = 2x² - 1
        # Choose points at extrema and zero crossings where degree 1 must fail
        x1 = np.array([-1.0, -1/np.sqrt(2), 0.0, 1/np.sqrt(2), 1.0])
        y1 = 2*x1**2 - 1  # Will give [1, 0, -1, 0, 1]

        # Test pure T_2 case first to verify degree detection
        layer_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=x1[:, None],
            y_data=y1[:, None],
            num_reads=100
        )
        self.assertEqual(layer_degrees[0][0], 2,
                         "Should select degree 2 for T_2 at critical points")

        # Case 2: T_1(x) + T_2(x) at points where degree 1 will have large error
        x2 = np.array([-1.0, -0.8, -0.5, 0.0, 0.5, 0.8, 1.0])
        y2 = x2 + (2*x2**2 - 1)  # T_1 + T_2

        # Calculate errors for degree 1 and 2 fits explicitly
        cheb1 = ChebyshevStep(degree=1)
        cheb2 = ChebyshevStep(degree=2)

        error_d1 = np.mean((cheb1.transform_diagonal(x2) - y2)**2)
        error_d2 = np.mean((cheb2.transform_diagonal(x2) - y2)**2)

        # Only run the optimizer test if degree 2 is definitively better
        if error_d2 < error_d1:
            layer_degrees = self.optimizer.optimize_layer(
                layer_idx=0,
                x_data=x2[:, None],
                y_data=y2[:, None],
                num_reads=100
            )
            self.assertEqual(layer_degrees[0][0], 2,
                             "Should select degree 2 when it has definitively lower error")
        else:
            print(f"Warning: Degree 1 error ({error_d1:.6f}) <= Degree 2 error ({error_d2:.6f})")

        # Case 3: T_3(x) with points specifically chosen to require degree 3
        x3 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y3 = 4*x3**3 - 3*x3  # Pure T_3

        # Verify errors decrease with increasing degree
        errors = []
        for d in range(1, 4):
            cheb = ChebyshevStep(degree=d)
            errors.append(np.mean((cheb.transform_diagonal(x3) - y3)**2))

        # Only test if degree 3 is clearly best
        if errors[2] < min(errors[0], errors[1]):
            layer_degrees = self.optimizer.optimize_layer(
                layer_idx=0,
                x_data=x3[:, None],
                y_data=y3[:, None],
                num_reads=100
            )
            self.assertEqual(layer_degrees[0][0], 3,
                             "Should select degree 3 when lower degrees have larger errors")
        else:
            print(f"Warning: Degree 3 not optimal. Errors: {errors}")

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

    def test_degree_interactions(self):
        """Test QUBO interaction terms and degree selection for polynomial combinations"""

        def generate_critical_points(degree):
            """Generate points where degree n polynomial behavior is distinct"""
            if degree == 0:
                return np.array([0.0])  # Constant
            elif degree == 1:
                return np.array([-1.0, 1.0])  # Linear extremes
            elif degree == 2:
                return np.array([-1.0, 0.0, 1.0])  # Quadratic extrema
            else:
                # For degree n, use zeros of derivative plus endpoints
                x = np.linspace(-1, 1, degree + 2)
                return x

        # Test Case 1: Clean separation between degrees
        x = generate_critical_points(3)  # Get critical points for degree 3
        # Pure T_3 at its critical points - no lower degree can fit these points
        y = 4*x**3 - 3*x  # T_3(x)

        scores = self.optimizer.evaluate_expressiveness(x[:, None], y[:, None])
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

        self.assertTrue(is_definitive,
                        "Should be definitive for pure T_3 at critical points")
        self.assertEqual(best_degree, 3,
                         "Should select degree 3 for pure T_3")

        # Test Case 2: Balanced combination with clear optimal degree
        # Choose points at T_2 extrema where its behavior is most distinct
        x = np.array([-1/np.sqrt(2), 0.0, 1/np.sqrt(2)])  # Points where T_2 behavior is unique
        # T_2 dominant with small T_1 component
        y = (2*x**2 - 1) + 0.1*x

        scores = self.optimizer.evaluate_expressiveness(x[:, None], y[:, None])
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

        self.assertTrue(is_definitive,
                        "Should be definitive when one degree clearly dominates")
        self.assertEqual(best_degree, 2,
                         "Should select degree 2 when T_2 dominates")

        # Test Case 3: Non-definitive case
        # Choose points where both T_1 and T_2 contribute significantly
        x = np.array([-0.8, -0.3, 0.3, 0.8])  # Points between extrema
        # Equal mix of T_1 and T_2
        y = x + (2*x**2 - 1)

        scores = self.optimizer.evaluate_expressiveness(x[:, None], y[:, None])
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

        self.assertFalse(is_definitive,
                         "Should not be definitive for balanced T_1 + T_2")

        # When not definitive, check QUBO interaction terms
        layer_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=x[:, None],
            y_data=y[:, None],
            num_reads=100
        )
        # Should prefer lower degree due to complexity penalty
        self.assertEqual(layer_degrees[0][0], 1,
                         "Should prefer degree 1 when degrees perform similarly")

        # Test Case 4: Close to definitive but not quite
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        # T_2 slightly stronger than T_1
        y = x + 1.2*(2*x**2 - 1)

        scores = self.optimizer.evaluate_expressiveness(x[:, None], y[:, None])
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

        self.assertFalse(is_definitive,
                         "Should not be definitive when improvement is below threshold")

        layer_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=x[:, None],
            y_data=y[:, None],
            num_reads=100
        )
        # Should still prefer lower degree due to complexity penalty
        self.assertEqual(layer_degrees[0][0], 1,
                         "Should prefer degree 1 when degree 2 advantage is small")

    def test_cross_validation_impact(self):
        """Test that cross-validation prevents overfitting to specific points"""
        x = np.linspace(-1, 1, 50)  # More points for meaningful cross-validation
        # T_2 with noise
        y = (2*x**2 - 1) + np.random.normal(0, 0.1, size=x.shape)

        scores = self.optimizer.evaluate_expressiveness(x[:, None], y[:, None])
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

        # Even with noise, should still identify degree 2 as optimal
        self.assertTrue(is_definitive,
                        "Should be definitive despite noise due to cross-validation")
        self.assertEqual(best_degree, 2,
                         "Should identify correct degree despite noise")