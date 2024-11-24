import time
import unittest

import numpy as np
from fable import fable
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from MulStep import MulStep


class LCUStep:
    def __init__(self, max_degree: int):
        """
        Initialize LCU step for QKAN.
        :param max_degree: Maximum degree of Chebyshev polynomials (d).
        """

        self.max_degree = max_degree

    def combine_weighted_polynomials(
            self,
            x: np.ndarray,
            mul_step:
            MulStep,
            K:int
    ) -> tuple[QuantumCircuit, float]:
        """
        Combine weighted Chebyshev polynomials using LCU.
        :param x: Input vector with values in [-1,1]
        :param mul_step: MulStep object containing weights for each degree
        :param K: Number of copies for dilation
        :return: Quantum circuit implementing combined polynomials and scaling factor
        """

        d = self.max_degree
        combined = np.zeros((len(x) * K, len(x) * K))

        for degree in range(d + 1):
            matrix = mul_step.get_weighted_polynomial_matrix(x, K, degree)
            combined += matrix / (d + 1)

        return fable(combined, 0)


class TestLCUStep(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = Aer.get_backend('unitary_simulator')
        np.random.seed(42)  # For reproducibility

    def verify_unitary(self, circuit, expected_matrix, scale, test_name=""):
        """Helper to verify circuit implements expected matrix"""
        print(f"\n=== Testing {test_name} ===")

        # Get circuit unitary
        compiled = transpile(circuit, self.simulator)
        result = self.simulator.run(compiled).result()
        unitary = np.asarray(result.get_unitary(compiled))

        # Extract top-left block
        block_size = expected_matrix.shape[0]
        actual = unitary[:block_size, :block_size] * scale * block_size

        print("\nExpected matrix:")
        print(np.array2string(expected_matrix, precision=4, suppress_small=True))
        print("\nActual matrix (scaled):")
        print(np.array2string(actual, precision=4, suppress_small=True))

        # For zero matrices, use absolute difference
        if np.allclose(expected_matrix, 0):
            print("\nZero matrix detected, using absolute difference")
            elem_diff = np.abs(actual - expected_matrix)
            diff = np.linalg.norm(actual - expected_matrix)
        else:
            # Element-wise relative differences
            with np.errstate(divide='ignore', invalid='ignore'):
                elem_diff = np.abs((actual - expected_matrix) / expected_matrix)
                elem_diff = np.nan_to_num(elem_diff)
            diff = np.linalg.norm(actual - expected_matrix) / np.linalg.norm(expected_matrix)

        print("\nElement-wise differences:")
        print(np.array2string(elem_diff, precision=4, suppress_small=True))
        print(f"\nOverall difference: {diff:.2e}")
        print(f"Scale factor: {scale}")
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Number of qubits: {circuit.num_qubits}")

        # Verify within tolerance
        self.assertTrue(diff < 1e-6, f"Difference too high: {diff}")

    def time_operation(self, operation, operation_name="Operation"):
        """Helper to time operations"""
        start_time = time.time()
        result = operation()
        end_time = time.time()
        print(f"{operation_name} time: {end_time - start_time:.4f} seconds")
        return result

    def test_combine_degree1(self):
        """Test combining polynomials up to degree 1 with equal weights 1/2"""
        N, K = 2, 2
        mul = MulStep(degree=1, num_weights=N * K)

        # Random input vector in [-1,1]
        x = np.random.uniform(-1, 1, size=N)

        # Random weights in [-1,1]
        weights0 = np.random.uniform(-1, 1, size=N * K)
        weights1 = np.random.uniform(-1, 1, size=N * K)
        mul.set_weights(0, weights0)
        mul.set_weights(1, weights1)

        # Get matrices for each polynomial
        matrix0 = mul.get_weighted_polynomial_matrix(x, K, 0)
        matrix1 = mul.get_weighted_polynomial_matrix(x, K, 1)

        # Each polynomial contributes with weight 1/(d+1)
        expected = (matrix0 + matrix1) / 2

        # Create LCU step and combine
        lcu = LCUStep(max_degree=1)
        circuit, scale = lcu.combine_weighted_polynomials(x, mul, K)
        self.verify_unitary(circuit, expected, scale, "Degree 1 Combined")

    def test_medium_systems(self):
        """Test different combinations of N, K, and degrees"""
        configs = [
            {"N": 4, "K": 4, "d": 5, "name": "4x4 High Degree"},
            {"N": 4, "K": 8, "d": 8, "name": "4x6 Medium"},
            {"N": 8, "K": 4, "d": 7, "name": "6x4 Medium"},
            {"N": 4, "K": 8, "d": 20, "name": "4x8 Wide"},
        ]

        for config in configs:
            N, K, d = config["N"], config["K"], config["d"]
            print(f"\n=== Testing {config['name']} ===")
            print(f"System size: {N}x{K} (dimension {N * K}) with degree {d}")

            mul = MulStep(degree=d, num_weights=N * K)
            x = np.random.uniform(-1, 1, size=N)

            # Time weight setup
            start = time.time()
            for degree in range(d + 1):
                weights = np.random.uniform(-1, 1, size=N * K)
                mul.set_weights(degree, weights)
            weight_time = time.time() - start
            print(f"Weight setup time: {weight_time:.4f}s")

            # Time matrix creation
            start = time.time()
            matrices = [mul.get_weighted_polynomial_matrix(x, K, degree)
                        for degree in range(d + 1)]
            expected = sum(matrices) / (d + 1)
            matrix_time = time.time() - start
            print(f"Matrix creation time: {matrix_time:.4f}s")

            # Time circuit creation
            start = time.time()
            lcu = LCUStep(max_degree=d)
            circuit, scale = lcu.combine_weighted_polynomials(x, mul, K)
            circuit_time = time.time() - start
            print(f"Circuit creation time: {circuit_time:.4f}s")

            # Time verification (most expensive part)
            start = time.time()
            self.verify_unitary(circuit, expected, scale, config["name"])
            verify_time = time.time() - start
            print(f"Verification time: {verify_time:.4f}s")

            print("\nBreakdown of computation:")
            total = weight_time + matrix_time + circuit_time + verify_time
            print(f"Weight setup: {weight_time / total * 100:.1f}%")
            print(f"Matrix creation: {matrix_time / total * 100:.1f}%")
            print(f"Circuit creation: {circuit_time / total * 100:.1f}%")
            print(f"Verification: {verify_time / total * 100:.1f}%")
            print(f"Memory footprint: {expected.nbytes / 1024 / 1024:.2f} MB")

    def test_edge_cases(self):
        """Test various edge cases"""
        N, K = 4, 4
        d = 2
        mul = MulStep(degree=d, num_weights=N * K)
        lcu = LCUStep(max_degree=d)

        cases = {
            "boundary_inputs": {
                "x": np.array([-1.0] * (N // 2) + [1.0] * (N // 2)),
                "weights": [np.random.uniform(-1, 1, N * K) for _ in range(d + 1)]
            },
            "alternating_weights": {
                "x": np.random.uniform(-1, 1, N),
                "weights": [np.array([1, -1] * (N * K // 2)) for _ in range(d + 1)]
            },
            "small_weights": {
                "x": np.random.uniform(-1, 1, N),
                "weights": [np.random.uniform(-0.01, 0.01, N * K) for _ in range(d + 1)]
            },
            "identical_weights": {
                "x": np.random.uniform(-1, 1, N),
                "weights": [np.array([0.5] * (N * K)) for _ in range(d + 1)]
            }
        }

        for case_name, case_data in cases.items():
            print(f"\nTesting edge case: {case_name}")

            def setup_case():
                for degree, weights in enumerate(case_data["weights"]):
                    mul.set_weights(degree, weights)

            def process_case():
                # Get matrices
                matrices = [mul.get_weighted_polynomial_matrix(case_data["x"], K, degree)
                            for degree in range(d + 1)]
                expected = sum(matrices) / (d + 1)

                # Create circuit
                circuit, scale = lcu.combine_weighted_polynomials(case_data["x"], mul, K)
                return expected, circuit, scale

            self.time_operation(setup_case, "Case setup")
            expected, circuit, scale = self.time_operation(
                process_case,
                "Case processing"
            )

            self.verify_unitary(circuit, expected, scale, f"Edge Case: {case_name}")