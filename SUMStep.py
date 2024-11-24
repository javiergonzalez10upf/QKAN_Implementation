import time
import unittest

import numpy as np
from fable import fable
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


class SUMStep():
    def __init__(self):
        """Initialize the SUM step for QKAN."""
        pass

    def apply_sum(
            self,
            matrix: np.ndarray,
            N: int,
            K:int
    ) -> tuple[QuantumCircuit, float]:
        """
        Apply summation over N inputs for each of K outputs.
        :param matrix: Input matrix from LCU step
        :param N: Number of input nodes
        :param K: Number of output nodes
        :return: Quantum circuit implementing final summer output and scaling factor
        """
        diag_elements = np.diag(matrix).reshape(N, K)
        summed = np.sum(diag_elements, axis=0) / N
        output = np.diag(summed)
        return fable(output, 0)


class TestSUMStep(unittest.TestCase):
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

    def test_basic_sum(self):
        """Test basic summation with small dimensions"""
        N, K = 4, 2  # 4 inputs, 2 outputs

        # Create a test matrix with known pattern
        x = np.arange(N * K).reshape(N, K).astype(float)
        input_matrix = np.zeros((N * K, N * K))
        for i in range(N):
            for j in range(K):
                input_matrix[i * K + j, i * K + j] = x[i, j]

        # Expected sum: each output j should get average of inputs
        expected = np.zeros((K, K))
        for j in range(K):
            expected[j, j] = np.mean(x[:, j])

        # Apply SUM step
        sum_step = SUMStep()
        circuit, scale = sum_step.apply_sum(input_matrix, N, K)

        self.verify_unitary(circuit, expected, scale, "Basic Sum")

    def test_power_of_two_systems(self):
        """Test power-of-two sized systems"""
        configs = [
            {"N": 4, "K": 4, "name": "4x4 Square"},
            {"N": 4, "K": 8, "name": "4x8 Wide"},
            {"N": 8, "K": 4, "name": "8x4 Tall"},
        ]

        for config in configs:
            N, K = config["N"], config["K"]
            print(f"\n=== Testing {config['name']} ===")

            # Create random input matrix
            input_matrix = np.zeros((N * K, N * K))
            values = np.random.uniform(-1, 1, (N, K))
            for i in range(N):
                for j in range(K):
                    input_matrix[i * K + j, i * K + j] = values[i, j]

            # Expected: each output j gets average of inputs
            expected = np.zeros((K, K))
            for j in range(K):
                expected[j, j] = np.mean(values[:, j])

            start_time = time.time()
            sum_step = SUMStep()
            circuit, scale = sum_step.apply_sum(input_matrix, N, K)
            compute_time = time.time() - start_time

            print(f"Computation time: {compute_time:.4f}s")
            self.verify_unitary(circuit, expected, scale, config['name'])

    def test_edge_cases(self):
        """Test edge cases for summation"""
        N, K = 4, 4
        sum_step = SUMStep()

        cases = {
            "uniform_input": {
                "values": np.ones((N, K)),
                "desc": "All inputs are 1",
            },
            "alternating_input": {
                "values": np.array([[1, -1] * (K // 2)] * N),
                "desc": "Alternating +1/-1 inputs",
            },
            "zero_input": {
                "values": np.zeros((N, K)),
                "desc": "All zero inputs",
            },
            "single_nonzero": {
                "values": np.eye(N, K),
                "desc": "Only diagonal elements are 1",
            }
        }

        for case_name, case_data in cases.items():
            print(f"\nTesting {case_name}: {case_data['desc']}")

            # Create input matrix
            input_matrix = np.zeros((N * K, N * K))
            values = case_data["values"]
            for i in range(N):
                for j in range(K):
                    input_matrix[i * K + j, i * K + j] = values[i, j]

            # Expected: each output j gets average of inputs
            expected = np.zeros((K, K))
            for j in range(K):
                expected[j, j] = np.mean(values[:, j])

            # Time the computation
            start_time = time.time()
            circuit, scale = sum_step.apply_sum(input_matrix, N, K)
            compute_time = time.time() - start_time

            print(f"Computation time: {compute_time:.4f}s")
            self.verify_unitary(circuit, expected, scale, f"Edge Case: {case_name}")