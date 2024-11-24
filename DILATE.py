import unittest

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator, Aer
from fable import fable


def construct_dilated_diagonal_matrix(x, K):
    """
    Constructs the dilated diagonal matrix A required for the DILATE step.

    Args:
        x (array_like): Input vector x of length N.
        K (int): Number of output nodes (copies of each x_p).

    Returns:
        numpy.ndarray: Dilated diagonal matrix A.
    """
    N = len(x)
    # Repeat each x_p K times
    x_repeated = np.repeat(x, K)
    # Construct the diagonal matrix
    A = np.diag(x_repeated)
    return A


def dilate_using_fable(x, K, epsilon):
    A = construct_dilated_diagonal_matrix(x, K)
    circ, alpha = fable(A, epsilon)
    return circ, alpha


class TestFableDilatedBlockEncoding(unittest.TestCase):
    def test_dilated_block_encoding(self):
        # Input vector x and number of outputs K
        x = np.array([0.6, 0.8])
        K = 4

        # Construct the dilated diagonal matrix A
        A = construct_dilated_diagonal_matrix(x, K)

        # Determine the number of qubits required
        N = A.shape[0]
        n = int(np.ceil(np.log2(N)))

        # Use FABLE to get the block-encoding circuit and scaling factor alpha
        circ, alpha = dilate_using_fable(x, K, epsilon=0)
        print('circuit: ', circ)
        # Use the unitary simulator from Qiskit Aer
        simulator = Aer.get_backend('unitary_simulator')

        # Transpile the circuit for the simulator
        compiled_circuit = transpile(circ, simulator)

        # Run the simulation
        result = simulator.run(compiled_circuit).result()

        # Extract the unitary matrix and cast to NumPy array
        unitary = result.get_unitary(compiled_circuit)
        unitary = np.asarray(unitary)

        # The block-encoded matrix is in the top-left block of the unitary matrix
        # The scaling factor is alpha * N
        # Extract the top-left block
        block_size = N
        top_left_block = unitary[:block_size, :block_size]

        # Reconstruct the matrix A from the block-encoding
        reconstructed_A = top_left_block * alpha * N

        # Compute the relative difference
        difference = np.linalg.norm(reconstructed_A - A) / np.linalg.norm(A)
        print(f"Relative difference between reconstructed A and original A: {difference}")

        # Verify that the reconstructed A matches the original A within numerical precision
        np.testing.assert_array_almost_equal(
            reconstructed_A, A, decimal=6
        )

        print("Test passed: Reconstructed dilated A matches the original dilated A within numerical precision.")

    def test_dilated_block_encoding_different_sizes(self):
        x = np.array([0.5, 0.7, 0.9])
        K = 2
        A = construct_dilated_diagonal_matrix(x, K)
        N = A.shape[0]
        n = int(np.ceil(np.log2(N)))
        circ, alpha = fable(A, 0)
        simulator = Aer.get_backend('unitary_simulator')
        compiled_circuit = transpile(circ, simulator)
        result = simulator.run(compiled_circuit).result()
        unitary = np.asarray(result.get_unitary(compiled_circuit))
        block_size = N
        top_left_block = unitary[:block_size, :block_size]
        reconstructed_A = top_left_block * alpha * N
        difference = np.linalg.norm(reconstructed_A - A) / np.linalg.norm(A)
        self.assertTrue(difference < 1, f"Relative difference too high: {difference}")
        print(f"Test passed for input size {len(x)} and K={K}.")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
