import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit_aer import AerSimulator, Aer
from qiskit.circuit.library import StatePreparation
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

def test_fable_dilated_block_encoding():


    # Input vector x and number of outputs K
    x = np.array([0.6, 0.8])
    K = 4

    # Construct the dilated diagonal matrix A
    A = construct_dilated_diagonal_matrix(x, K)

    # Determine the number of qubits required
    N = A.shape[0]
    n = int(np.ceil(np.log2(N)))

    # Use FABLE to get the block-encoding circuit and scaling factor alpha
    circ, alpha = fable(A)

    # Use the unitary simulator from Qiskit Aer
    simulator = Aer.get_backend('unitary_simulator')

    # Transpile the circuit for the simulator
    compiled_circuit = transpile(circ, simulator)

    # Run the simulation
    result = simulator.run(compiled_circuit).result()

    # Extract the unitary matrix
    unitary = np.asarray(result.get_unitary(compiled_circuit))

    # The block-encoded matrix is in the top-left block of the unitary matrix
    # The scaling factor is alpha * N
    # Extract the top-left block
    block_size = N
    top_left_block = unitary[:block_size, :block_size]
    np.set_printoptions(formatter={'all': lambda x: f'{x:.1f}'})
    # Reconstruct the matrix A from the block-encoding
    reconstructed_A = top_left_block * alpha * N
    print(reconstructed_A)
    print('A: ', A)
    # Compute the relative difference
    difference = np.linalg.norm(reconstructed_A - A) / np.linalg.norm(A)
    print(f"Relative difference between reconstructed A and original A: {difference}")

    # Verify that the reconstructed A matches the original A within numerical precision
    np.testing.assert_array_almost_equal(
        reconstructed_A, A, decimal=6
    )

    print("Test passed: Reconstructed dilated A matches the original dilated A within numerical precision.")

# Run the test
test_fable_dilated_block_encoding()



