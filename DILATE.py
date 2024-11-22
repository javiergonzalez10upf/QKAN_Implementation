import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit_aer import Aer


import numpy as np
from qiskit.circuit.library import StatePreparation


def construct_dilated_block_encoding(x, K):
    """
    Constructs a block-encoding of the dilated diagonal matrix as per the DILATE step.

    Args:
        x (array_like): Input vector x of length N.
        K (int): Number of output nodes (copies of each x_p).

    Returns:
        QuantumCircuit, float: Circuit implementing the dilated block-encoding and the scaling factor alpha.
    """
    if K < 0:
        raise ValueError('K is negative')
    N = len(x)
    n = int(np.ceil(np.log2(N)))  # Number of qubits for input vector
    k = int(np.ceil(np.log2(K)))  # Number of auxiliary qubits to dilate

    # Pad x to length 2^n if necessary
    x_padded = np.pad(x, (0, 2**n - N), 'constant')

    # Check if x_padded is all zeros
    if not np.any(x_padded):
        raise ValueError("Input vector x cannot be all zeros.")

    # State preparation normalization
    x_padded_norm = np.linalg.norm(x_padded)
    x_state = x_padded / x_padded_norm  # For state preparation (sum of squares equals 1)

    # Scaling factor alpha (maximum absolute value of x_padded)
    alpha = np.max(np.abs(x_padded))

    # Create registers
    x_register = QuantumRegister(n, name='x_reg')
    aux_register = QuantumRegister(k, name='aux_reg') if k > 0 else None
    ancilla_qubit = QuantumRegister(1, name='anc')  # One ancilla qubit
    circuit = QuantumCircuit(x_register, ancilla_qubit)

    if aux_register:
        circuit.add_register(aux_register)

    # Step 1: Prepare the state |ψ⟩ = sum x_p / ||x|| |p⟩ using StatePreparation
    # Initialize U_psi with x_register qubits
    U_psi = QuantumCircuit(x_register)
    U_psi.append(StatePreparation(x_state), x_register)

    # Step 2: Construct the block-encoding using controlled-U_psi and its adjoint
    # Apply H gate to ancilla qubit
    circuit.h(ancilla_qubit[0])

    # Controlled-U_psi (controlled on ancilla qubit being |1⟩)
    c_U_psi = U_psi.to_gate().control(1)
    circuit.append(c_U_psi, [ancilla_qubit[0]] + x_register[:])

    # Apply H gate to ancilla qubit
    circuit.h(ancilla_qubit[0])

    # Controlled-U_psi_dagger (controlled on ancilla qubit being |1⟩)
    U_psi_dagger = U_psi.inverse()
    c_U_psi_dagger = U_psi_dagger.to_gate().control(1)
    circuit.append(c_U_psi_dagger, [ancilla_qubit[0]] + x_register[:])

    # The circuit now represents the block-encoding of diag(x_padded) / alpha

    # Step 3: Append auxiliary qubits (if any)
    # No action required since I_k acts trivially on aux_register

    return circuit, alpha







class TestDilatedBlockEncoding(unittest.TestCase):
    def test_dilated_block_encoding(self):
        # Example input vector x and K
        x = np.array([0.6, 0.8])
        K = 4  # Number of output nodes

        # Construct the dilated block-encoding circuit
        dilated_circuit, alpha = construct_dilated_block_encoding(x, K)

        # Simulate the circuit using the unitary simulator
        backend = Aer.get_backend('unitary_simulator')
        compiled_circuit = transpile(dilated_circuit, backend)
        result = backend.run(compiled_circuit).result()
        unitary = result.get_unitary(compiled_circuit)

        # Get the number of qubits
        num_qubits = dilated_circuit.num_qubits
        dim_full = 2**num_qubits

        # Identify indices where ancilla qubit is in |0>
        k = int(np.ceil(np.log2(K)))
        ancilla_qubit = dilated_circuit.qubits[dilated_circuit.num_qubits - (1 + k)]
        ancilla_position = dilated_circuit.find_bit(ancilla_qubit).index

        # Generate the list of indices where the ancilla qubit is in |0>
        indices = [i for i in range(dim_full) if ((i >> ancilla_position) & 1) == 0]

        # Extract the top-left block corresponding to ancilla qubit in |0>
        top_left_block = unitary[np.ix_(indices, indices)]

        # Calculate expected diagonal entries
        N = len(x)
        n = int(np.ceil(np.log2(N)))
        x_padded = np.pad(x, (0, 2**n - N), 'constant')

        # Expected diagonal entries are x_padded / alpha, repeated 2^k times
        expected_diag = np.repeat(x_padded / alpha, 2**k)

        # The diagonal of the top-left block should match expected_diag
        actual_diag = np.diag(top_left_block)

        print('actual diag',actual_diag)
        print('expected diag',expected_diag)
        self.assertTrue(np.allclose(np.abs(actual_diag), expected_diag, atol=1e-7),
                        "The dilated block-encoding does not match the expected values.")

    def test_invalid_input(self):
        # Test with zero vector
        x = np.array([0, 0])
        K = 2
        with self.assertRaises(ValueError):
            construct_dilated_block_encoding(x, K)

        # Test with negative K
        x = np.array([0.6, 0.8])
        K = -1
        with self.assertRaises(ValueError):
            construct_dilated_block_encoding(x, K)

def main():
    x = np.array([0.6, 0.8])
    k = 1
    dilated_circuit,alpha = construct_dilated_block_encoding(x, k)

    print(dilated_circuit.draw())
# Run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
    main()
