import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Operator


def create_R_operator(x: float) -> np.ndarray:
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1,1]")
    sqrt_1_minus_x2 = np.sqrt(1 - x ** 2)
    return np.array([[x, sqrt_1_minus_x2], [sqrt_1_minus_x2, -x]])


def decompose_R_operator(circuit: QuantumCircuit, qubit: int, x: float):
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1,1]")
    theta = 2 * np.arccos(x)
    circuit.ry(theta, qubit)
    circuit.z(qubit)


def construct_chebyshev_circuit(x: int, d: int) -> QuantumCircuit:
    """
        Construct a quantum circuit that implements the Chebyshev polynomial T_d(x)
        using quantum signal processing as per Proposition 6.

        Args:
            x (float): Input value in [-1, 1]
            d (int): Degree of the Chebyshev polynomial

        Returns:
            QuantumCircuit: Quantum circuit implementing the Chebyshev polynomial
    """
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1,1]")
    if d < 1 or not isinstance(d, int):
        raise ValueError("d must be a positive integer")
    circuit = QuantumCircuit(1)
    theta = 2 * np.arccos(x)

    phi_list = [(1 - d) * np.pi / 2] + [np.pi / 2] * (d - 1)

    for phi in phi_list:
        circuit.rz(2*phi, 0)
        circuit.ry(theta, 0)
        circuit.z(0)
    return circuit


class TestROperatorDecomposition(unittest.TestCase):
    def test_decompose_R_operator(self):
        test_values = [-1, -0.5, 0, 0.5, 1]
        for x in test_values:
            with self.subTest(x=x):
                # Create the expected R(x) operator
                expected_R = create_R_operator(x)

                # Build the quantum circuit
                circuit = QuantumCircuit(1)
                decompose_R_operator(circuit, 0, x)

                # Get the unitary matrix of the circuit
                simulator = Aer.get_backend('unitary_simulator')
                compiled_circuit = transpile(circuit, simulator)
                result = simulator.run(compiled_circuit).result()
                unitary = result.get_unitary(compiled_circuit)

                # Extract the 2x2 unitary matrix
                circuit_matrix = np.array(unitary)

                # Compare the absolute values of the matrices
                np.testing.assert_allclose(
                    np.abs(circuit_matrix),
                    np.abs(expected_R),
                    atol=1e-7,
                    err_msg=f"Mismatch for x = {x}"
                )

    def test_invalid_x_value(self):
        with self.assertRaises(ValueError):
            create_R_operator(1.5)
        with self.assertRaises(ValueError):
            create_R_operator(-1.5)
        circuit = QuantumCircuit(1)
        with self.assertRaises(ValueError):
            decompose_R_operator(circuit, 0, 1.5)

class TestChebyshevCircuit(unittest.TestCase):
    def test_chebyshev_circuit(self):
        # Test for multiple values of x and degrees d
        test_cases = [
            {'x': 0.5, 'd': 1},
            {'x': 0.5, 'd': 2},
            {'x': 0.5, 'd': 3},
            {'x': 0.5, 'd': 4},
            {'x': 0.5, 'd': 5},
            {'x': -0.5, 'd': 3},
            {'x': 0, 'd': 2},
            {'x': 1, 'd': 1},
            {'x': -1, 'd': 2},
        ]

        for case in test_cases:
            x = case['x']
            d = case['d']
            with self.subTest(x=x, d=d):
                # Construct the circuit
                circuit = construct_chebyshev_circuit(x, d)

                # Get the unitary matrix
                simulator = Aer.get_backend('unitary_simulator')
                compiled_circuit = transpile(circuit, simulator)
                result = simulator.run(compiled_circuit).result()
                unitary = result.get_unitary(compiled_circuit)

                # Extract the (0,0) element
                U00 = unitary[0, 0]

                # Compute expected T_d(x)
                Td = np.cos(d * np.arccos(x))

                # Since the unitary might have a global phase, compare the absolute values
                self.assertTrue(np.isclose(np.abs(U00), np.abs(Td), atol=1e-7),
                                f"Mismatch for x = {x}, d = {d}")

    def test_invalid_inputs(self):
        # Test invalid x
        with self.assertRaises(ValueError):
            construct_chebyshev_circuit(1.5, 3)
        with self.assertRaises(ValueError):
            construct_chebyshev_circuit(-1.5, 3)
        # Test invalid d
        with self.assertRaises(ValueError):
            construct_chebyshev_circuit(0.5, 0)
        with self.assertRaises(ValueError):
            construct_chebyshev_circuit(0.5, -1)
        with self.assertRaises(ValueError):
            construct_chebyshev_circuit(0.5, 2.5)
# Run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
