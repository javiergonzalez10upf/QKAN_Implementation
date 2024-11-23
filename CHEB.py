import numpy as np
from fable import fable
from qiskit import QuantumCircuit
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)


def construct_dilated_diagonal_matrix(x, K):
    N = len(x)
    x_repeated = np.repeat(x, K)
    A = np.diag(x_repeated)
    return A


def dilate_using_fable(x, K, epsilon):
    A = construct_dilated_diagonal_matrix(x, K)
    circ, alpha = fable(A, epsilon)
    return circ, alpha


def chebyshev_target_function(x, d):
    return np.cos(d * np.arccos(x))


def approximate_sqrt_1_minus_x_squared(x, d_approx):
    func_sqrt = lambda x: np.sqrt(1 - np.square(x))
    poly_sqrt = PolyTaylorSeries().taylor_series(
        func=func_sqrt,
        degree=d_approx,
        max_scale=0.9,
        chebyshev_basis=True,
        cheb_samples=2 * d_approx
    )
    return poly_sqrt


def construct_R_Ux(A, d_approx, epsilon_p2):
    poly_sqrt = approximate_sqrt_1_minus_x_squared(np.diag(A), d_approx)
