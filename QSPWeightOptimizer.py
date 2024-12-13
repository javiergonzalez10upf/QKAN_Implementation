from typing import List, Dict, Optional, Tuple
import pennylane as qml
import polars as pl
import numpy as np
import torch

from BaseOptimizer import BaseOptimizer
from QKAN_Steps.ChebyshevStep import ChebyshevStep
from QKAN_Steps.QKANLayer import QKANLayer


class QSPOptimizer(BaseOptimizer):
    def __init__(
            self,
            network_shape: List[int],
            optimized_degrees: Dict[str, int],
            validation_weight: float = 0.5,
            random_seed: Optional[int] = None
    ):
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.network_shape = network_shape
        self.optimized_degrees = optimized_degrees
        self.validation_weight = validation_weight

        # Initialize QSP angles
        self.qsp_angles = {}
        for layer_idx, max_degree in optimized_degrees.items():
            angles = torch.rand(max_degree + 1, requires_grad=True) * 2 * torch.pi
            self.qsp_angles[layer_idx] = torch.nn.Parameter(angles)

    def _compute_transforms(self, feature_data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Implementation of abstract method from BaseOptimizer.
        Compute and cache Chebyshev transforms.
        """
        transforms = {}
        for d in range(max(self.optimized_degrees.values()) + 1):
            cheb_step = ChebyshevStep(degree=d)
            transforms[d] = cheb_step.transform_diagonal(feature_data)
        return transforms

    def _build_qsp_circuit(
            self,
            layer_idx: str,
            input_data: pl.DataFrame,
            fold_id: Optional[str] = None
    ) -> Tuple[qml.QNode, float]:
        """
        Build QSP circuit using collapsed combinations.
        """
        # Get collapsed combinations using BaseOptimizer method
        collapsed_combinations = self._compute_collapsed_combinations(input_data, fold_id)

        # Get circuit dimensions
        N = self.network_shape[int(layer_idx)]
        K = self.network_shape[int(layer_idx) + 1]
        max_degree = self.optimized_degrees[layer_idx]

        # Setup quantum device
        n_qubits = N + K + 2  # Input + Output + 2 ancilla
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(phi: torch.Tensor):
            """QSP circuit using collapsed combinations"""
            qml.Hadamard(wires=0)

            for angle in phi[:-1]:
                qml.RZ(float(angle), wires=0)
                # Build block encoding from collapsed combinations
                block_encoding = QKANLayer.build_block_encoding(
                    collapsed_combinations,
                    max_degree,
                    N,
                    K
                )
                qml.QubitUnitary(block_encoding, wires=range(n_qubits))

            qml.RZ(float(phi[-1]), wires=0)
            qml.Hadamard(wires=0)

            return qml.expval(qml.PauliZ(0))

        return circuit, 1.0

    def optimize_weights(
            self,
            training_data: Dict[str, pl.DataFrame],
            cv_strategy: str = 'expanding_window',
            num_epochs: int = 1000,
            learning_rate: float = 1e-3
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Optimize QSP angles using collapsed combinations and cross-validation.
        """
        optimized_params = {}

        for layer_idx in self.optimized_degrees.keys():
            x_data = training_data[f'layer_{layer_idx}_input']
            y_data = training_data[f'layer_{layer_idx}_output']

            # Get folds based on strategy
            if cv_strategy == 'expanding_window':
                folds = self._get_expanding_window_folds(x_data)
            else:
                folds = self._get_time_based_folds(x_data)

            layer_params = {}
            for fold_idx, (train_mask, val_mask) in enumerate(folds):
                fold_id = f"fold_{fold_idx}"

                # Get train/val data
                train_data = x_data.filter(train_mask)
                val_data = x_data.filter(val_mask)

                # Build and optimize circuit
                train_circuit, _ = self._build_qsp_circuit(layer_idx, train_data, fold_id)
                val_circuit, _ = self._build_qsp_circuit(layer_idx, val_data, fold_id)

                # TODO: Implement optimization loop similar to function_fitting_qsp.py
                # Using train_circuit and val_circuit

            optimized_params[layer_idx] = layer_params

        return optimized_params