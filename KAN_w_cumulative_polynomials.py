import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FixedKANConfig:
    """Configuration for Fixed Architecture KAN"""
    network_shape: List[int]  # [input_dim, ..., output_dim]
    max_degree: int  # Maximum polynomial degree
    complexity_weight: float = 0.1

class KANNeuron(nn.Module):
    """
    Individual neuron that uses cumulative polynomials up to selected degree
    """
    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.max_degree = max_degree
        self.selected_degree = None  # Will be set by QUBO
        self.coefficients = None #Store coefficients after optimization
    def _compute_cumulative_transform(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute transforms using all polynomials up to degree"""
        transforms = []
        for d in range(degree + 1):  # Use ALL polynomials up to degree
            transform = torch.special.chebyshev_polynomial_t(x.squeeze(), n=d)
            transforms.append(transform)
        return torch.stack(transforms, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cumulative polynomials"""
        if self.selected_degree is None:
            raise RuntimeError("Neuron degree not set. Run optimization first.")
        if self.coefficients is None:
            raise RuntimeError("Neuron coefficients not set. Run optimization first.")

        # Get transform
        transform = self._compute_cumulative_transform(x, self.selected_degree)
        # Apply stored coefficients
        return transform @ self.coefficients

class KANLayer(nn.Module):
    """
    Layer of KAN with fixed number of neurons
    """
    def __init__(self, input_dim: int, output_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_degree = max_degree

        # Create fixed number of neurons
        self.neurons = nn.ModuleList([
            KANNeuron(input_dim, max_degree)
            for _ in range(output_dim)
        ])

    def optimize_degrees(self, x_data: torch.Tensor, y_data: torch.Tensor) -> None:
        """Use QUBO to select optimal degrees for each neuron"""
        from pyqubo import Array
        import neal

        # Create binary variables for degree selection
        q = Array.create('q', shape=(self.output_dim, self.max_degree + 1),
                         vartype='BINARY')

        # Build QUBO
        H = 0.0
        degree_coeffs = {}
        # Evaluate each degree option
        for neuron_idx in range(self.output_dim):
            neuron = self.neurons[neuron_idx]
            degree_coeffs[neuron_idx] = {}
            for d in range(self.max_degree + 1):
                # Get transform matrix [batch_size, d+1]
                X = neuron._compute_cumulative_transform(x_data, d)

                # Solve least squares: X @ β = y
                coeffs = torch.linalg.lstsq(X, y_data).solution  # [(d+1), 1]
                degree_coeffs[neuron_idx][d] = coeffs

                y_pred = X @ coeffs  # [batch_size, 1]
                mse = torch.mean((y_data - y_pred) ** 2)

                # Add to QUBO
                H += mse * q[neuron_idx, d]
        # Add one-hot constraint - exactly one degree per neuron
        for i in range(self.output_dim):
            constraint = (sum(q[i, d] for d in range(self.max_degree + 1)) - 1)**2
            H += 10.0 * constraint

            # Solve QUBO
        model = H.compile()
        bqm = model.to_bqm()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1000)

        # Extract results
        best_sample = min(model.decode_sampleset(sampleset),
                              key=lambda x: x.energy).sample

        # Set degrees
        for neuron_idx in range(self.output_dim):
            for d in range(self.max_degree + 1):
                if best_sample[f'q[{neuron_idx}][{d}]'] == 1:
                    self.neurons[neuron_idx].selected_degree = d
                    self.neurons[neuron_idx].coefficients = degree_coeffs[neuron_idx][d]
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining all neuron outputs"""
        # Get each neuron's contribution
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(x))

        # Combine outputs (sum as per KAN theorem)
        return torch.stack(outputs).sum(dim=0)

class FixedKAN(nn.Module):
    """
    Complete Fixed Architecture KAN
    """
    def __init__(self, config: FixedKANConfig):
        super().__init__()
        self.config = config

        # Create fixed layers
        self.layers = nn.ModuleList([
            KANLayer(
                config.network_shape[i],
                config.network_shape[i+1],
                config.max_degree
            )
            for i in range(len(config.network_shape)-1)
        ])

    def optimize(self, x_data: torch.Tensor, y_data: torch.Tensor) -> None:
        """Optimize degrees for all layers"""
        current_input = x_data
        for i, layer in enumerate(self.layers):
            # For last layer use y_data, otherwise use intermediate target
            if i == len(self.layers) - 1:
                target = y_data
            else:
                # TODO: Add intermediate target computation
                target = y_data

            layer.optimize_degrees(current_input, target)
            # Update input for next layer
            with torch.no_grad():
                current_input = layer(current_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        current = x
        for layer in self.layers:
            current = layer(current)
        return current

    def analyze_network(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze network showing neuron contributions"""
        analysis = {}
        current_input = x

        for layer_idx, layer in enumerate(self.layers):
            # Get each neuron's contribution
            neuron_outputs = []
            for neuron in layer.neurons:
                output = neuron(current_input)
                neuron_outputs.append(output)

            # Store analysis
            analysis[f'layer_{layer_idx}'] = {
                'neuron_outputs': torch.stack(neuron_outputs),
                'degrees': [n.selected_degree for n in layer.neurons],
                'combined_output': torch.stack(neuron_outputs).sum(dim=0)
            }

            # Update input for next layer
            current_input = analysis[f'layer_{layer_idx}']['combined_output']

        return analysis

    def visualize_analysis(
            self,
            analysis: Dict[str, torch.Tensor],
            x_data: torch.Tensor,
            y_data: Optional[torch.Tensor] = None
    ) -> None:
        """Visualize network behavior"""
        import matplotlib.pyplot as plt
        num_layers = len(self.layers)

        fig = plt.figure(figsize=(15, 5*num_layers))
        gs = plt.GridSpec(num_layers, 2)

        for layer_idx in range(num_layers):
            layer_data = analysis[f'layer_{layer_idx}']

            # Plot contributions
            ax1 = fig.add_subplot(gs[layer_idx, 0])
            x_plot = x_data.detach().cpu().numpy()

            # Plot original data for final layer
            if layer_idx == num_layers-1 and y_data is not None:
                y_plot = y_data.detach().cpu().numpy()
                ax1.scatter(x_plot, y_plot, alpha=0.3, label='Target')

            # Plot neuron outputs
            neuron_outputs = layer_data['neuron_outputs'].detach().cpu().numpy()
            for i, (output, degree) in enumerate(zip(
                    neuron_outputs,
                    layer_data['degrees']
            )):
                ax1.plot(x_plot, output, '--', alpha=0.7,
                         label=f'Neuron {i} (deg={degree})')

            # Plot combined output
            combined = layer_data['combined_output'].detach().cpu().numpy()
            ax1.plot(x_plot, combined, 'r-', linewidth=2,
                     label='Layer Output')

            ax1.set_title(f'Layer {layer_idx+1} Contributions')
            ax1.legend()
            ax1.grid(True)

            # Plot degree distribution
            ax2 = fig.add_subplot(gs[layer_idx, 1])
            degrees = layer_data['degrees']
            ax2.hist(degrees, bins=range(self.config.max_degree + 2),
                     alpha=0.7, rwidth=0.8)
            ax2.set_title(f'Layer {layer_idx+1} Degree Distribution')
            ax2.set_xlabel('Degree')
            ax2.set_ylabel('Count')

        plt.tight_layout()
        plt.show()