import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from torch import Tensor


@dataclass
class FixedKANConfig:
    """Configuration for Fixed Architecture KAN"""
    network_shape: List[int]  # [input_dim, ..., output_dim]
    max_degree: int  # Maximum polynomial degree
    complexity_weight: float = 0.1
    trainable_coefficients: bool = False

class KANNeuron(nn.Module):
    """
    Individual neuron that uses cumulative polynomials up to selected degree
    """
    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.max_degree = max_degree
        self.register_buffer('selected_degree', torch.tensor([-1], dtype=torch.long))
        self.coefficients = None #Store coefficients after optimization
        #Projection weights/bias for multi-dim input -> scalar
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(input_dim))
    @property
    def degree(self) -> int:
        degree_val = self.selected_degree.item()
        if degree_val < 0:
            raise RuntimeError("Degree not set. Run optimization first.")
        return degree_val

    def set_coefficients(self, coeffs_list: torch.Tensor, train_coeffs: bool = False):
        """Set coefficients as a ParameterList"""
        # For degree d, we expect d+1 coefficients for T_0(x), T_1(x), ..., T_d(x)
        if len(coeffs_list) != self.degree + 1:
            raise ValueError(f'Expected {self.degree + 1} coefficients, got {len(coeffs_list)}')

        # Convert to ParameterList
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.tensor(coeff.clone().detach()), requires_grad=train_coeffs) for coeff in coeffs_list
        ])

    def _compute_cumulative_transform(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute transforms using all polynomials up to degree"""
        transforms = []
        if x.dim() == 1:
            print('Expanding 1D input')
            x = x.unsqueeze(-1)
        print(f'After reshape x shape: {x.shape}')
        alpha = x.matmul(self.w) + self.b
        print(f'After projectioin alpha shape {alpha.shape}')
        #Compute T_k(alpha) for k = 0.. degree
        for d in range(degree + 1):
            t_d = torch.special.chebyshev_polynomial_t(alpha, n=d)
            transforms.append(t_d.unsqueeze(1))

        X = torch.cat(transforms, dim=1)
        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cumulative polynomials"""
        if self.selected_degree is None:
            raise RuntimeError("Neuron degree not set. Run optimization first.")
        if self.coefficients is None:
            raise RuntimeError("Neuron coefficients not set. Run optimization first.")

        # Get transform
        transform = self._compute_cumulative_transform(x, self.selected_degree)
        #TODO: we need to highlight the dimensions here for future.
        # I realize the unsqueeze to make it [batch_size, 1] was already done in the compute function
        return torch.matmul(transform, torch.stack([coeffs for coeffs in self.coefficients]))

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
        # "Big W" for combining the stacked neuron outputs => shape [output_dim, output_dim].
        # We'll also have a bias => shape [output_dim].
        self.combine_W = nn.Parameter(torch.eye(output_dim))
        self.combin_b = nn.Parameter(torch.zeros(output_dim))

    def optimize_degrees(self, x_data: torch.Tensor, y_data: torch.Tensor, train_coeffs: bool) -> None:
        """Use QUBO to select optimal degrees for each neuron"""
        print(f'\nKANLayer optimize_degrees input shape: {x_data.shape}')
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
                print(f"\nBefore transform - x_data shape: {x_data.shape}")
                # Get transform matrix [batch_size, d+1]
                X = neuron._compute_cumulative_transform(x_data, d)
                print(f"After transform - X shape: {X.shape}")
                # Solve least squares: X @ β = y
                coeffs = torch.linalg.lstsq(X, y_data).solution  # [(d+1), 1]
                degree_coeffs[neuron_idx][d] = coeffs

                y_pred = torch.matmul(X, coeffs)  # [batch_size, 1]
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
                    # Set degree
                    self.neurons[neuron_idx].selected_degree[0] = d
                    # Set coefficients as ParameterList
                    coeffs_list = degree_coeffs[neuron_idx][d]
                    self.neurons[neuron_idx].set_coefficients(coeffs_list, train_coeffs=train_coeffs)
                    break
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining all neuron outputs"""
        # Get each neuron's contribution
        neuron_outs = [neuron(x) for neuron in self.neurons] # list of [batch_size, 1]
        stack_out = torch.cat(neuron_outs, dim=1) # => [batch_size, output_dim]
        # Now apply the "big W" for combining => shape [batch_size, output_dim]
        # Then add a bias for each dimension
        # a reminder that the bigW here is a vector with w_i for each of the neuron_i(x)s
        return torch.matmul(stack_out, self.combine_W) + self.combin_b
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
        print(f'Initial x_data shape: {x_data.shape}')
        for i, layer in enumerate(self.layers):
            print(f'\nLayer {i} input shape: {current_input.shape}')
            # For last layer use y_data, otherwise use intermediate target
            if i == len(self.layers) - 1:
                target = y_data
            else:
                # TODO: Add intermediate target computation
                target = y_data

            layer.optimize_degrees(current_input, target, train_coeffs=self.config.trainable_coefficients)
            # Update input for next layer
            with torch.no_grad():
                current_input = layer(current_input)
                print(f'Layer {i} output shape: {current_input.shape}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        current = x
        for layer in self.layers:
            current = layer(current)
        return current
    def save_model(self, path: str):
        """Save model state including network shape, degrees, and coefficients"""
        # Get degrees for each neuron
        degrees = {}
        for layer_idx, layer in enumerate(self.layers):
            degrees[layer_idx] = {}
            for neuron_idx, neuron in enumerate(layer.neurons):
                degrees[layer_idx][neuron_idx] = neuron.degree

        # Save everything
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'degrees': degrees  # Save degrees separately
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'FixedKAN':
        """Load model from file"""
        checkpoint = torch.load(path)

        # Create model with config
        model = cls(checkpoint['config'])

        # Initialize the structure by setting degrees
        # This will create the ParameterLists
        for layer_idx, layer_degrees in checkpoint['degrees'].items():
            for neuron_idx, degree in layer_degrees.items():
                neuron = model.layers[layer_idx].neurons[neuron_idx]
                neuron.selected_degree[0] = degree
                # Initialize empty coefficient list of correct size
                neuron.coefficients = nn.ParameterList([
                    nn.Parameter(torch.zeros(1)) for _ in range(degree + 1)
                ])

        # Now load the state dict with coefficients
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train_model(
            self,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            num_epochs: int = 50,
            lr: float = 1e-3,
            complexity_weight: float = 0.1,
            do_qubo: bool = True
    ):
        """
        1) (Optionally) run QUBO-based optimize() to select polynomial degrees and
            set Chebyshev coefficients (frozen).
        2) Gather trainable parameters: neuron (w, b) + layer combine_W, combine_b.
        3) Train with MSE + optional L2 penalty on w.
        :param x_data: input data
        :param y_data: ground truths
        :param num_epochs: number of training epochs
        :param lr: learning rate
        :param complexity_weight: complexity penalty
        :param do_qubo: use QUBO to select polynomials
        :return:
        """
        if do_qubo:
            print('[(train_model)] Running QUBO-based optimize first...')
            self.optimize(x_data, y_data)
        params_to_train = []
        for layer in self.layers:
            params_to_train.append(layer.combine_W)
            params_to_train.append(layer.combine_b)

            for neuron in layer.neurons:
                params_to_train.append(neuron.w)
                params_to_train.append(neuron.b)

        optimizer = torch.optim.Adam(params_to_train, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            y_pred = self.forward(x_data)
            mse = torch.mean((y_pred - y_data) **2)

            w_norm = 0.0
            # for layer in self.layers:
            #     for neuron in layer.neurons:
            #         w_norm += torch.norm(neuron.w, p=2) ** 2
            #
            loss = mse + complexity_weight * w_norm
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss.item():6f}, MSE={mse.item():.6f}')
        print('[(train_model] Done training!')

    def analyze_network(self, x_data: torch.Tensor) -> dict[str, dict[str, int | Tensor | list[Any]]]:
        """
        Analyze network showing each neuron's output and the combined layer output.
        This version matches how forward() computes alpha = x @ w + b, then T_k(alpha).
        """
        analysis = {}
        current_input = x_data

        for layer_idx, layer in enumerate(self.layers):
            neuron_outputs = []
            degrees = []

            # For each neuron in the layer
            for neuron in layer.neurons:
                # 1. Project input to scalar alpha
                alpha = current_input @ neuron.w + neuron.b  # shape [batch_size]

                # 2. Compute Chebyshev polynomials: T_0, ..., T_degree
                poly_list = []
                deg = neuron.degree  # read from selected_degree
                for d in range(deg + 1):
                    poly_d = torch.special.chebyshev_polynomial_t(alpha, n=d)
                    poly_list.append(poly_d.unsqueeze(1))  # shape [batch_size, 1]

                # 3. Concatenate polynomials along dim=1
                X = torch.cat(poly_list, dim=1)  # shape [batch_size, deg+1]

                # 4. Multiply by the neuron's coefficients
                coeffs = torch.stack([c for c in neuron.coefficients])  # shape [deg+1]
                output = X @ coeffs  # shape [batch_size]

                # Save the neuron output
                neuron_outputs.append(output.unsqueeze(-1))  # [batch_size, 1]
                degrees.append(deg)

            # Stack all neuron outputs in this layer => [num_neurons, batch_size, 1]
            neuron_outputs = torch.stack(neuron_outputs, dim=1)

            # Sum across all neurons => layer output [batch_size, 1] if you are summing
            combined_output = neuron_outputs.sum(dim=0)

            analysis[f'layer_{layer_idx}'] = {
                'neuron_outputs': neuron_outputs,      # [num_neurons, batch_size, 1]
                'degrees': degrees,                    # list of int
                'combined_output': combined_output,    # [batch_size, 1]
                'input_dim': current_input.shape[1],
            }

            # Update current_input for the next layer
            current_input = combined_output

        return analysis

    def visualize_analysis(self, analysis: Dict[str, torch.Tensor], x_data: torch.Tensor,
                           y_data: Optional[torch.Tensor] = None) -> None:
        """
        Visualize network analysis for multivariate input.
        Args:
            analysis: Dictionary from analyze_network()
            x_data: Input tensor [batch_size, input_dim]
            y_data: Optional target tensor [batch_size, 1]
        """

        num_layers = len(self.layers)
        input_dim = x_data.shape[1]

        if input_dim == 2:
            # 2D visualization with surface and contour plots
            fig = plt.figure(figsize=(20, 8*num_layers))
            gs = plt.GridSpec(num_layers, 3)

            for layer_idx in range(num_layers):
                layer_data = analysis[f'layer_{layer_idx}']
                x_plot = x_data.detach().cpu().numpy()

                # Sort points for better surface plotting
                sort_idx = np.lexsort((x_plot[:, 1], x_plot[:, 0]))
                x_plot = x_plot[sort_idx]

                # 3D surface plot
                ax1 = fig.add_subplot(gs[layer_idx, 0], projection='3d')

                # Plot neuron outputs
                neuron_outputs = layer_data['neuron_outputs'].detach().cpu().numpy()
                for i, (output, degree) in enumerate(zip(neuron_outputs, layer_data['degrees'])):
                    output = output.squeeze()[sort_idx]
                    ax1.scatter(x_plot[:, 0], x_plot[:, 1], output,
                                alpha=0.3, label=f'Neuron {i} (deg={degree})')

                # Plot combined output
                combined = layer_data['combined_output'].detach().cpu().numpy().squeeze()[sort_idx]
                ax1.scatter(x_plot[:, 0], x_plot[:, 1], combined,
                            c='red', alpha=0.7, label='Layer Output')

                if layer_idx == num_layers-1 and y_data is not None:
                    y_plot = y_data.detach().cpu().numpy().squeeze()[sort_idx]
                    ax1.scatter(x_plot[:, 0], x_plot[:, 1], y_plot,
                                c='black', alpha=0.5, label='Target')

                ax1.set_title(f'Layer {layer_idx+1} Contributions')
                ax1.set_xlabel('Input 1')
                ax1.set_ylabel('Input 2')
                ax1.set_zlabel('Output')
                ax1.legend()

                # Create regular grid for interpolation
                n_grid = 50
                x1_unique = np.linspace(x_plot[:, 0].min(), x_plot[:, 0].max(), n_grid)
                x2_unique = np.linspace(x_plot[:, 1].min(), x_plot[:, 1].max(), n_grid)
                X1, X2 = np.meshgrid(x1_unique, x2_unique)

                # Contour plot of combined output
                ax2 = fig.add_subplot(gs[layer_idx, 1])
                grid_points = np.column_stack([X1.ravel(), X2.ravel()])
                Z = griddata(x_plot, combined, (X1, X2), method='cubic')
                contour = ax2.contourf(X1, X2, Z, levels=20, cmap='viridis')
                plt.colorbar(contour, ax=ax2)

                ax2.set_title(f'Layer {layer_idx+1} Output Contours')
                ax2.set_xlabel('Input 1')
                ax2.set_ylabel('Input 2')

                # Degree distribution
                ax3 = fig.add_subplot(gs[layer_idx, 2])
                ax3.hist(layer_data['degrees'], bins=range(self.config.max_degree + 2),
                         alpha=0.7, rwidth=0.8)
                ax3.set_title(f'Layer {layer_idx+1} Degree Distribution')
                ax3.set_xlabel('Degree')
                ax3.set_ylabel('Count')

        else:
            # For 1D or higher dimensions, show summary plots
            fig = plt.figure(figsize=(15, 5*num_layers))
            gs = plt.GridSpec(num_layers, 2)

            for layer_idx in range(num_layers):
                layer_data = analysis[f'layer_{layer_idx}']

                # Plot combined output against first two dimensions
                ax1 = fig.add_subplot(gs[layer_idx, 0])
                x_plot = x_data.detach().cpu().numpy()
                combined = layer_data['combined_output'].detach().cpu().numpy().squeeze()

                if input_dim == 1:
                    ax1.scatter(x_plot, combined, alpha=0.5)
                    ax1.set_xlabel('Input')
                else:
                    scatter = ax1.scatter(x_plot[:, 0], x_plot[:, 1], c=combined,
                                          cmap='viridis', alpha=0.5)
                    plt.colorbar(scatter, ax=ax1)
                    ax1.set_xlabel('Input 1')
                    ax1.set_ylabel('Input 2')

                ax1.set_title(f'Layer {layer_idx+1} Output')

                # Degree distribution
                ax2 = fig.add_subplot(gs[layer_idx, 1])
                ax2.hist(layer_data['degrees'], bins=range(self.config.max_degree + 2),
                         alpha=0.7, rwidth=0.8)
                ax2.set_title(f'Layer {layer_idx+1} Degree Distribution')
                ax2.set_xlabel('Degree')
                ax2.set_ylabel('Count')

        plt.tight_layout()
        plt.show()