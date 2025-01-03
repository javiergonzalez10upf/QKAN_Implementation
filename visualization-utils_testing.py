import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional

from TorchDegreeOptimizer import DegreeOptimizer, DegreeOptimizerConfig


def plot_degree_optimization(
    optimizer: DegreeOptimizer,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    title: str = "Degree Optimization Results"
) -> None:
    """Plot degree optimization results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Input data and approximation
    plt.subplot(1, 3, 1)
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy()
    
    plt.scatter(x_np, y_np, alpha=0.5, label='Data')
    plt.title('Data and Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Plot 2: Degree distribution
    plt.subplot(1, 3, 2)
    degrees = [d for sublist in optimizer.optimal_degrees for d in sublist]
    plt.hist(degrees, bins=range(max(degrees) + 2), alpha=0.7)
    plt.title('Distribution of Selected Degrees')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    
    # Plot 3: Error metrics vs degree
    plt.subplot(1, 3, 3)
    scores, comp_r2 = optimizer.evaluate_degree(x_data, y_data)
    degrees = range(len(scores))
    
    plt.plot(degrees, scores.detach().cpu().numpy(), 'b-', label='MSE')
    plt.plot(degrees, comp_r2.detach().cpu().numpy(), 'r--', label='R²')
    plt.title('Error Metrics vs Degree')
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def run_interactive_test(
    test_functions: List[callable],
    optimizer: DegreeOptimizer,
    n_samples: int = 1000,
    noise_level: float = 0.1
) -> None:
    """Run interactive tests with visualization"""
    device = next(optimizer.parameters()).device
    
    for i, func in enumerate(test_functions):
        # Generate data
        x = torch.linspace(-1, 1, n_samples, device=device).reshape(-1, 1)
        y = func(x)
        if noise_level > 0:
            y += noise_level * torch.randn_like(y)
            
        # Optimize and plot
        optimizer.fit(x, y)
        plot_degree_optimization(
            optimizer, x, y,
            title=f'Test Function {i+1}: {func.__name__}'
        )
        
# Example usage:
if __name__ == "__main__":
    config = DegreeOptimizerConfig(
        network_shape=[1, 10, 1],
        max_degree=5,
        complexity_weight=0.1,
        significance_weight=0.05
    )
    optimizer = DegreeOptimizer(config)
    
    # Define test functions
    def polynomial(x: torch.Tensor) -> torch.Tensor:
        return 1.0 + 2.0*x + 3.0*x**2
        
    def sinusoidal(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2*torch.pi*x)
        
    def complex_function(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2*torch.pi*x) + 0.5*x**2
        
    test_functions = [polynomial, sinusoidal, complex_function]
    
    # Run tests with visualization
    run_interactive_test(test_functions, optimizer)