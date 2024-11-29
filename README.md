# QKAN Implementation

An implementation of Quantum Kolmogorov-Arnold Networks (QKAN) focusing on efficient model optimization through quantum computing frameworks. This project explores the intersection of quantum computing and neural networks, implementing a step-based architecture for quantum-compatible neural network training.

## Overview

QKAN introduces a novel approach to neural network optimization by adapting Kolmogorov-Arnold Networks for quantum systems. The implementation breaks down complex quantum-classical interactions into clear, modular steps:

- **ChebyshevStep**: Implements quantum-compatible Chebyshev polynomial transformations 
- **MulStep**: Handles weighted polynomial operations in quantum context
- **LCUStep**: Manages Linear Combination of Unitaries for polynomial combination
- **SUMStep**: Performs efficient quantum summation operations
- **QKANLayer**: Orchestrates the complete quantum-neural network layer

## Key Features

- Modular, step-based quantum neural network implementation
- Clean integration with Qiskit quantum computing framework
- Comprehensive test suite for each component
- Efficient quantum state preparation and manipulation
- Clear separation of classical and quantum operations

## Technical Details

### Core Components

```python
class QKANLayer:
    """
    Quantum Kolmogorov-Arnold Network layer implementation.
    Orchestrates quantum neural network operations through distinct steps:
    1. Chebyshev polynomial transformations
    2. Quantum multiplication operations
    3. Linear combination of unitaries
    4. Quantum summation
    """
```

### Implementation Structure

- Each step is implemented as a separate class with clear responsibilities
- Extensive unit testing ensures reliable quantum operations
- Efficient quantum circuit construction and optimization
- Careful management of quantum resources and gate complexity

## Testing and Verification

Each component includes comprehensive unit tests demonstrating:
- Correct quantum state manipulation
- Accurate polynomial transformations
- Proper handling of quantum circuits
- Verification of unitary operations

## Getting Started

```python
# Example usage
layer = QKANLayer(N=4, K=4, max_degree=3)
x = np.random.uniform(-1, 1, 4)
weights = [np.random.uniform(-1, 1, 16) for _ in range(4)]
output = layer.forward(x, weights)
```

## Future Directions

Currently exploring:
- Enhanced model optimization techniques
- Improved quantum circuit efficiency
- Extended polynomial basis functions
- Integration with various quantum backends
- Active model training and performance evaluation in progress
- Investigating speedup potential in quantum vs classical implementations

## Dependencies

- Qiskit
- NumPy
- PyTest (for testing)
- FABLE (for quantum block encoding)