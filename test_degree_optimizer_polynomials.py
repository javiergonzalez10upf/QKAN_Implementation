import unittest
import numpy as np
import polars as pl
from DegreeOptimizer import DegreeOptimizer
from QKAN_Steps.ChebyshevStep import ChebyshevStep
from numpy.polynomial.chebyshev import chebval

class TestDegreeOptimizerPolynomials(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.max_degree = 8
        self.network_shape = [1, 1]  # Simple shape for testing
        self.optimizer = DegreeOptimizer(
            network_shape=self.network_shape,
            max_degree=self.max_degree,
            complexity_weight=0.1
        )
        
    def generate_chebyshev_data(self, degrees: list, weights: list = None, 
                               num_points: int = 100, noise_level: float = 0.1,
                               time_ordered: bool = True):
        """
        Generate data from combination of Chebyshev polynomials
        Args:
            degrees: List of degrees to combine
            weights: List of weights for each polynomial (default: all 1.0)
            num_points: Number of data points
            noise_level: Standard deviation of Gaussian noise
            time_ordered: Whether to add time ordering for cross-validation
        """
        if weights is None:
            weights = [1.0] * len(degrees)
            
        # Generate x values
        x = np.linspace(-1, 1, num_points)
        y = np.zeros_like(x)
        
        # Combine Chebyshev polynomials
        for degree, weight in zip(degrees, weights):
            cheb = ChebyshevStep(degree=degree)
            y += weight * cheb.transform_diagonal(x[:, None]).flatten()

        # Add noise
        if noise_level > 0:
            y += np.random.normal(0, noise_level, size=y.shape)
            
        # Create DataFrame with time ordering if needed
        if time_ordered:
            date_id = np.arange(num_points)
            df = pl.DataFrame({
                'feature_0': x,
                'date_id': date_id
            })
        else:
            df = pl.DataFrame({
                'feature_0': x
            })
            
        return df, y
        
    def test_simple_combination(self):
        """Test combination of T_0 + T_1"""
        df, y = self.generate_chebyshev_data(
            degrees=[0, 1],
            weights=[1.0, 1.0],
            noise_level=0.3
        )
        x = df['feature_0'].to_numpy()
        test_degree = 1
        cheb = ChebyshevStep(test_degree)
        our_vals = cheb.transform_diagonal(x[:5, None]).flatten()
        expected_vals = chebval(x[:5], [0]*test_degree + [1])
        print("Degree 1, first 5 points, Expected:", expected_vals, "Our:", our_vals)
        # Test both cross-validation strategies
        for strategy in ['time_based', 'expanding_window']:
            scores = self.optimizer.evaluate_expressiveness(df, y, cv_strategy=strategy)
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
            
            # Should identify degree 1 as sufficient
            self.assertEqual(best_degree, 1, 
                           f"Strategy {strategy} failed to identify correct degree")
            
    def test_higher_degree_combination(self):
        """Test combination of T_2 + T_3"""
        df, y = self.generate_chebyshev_data(
            degrees=[2, 3],
            weights=[1.0, 1.0],
            noise_level=0.05
        )
        
        for strategy in ['time_based', 'expanding_window']:
            scores = self.optimizer.evaluate_expressiveness(df, y, cv_strategy=strategy)
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
            
            # Should identify degree 3 as necessary
            self.assertEqual(best_degree, 3,
                           f"Strategy {strategy} failed to identify correct degree")
            
    def test_weighted_combination(self):
        """Test weighted combination 0.5*T_1 + 2*T_2"""
        df, y = self.generate_chebyshev_data(
            degrees=[1, 2],
            weights=[0.5, 2.0],
            noise_level=0.05
        )
        
        for strategy in ['time_based', 'expanding_window']:
            scores = self.optimizer.evaluate_expressiveness(df, y, cv_strategy=strategy)
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
            
            # Should identify degree 2 as necessary due to stronger T_2 term
            self.assertEqual(best_degree, 2,
                           f"Strategy {strategy} failed to identify correct degree")
            
    def test_multiple_polynomials(self):
        """Test combination of T_1 + T_2 + T_3"""
        df, y = self.generate_chebyshev_data(
            degrees=[1, 2, 3],
            weights=[1.0, 1.0, 1.0],
            noise_level=0.05
        )
        
        for strategy in ['time_based', 'expanding_window']:
            scores = self.optimizer.evaluate_expressiveness(df, y, cv_strategy=strategy)
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
            
            # Should identify degree 3 as necessary
            self.assertEqual(best_degree, 3,
                           f"Strategy {strategy} failed to identify correct degree")
            
    def test_noise_sensitivity(self):
        """Test how well it handles increasing noise levels"""
        degrees = [1, 2]
        weights = [1.0, 1.0]
        noise_levels = [0.01, 0.1, 0.5]
        
        for noise in noise_levels:
            df, y = self.generate_chebyshev_data(
                degrees=degrees,
                weights=weights,
                noise_level=noise
            )
            
            # Test with expanding_window strategy
            scores = self.optimizer.evaluate_expressiveness(df, y, cv_strategy='expanding_window')
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
            
            if noise <= 0.1:  # For reasonable noise levels
                self.assertEqual(best_degree, 2,
                               f"Failed to identify correct degree with noise {noise}")
            
    def test_optimize_layer(self):
        """Test full layer optimization with polynomial combinations"""
        # Generate data with T_1 + T_2 combination
        df, y = self.generate_chebyshev_data(
            degrees=[1, 2],
            weights=[1.0, 1.0],
            noise_level=0.05
        )
        
        # Optimize layer
        optimal_degrees = self.optimizer.optimize_layer(
            layer_idx=0,
            x_data=df,
            y_data=y[:, None],
            num_reads=100
        )
        
        # Should select degree 2 for the connection
        self.assertEqual(optimal_degrees[0][0], 2,
                        "Failed to optimize layer with polynomial combination")

    def test_very_high_degree(self):
        """Test combination with degree 8 polynomial"""
        df, y = self.generate_chebyshev_data(
            degrees=[7],
            weights=[1.0],
            noise_level=0.05
        )
        scores = self.optimizer.evaluate_expressiveness(df, y)
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
        self.assertEqual(best_degree, 7)

    def test_sinusoidal_approximation(self):
        """Test approximating sin(πx) with Chebyshev polynomials"""
        x = np.linspace(-1, 1, 100)
        y = np.sin(np.pi * x)
        df = pl.DataFrame({
            'feature_0': x,
            'date_id': np.arange(len(x))
        })
        scores = self.optimizer.evaluate_expressiveness(df, y)
        is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)
        # Should pick a reasonably high degree to approximate sin
        self.assertGreaterEqual(best_degree, 3)


    def test_comprehensive_noise(self):
        """Test various noise levels and types"""
        degrees = [1, 2]
        weights = [1.0, 1.0]
        noise_levels = [0.01, 0.1, 0.5]

        for noise in noise_levels:
            df, y = self.generate_chebyshev_data(
                degrees=degrees,
                weights=weights,
                noise_level=noise
            )
            scores = self.optimizer.evaluate_expressiveness(df, y)
            is_definitive, best_degree = self.optimizer.is_degree_definitive(scores)

            if noise <= 0.1:
                # Should still identify correct degree for reasonable noise
                self.assertEqual(best_degree, 2)

if __name__ == '__main__':
    unittest.main()