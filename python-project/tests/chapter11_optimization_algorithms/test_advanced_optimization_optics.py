"""
Test suite for Chapter 11: Advanced Optimization Algorithms in Optical Design
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from chapter11_optimization_algorithms.advanced_optimization_optics import AdvancedOptimizationOptics


class TestAdvancedOptimizationOptics:
    """Test suite for Advanced Optimization Algorithms in Optical Design."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a default AdvancedOptimizationOptics instance."""
        return AdvancedOptimizationOptics(grid_size=50)
    
    @pytest.fixture
    def test_functions(self):
        """Create test functions for optimization."""
        def quadratic_objective(x):
            """Simple quadratic function: f(x) = xᵀx."""
            return np.sum(x**2)
        
        def quadratic_gradient(x):
            """Gradient of quadratic function: ∇f(x) = 2x."""
            return 2 * x
        
        def quadratic_hessian(x):
            """Hessian of quadratic function: H(x) = 2I."""
            return 2 * np.eye(len(x))
        
        def rosenbrock_objective(x):
            """Rosenbrock function."""
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        def rosenbrock_gradient(x):
            """Gradient of Rosenbrock function."""
            grad = np.zeros_like(x)
            grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            grad[1:] += 200 * (x[1:] - x[:-1]**2)
            return grad
        
        def rosenbrock_hessian(x):
            """Hessian of Rosenbrock function."""
            n = len(x)
            hessian = np.zeros((n, n))
            
            for i in range(n-1):
                hessian[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
                hessian[i, i+1] = -400 * x[i]
                hessian[i+1, i] = -400 * x[i]
                hessian[i+1, i+1] = 200
            
            return hessian
        
        return {
            'quadratic': {
                'objective': quadratic_objective,
                'gradient': quadratic_gradient,
                'hessian': quadratic_hessian
            },
            'rosenbrock': {
                'objective': rosenbrock_objective,
                'gradient': rosenbrock_gradient,
                'hessian': rosenbrock_hessian
            }
        }
    
    def test_initialization(self):
        """Test proper initialization of AdvancedOptimizationOptics."""
        optimizer = AdvancedOptimizationOptics(grid_size=100)
        assert optimizer.grid_size == 100
        assert len(optimizer.x) == 100
        assert optimizer.dx == pytest.approx(2/99, rel=1e-10)
        assert np.isclose(optimizer.x[0], -1.0)
        assert np.isclose(optimizer.x[-1], 1.0)
    
    def test_newton_method_functional_quadratic(self, optimizer, test_functions):
        """Test Newton method with quadratic function."""
        quadratic = test_functions['quadratic']
        initial_guess = np.array([1.0, 2.0, -1.0, 0.5])
        
        history = optimizer.newton_method_functional(
            initial_guess, quadratic['objective'], quadratic['gradient'], 
            quadratic['hessian'], num_iterations=20, line_search=True
        )
        
        # Check structure of history
        assert len(history) > 0
        assert len(history) <= 20
        
        for record in history:
            assert 'iteration' in record
            assert 'merit' in record
            assert 'gradient_norm' in record
            assert 'step_size' in record
            assert 'alpha' in record
            
            assert record['merit'] >= 0
            assert record['gradient_norm'] >= 0
            assert record['step_size'] >= 0
            assert 0 <= record['alpha'] <= 1
        
        # Should converge to minimum (zero)
        final_merit = history[-1]['merit']
        assert final_merit < 1e-6
        
        # Gradient should be small at convergence
        final_gradient_norm = history[-1]['gradient_norm']
        assert final_gradient_norm < 1e-6
    
    def test_newton_method_functional_without_hessian(self, optimizer, test_functions):
        """Test Newton method without explicit Hessian."""
        quadratic = test_functions['quadratic']
        initial_guess = np.array([1.0, -1.0, 0.5])
        
        history = optimizer.newton_method_functional(
            initial_guess, quadratic['objective'], quadratic['gradient'], 
            hessian_function=None, num_iterations=15, line_search=True
        )
        
        # Should still converge (though possibly slower)
        assert len(history) > 0
        assert len(history) <= 15
        
        final_merit = history[-1]['merit']
        assert final_merit < 0.1  # Should make progress
    
    def test_newton_method_functional_rosenbrock(self, optimizer, test_functions):
        """Test Newton method with Rosenbrock function."""
        rosenbrock = test_functions['rosenbrock']
        initial_guess = np.random.randn(5) * 0.1  # Small random initial guess
        
        history = optimizer.newton_method_functional(
            initial_guess, rosenbrock['objective'], rosenbrock['gradient'], 
            rosenbrock['hessian'], num_iterations=30, line_search=True
        )
        
        # Check convergence
        assert len(history) > 0
        assert len(history) <= 30
        
        # Should make progress toward minimum
        final_merit = history[-1]['merit']
        initial_merit = history[0]['merit']
        assert final_merit < initial_merit
        
        # Should converge to reasonable value
        assert final_merit < 1e-2  # Rosenbrock minimum is at 0
    
    def test_trust_region_method_quadratic(self, optimizer, test_functions):
        """Test trust region method with quadratic function."""
        quadratic = test_functions['quadratic']
        initial_guess = np.array([2.0, -1.0, 0.5, -0.3])
        
        history = optimizer.trust_region_method(
            initial_guess, quadratic['objective'], quadratic['gradient'],
            hessian_function=quadratic['hessian'], trust_radius=0.5, max_iterations=25
        )
        
        # Check structure of history
        assert len(history) > 0
        assert len(history) <= 25
        
        for record in history:
            assert 'iteration' in record
            assert 'merit' in record
            assert 'trust_radius' in record
            assert 'step_size' in record
            assert 'step_accepted' in record
            assert 'rho' in record
            
            assert record['merit'] >= 0
            assert record['trust_radius'] > 0
            assert record['step_size'] >= 0
            assert record['step_accepted'] in [True, False]
            assert record['rho'] >= 0
        
        # Should converge to minimum
        final_merit = history[-1]['merit']
        assert final_merit < 1e-6
    
    def test_trust_region_method_rosenbrock(self, optimizer, test_functions):
        """Test trust region method with Rosenbrock function."""
        rosenbrock = test_functions['rosenbrock']
        initial_guess = np.random.randn(4) * 0.2
        
        history = optimizer.trust_region_method(
            initial_guess, rosenbrock['objective'], rosenbrock['gradient'],
            trust_radius=0.3, max_iterations=40
        )
        
        # Should make progress
        assert len(history) > 0
        assert len(history) <= 40
        
        final_merit = history[-1]['merit']
        initial_merit = history[0]['merit']
        assert final_merit < initial_merit
        
        # Trust radius should adapt
        trust_radii = [record['trust_radius'] for record in history]
        assert min(trust_radii) > 0
        assert max(trust_radii) <= 1.0
    
    def test_multi_objective_optimization(self, optimizer):
        """Test multi-objective optimization."""
        # Define multiple objectives
        def objective1(x):
            return np.sum(x**2)  # Minimize squared norm
        
        def objective2(x):
            return np.sum((x - 1)**2)  # Minimize distance from 1
        
        def objective3(x):
            return np.sum(np.abs(x))  # Minimize L1 norm
        
        objectives = [objective1, objective2, objective3]
        weights = [0.4, 0.4, 0.2]
        initial_design = np.random.randn(5) * 0.5
        
        results = optimizer.multi_objective_optimization(
            initial_design, objectives, weights
        )
        
        # Check results structure
        assert len(results) > 0
        
        for method, result in results.items():
            if result['success']:
                assert 'x' in result
                assert 'fun' in result
                assert 'nit' in result
                assert len(result['x']) == len(initial_design)
                assert result['fun'] >= 0
                assert result['nit'] >= 0
    
    def test_pareto_optimization(self, optimizer):
        """Test Pareto optimization."""
        # Define conflicting objectives
        def objective1(x):
            return np.sum(x**2)  # Minimize norm
        
        def objective2(x):
            return np.sum((x - 2)**2)  # Minimize distance from 2
        
        objectives = [objective1, objective2]
        bounds = [(-2, 2)] * 3  # 3D optimization
        
        result = optimizer.pareto_optimization(
            objectives, bounds, population_size=20, generations=30
        )
        
        # Check structure
        assert 'pareto_points' in result
        assert 'num_points' in result
        assert 'representative_solution' in result
        
        assert result['num_points'] >= 0
        assert len(result['pareto_points']) == result['num_points']
        
        # Check Pareto points
        for point in result['pareto_points']:
            assert 'design' in point
            assert 'objectives' in point
            assert 'weights' in point
            assert len(point['objectives']) == len(objectives)
            assert all(obj_val >= 0 for obj_val in point['objectives'])
    
    def test_global_optimization_demo(self, optimizer):
        """Test global optimization demonstration."""
        # Multi-modal objective function
        def multimodal_objective(x):
            return (np.sin(5 * x[0]) * np.cos(5 * x[1]) + 
                   0.1 * np.sum(x**2) + 
                   0.2 * np.sin(10 * x[0]) * np.sin(10 * x[1]))
        
        bounds = [(-2, 2), (-2, 2)]
        
        results = optimizer.global_optimization_demo(
            multimodal_objective, bounds, methods=['differential_evolution']
        )
        
        # Check results
        assert 'differential_evolution' in results
        de_result = results['differential_evolution']
        
        if 'error' not in de_result:
            assert 'success' in de_result
            assert 'x' in de_result
            assert 'fun' in de_result
            assert 'nit' in de_result
            assert len(de_result['x']) == 2
            assert np.isfinite(de_result['fun'])
    
    def test_convergence_behavior(self, optimizer, test_functions):
        """Test convergence behavior of optimization methods."""
        quadratic = test_functions['quadratic']
        initial_guess = np.array([3.0, -2.0, 1.0])
        
        # Test Newton method
        newton_history = optimizer.newton_method_functional(
            initial_guess, quadratic['objective'], quadratic['gradient'], 
            quadratic['hessian'], num_iterations=10
        )
        
        # Merit should decrease monotonically
        merits = [record['merit'] for record in newton_history]
        for i in range(1, len(merits)):
            assert merits[i] <= merits[i-1] * 1.01  # Allow small numerical errors
        
        # Test trust region method
        trust_history = optimizer.trust_region_method(
            initial_guess, quadratic['objective'], quadratic['gradient'],
            hessian_function=quadratic['hessian'], max_iterations=15
        )
        
        # Merit should generally decrease
        trust_merits = [record['merit'] for record in trust_history]
        final_merit = trust_merits[-1]
        initial_merit = trust_merits[0]
        assert final_merit <= initial_merit
    
    def test_edge_cases(self, optimizer, test_functions):
        """Test edge cases and error handling."""
        quadratic = test_functions['quadratic']
        
        # Test with zero initial guess
        zero_guess = np.zeros(5)
        history = optimizer.newton_method_functional(
            zero_guess, quadratic['objective'], quadratic['gradient'], 
            quadratic['hessian'], num_iterations=5
        )
        assert len(history) > 0
        assert history[-1]['merit'] == 0.0  # Should stay at minimum
        
        # Test with very large initial guess
        large_guess = np.ones(3) * 1e6
        history = optimizer.trust_region_method(
            large_guess, quadratic['objective'], quadratic['gradient'],
            trust_radius=1.0, max_iterations=10
        )
        assert len(history) > 0
        assert history[-1]['merit'] < history[0]['merit']  # Should make progress
        
        # Test with single dimension
        single_guess = np.array([5.0])
        history = optimizer.newton_method_functional(
            single_guess, quadratic['objective'], quadratic['gradient'], 
            quadratic['hessian'], num_iterations=10
        )
        assert len(history) > 0
        assert history[-1]['merit'] < 1e-6
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_advanced_optimization(self, mock_figure, mock_show):
        """Test the demonstration function."""
        # Mock matplotlib to avoid showing plots during tests
        mock_figure.return_value = MagicMock()
        mock_show.return_value = None
        
        # This should run without errors
        from chapter11_optimization_algorithms.advanced_optimization_optics import demonstrate_advanced_optimization
        demonstrate_advanced_optimization()
        
        # Verify matplotlib was called
        mock_figure.assert_called()
        mock_show.assert_called()


class TestAdvancedOptimizationMathematicalProperties:
    """Test mathematical properties of advanced optimization algorithms."""
    
    @pytest.fixture
    def optimizer(self):
        """Create AdvancedOptimizationOptics instance for mathematical tests."""
        return AdvancedOptimizationOptics(grid_size=50)
    
    def test_newton_method_quadratic_convergence(self, optimizer):
        """Test that Newton method converges in one iteration for quadratic functions."""
        # Quadratic function
        def objective(x):
            return 0.5 * np.sum(x**2)
        
        def gradient(x):
            return x
        
        def hessian(x):
            return np.eye(len(x))
        
        initial = np.array([1.0, 2.0, -1.0])
        
        history = optimizer.newton_method_functional(
            initial, objective, gradient, hessian, num_iterations=5
        )
        
        # Should converge very quickly (though may need a few iterations due to line search)
        final_merit = history[-1]['merit']
        assert final_merit < 1e-6
    
    def test_trust_region_radius_adaptation(self, optimizer):
        """Test that trust region radius adapts during optimization."""
        def objective(x):
            return np.sum(x**4)  # Non-quadratic function
        
        def gradient(x):
            return 4 * x**3
        
        initial = np.array([1.0, 0.5, -0.8])
        
        history = optimizer.trust_region_method(
            initial, objective, gradient, trust_radius=0.1, max_iterations=20
        )
        
        # Trust radius should change during optimization
        trust_radii = [record['trust_radius'] for record in history]
        assert len(set(trust_radii)) > 1  # Should have different values
        
        # Should stay within reasonable bounds
        assert min(trust_radii) >= 1e-4
        assert max(trust_radii) <= 1.0
    
    def test_multi_objective_weight_sensitivity(self, optimizer):
        """Test sensitivity of multi-objective optimization to weights."""
        def obj1(x):
            return np.sum(x**2)
        
        def obj2(x):
            return np.sum((x - 1)**2)
        
        initial = np.array([0.5, 0.5, 0.5])
        
        # Test different weight combinations
        weight_sets = [
            [1.0, 0.0],  # Only objective 1
            [0.0, 1.0],  # Only objective 2
            [0.5, 0.5],  # Equal weights
            [0.8, 0.2],  # Favor objective 1
            [0.2, 0.8]   # Favor objective 2
        ]
        
        results = []
        for weights in weight_sets:
            result = optimizer.multi_objective_optimization(
                initial, [obj1, obj2], weights
            )
            if 'BFGS' in result and result['BFGS']['success']:
                optimal_x = result['BFGS']['x']
                obj1_val = obj1(optimal_x)
                obj2_val = obj2(optimal_x)
                results.append({
                    'weights': weights,
                    'optimal_x': optimal_x,
                    'obj1_val': obj1_val,
                    'obj2_val': obj2_val
                })
        
        # Should have different optimal points for different weights
        assert len(results) >= 3
        
        # Pure objective 1 should give solution close to origin
        pure_obj1 = next(r for r in results if r['weights'] == [1.0, 0.0])
        assert np.allclose(pure_obj1['optimal_x'], 0.0, atol=0.1)
        
        # Pure objective 2 should give solution close to ones
        pure_obj2 = next(r for r in results if r['weights'] == [0.0, 1.0])
        assert np.allclose(pure_obj2['optimal_x'], 1.0, atol=0.1)
    
    def test_pareto_front_properties(self, optimizer):
        """Test properties of Pareto front."""
        def obj1(x):
            return np.sum(x**2)  # Distance from origin
        
        def obj2(x):
            return np.sum((x - 2)**2)  # Distance from (2,2,2)
        
        bounds = [(-1, 3)] * 3
        
        result = optimizer.pareto_optimization(
            [obj1, obj2], bounds, population_size=30, generations=50
        )
        
        pareto_points = result['pareto_points']
        
        # Should have multiple Pareto points
        assert len(pareto_points) >= 5
        
        # Check Pareto dominance: no point should dominate another
        for i, point1 in enumerate(pareto_points):
            for j, point2 in enumerate(pareto_points):
                if i != j:
                    obj1_1, obj1_2 = point1['objectives'][0], point1['objectives'][1]
                    obj2_1, obj2_2 = point2['objectives'][0], point2['objectives'][1]
                    
                    # Point1 dominates point2 if it's better in all objectives
                    dominates = (obj1_1 <= obj2_1 and obj1_2 <= obj2_2 and 
                               (obj1_1 < obj2_1 or obj1_2 < obj2_2))
                    
                    # In a proper Pareto front, no point should dominate another
                    # (This is a statistical test due to sampling)
                    assert not dominates or abs(obj1_1 - obj2_1) < 1e-6
    
    def test_global_optimization_escape_local_minima(self, optimizer):
        """Test that global optimization can escape local minima."""
        # Create function with multiple local minima
        def multimodal_objective(x):
            return (np.sin(10 * x[0])**2 + np.sin(10 * x[1])**2 + 
                   0.1 * (x[0]**2 + x[1]**2))
        
        bounds = [(-2, 2), (-2, 2)]
        
        # Start from local minimum
        local_minimum = np.array([np.pi/10, np.pi/10])
        
        # Local optimization would get stuck
        local_result = optimizer.newton_method_functional(
            local_minimum, multimodal_objective, 
            lambda x: np.array([20 * np.sin(10 * x[0]) * np.cos(10 * x[0]) + 0.2 * x[0],
                               20 * np.sin(10 * x[1]) * np.cos(10 * x[1]) + 0.2 * x[1]]),
            num_iterations=10
        )
        
        # Global optimization should find better solution
        global_results = optimizer.global_optimization_demo(
            multimodal_objective, bounds, methods=['differential_evolution']
        )
        
        if 'differential_evolution' in global_results:
            de_result = global_results['differential_evolution']
            if 'error' not in de_result and de_result['success']:
                global_value = de_result['fun']
                local_value = local_result[-1]['merit']
                
                # Global optimization should find better or equal value
                assert global_value <= local_value * 1.01  # Allow small tolerance


if __name__ == "__main__":
    pytest.main([__file__])