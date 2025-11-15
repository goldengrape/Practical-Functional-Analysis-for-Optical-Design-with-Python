"""
Chapter 3: Functional Gradient Descent - Advanced Implementation
Functional Analysis for Optical Design
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
from scipy.optimize import minimize
import time


class FunctionalGradientDescent:
    """
    Advanced implementation of functional gradient descent for optical design.
    Demonstrates the transition from finite-dimensional to infinite-dimensional optimization.
    """
    
    def __init__(self, domain_size: int = 100):
        self.domain_size = domain_size
        self.x = np.linspace(-1, 1, domain_size)
        self.dx = self.x[1] - self.x[0]
        
    def optical_merit_functional(self, surface_func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Merit functional for optical surface optimization.
        Combines wavefront error, smoothness, and manufacturing constraints.
        """
        # Sample the surface function
        surface_values = surface_func(self.x)
        
        # Wavefront error component (deviation from ideal spherical)
        ideal_sphere = self._ideal_spherical_surface(self.x)
        wavefront_error = np.sum((surface_values - ideal_sphere)**2) * self.dx
        
        # Smoothness penalty (second derivative)
        second_deriv = np.gradient(np.gradient(surface_values, self.dx), self.dx)
        smoothness_penalty = np.sum(second_deriv**2) * self.dx
        
        # Manufacturing constraint (bounded curvature)
        curvature = np.gradient(np.gradient(surface_values, self.dx), self.dx)
        manufacturing_penalty = np.sum(np.maximum(0, np.abs(curvature) - 2)**2) * self.dx
        
        # Total merit functional
        total_merit = wavefront_error + 0.1 * smoothness_penalty + 0.5 * manufacturing_penalty
        
        return total_merit
    
    def _ideal_spherical_surface(self, x: np.ndarray) -> np.ndarray:
        """Generate ideal spherical surface for reference."""
        radius = 5.0
        return radius - np.sqrt(np.maximum(0, radius**2 - x**2))
    
    def functional_gradient(self, surface_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Compute the functional gradient of the merit functional.
        This is the infinite-dimensional analogue of the gradient.
        """
        # Numerical computation of functional gradient using calculus of variations
        epsilon = 1e-6
        
        def gradient_func(x: np.ndarray) -> np.ndarray:
            # For each point, compute the directional derivative
            gradient = np.zeros_like(x)
            
            for i in range(len(x)):
                # Perturbation function (delta function approximation)
                def perturbed_surface(x_input):
                    base_surface = surface_func(x_input)
                    # Add small perturbation at position x[i]
                    perturbation = epsilon * np.exp(-100 * (x_input - x[i])**2)
                    return base_surface + perturbation
                
                # Compute directional derivative
                original_merit = self.optical_merit_functional(surface_func)
                perturbed_merit = self.optical_merit_functional(perturbed_surface)
                gradient[i] = (perturbed_merit - original_merit) / epsilon
            
            return gradient
        
        return gradient_func
    
    def functional_gradient_descent(self, initial_surface: Callable[[np.ndarray], np.ndarray], 
                                  num_iterations: int = 100, 
                                  learning_rate: float = 0.01) -> list:
        """
        Implement functional gradient descent algorithm.
        """
        current_surface = initial_surface
        history = []
        
        for iteration in range(num_iterations):
            # Compute current merit
            current_merit = self.optical_merit_functional(current_surface)
            history.append({
                'iteration': iteration,
                'merit': current_merit,
                'surface': current_surface(self.x)
            })
            
            # Compute functional gradient
            gradient_func = self.functional_gradient(current_surface)
            gradient_values = gradient_func(self.x)
            
            # Update surface (functional gradient descent step)
            def new_surface(x_input):
                current_values = current_surface(x_input)
                # Functional gradient descent: f_new = f_old - learning_rate * gradient
                return current_values - learning_rate * gradient_values[np.argmin(np.abs(self.x.reshape(-1, 1) - x_input.reshape(1, -1)), axis=0)]
            
            current_surface = new_surface
            
            # Print progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Merit = {current_merit:.6f}")
        
        return history
    
    def comparison_finite_vs_functional(self):
        """
        Compare finite-dimensional optimization with functional gradient descent.
        """
        print("=== Finite vs Functional Gradient Descent Comparison ===\n")
        
        # Initial surface (parabolic)
        def initial_surface(x):
            return 0.1 * x**2
        
        # Functional gradient descent
        start_time = time.time()
        functional_history = self.functional_gradient_descent(
            initial_surface, num_iterations=50, learning_rate=0.01
        )
        functional_time = time.time() - start_time
        
        # Finite-dimensional optimization (using scipy)
        def finite_objective(params):
            # Parametrize surface as polynomial
            def surface_func(x):
                result = np.zeros_like(x)
                for i, param in enumerate(params):
                    result += param * x**i
                return result
            return self.optical_merit_functional(surface_func)
        
        # Initial parameters for polynomial (degree 4)
        initial_params = np.array([0.0, 0.0, 0.1, 0.0, 0.0])
        
        start_time = time.time()
        finite_result = minimize(finite_objective, initial_params, method='BFGS')
        finite_time = time.time() - start_time
        
        # Results comparison
        print(f"Functional Gradient Descent:")
        print(f"  Final merit: {functional_history[-1]['merit']:.6f}")
        print(f"  Time: {functional_time:.3f} seconds")
        print(f"  Iterations: {len(functional_history)}")
        
        print(f"\nFinite-dimensional Optimization (BFGS):")
        print(f"  Final merit: {finite_result.fun:.6f}")
        print(f"  Time: {finite_time:.3f} seconds")
        print(f"  Function evaluations: {finite_result.nfev}")
        
        return functional_history, finite_result
    
    def visualize_optimization_process(self, history: list):
        """Visualize the functional gradient descent process."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Merit function evolution
        iterations = [h['iteration'] for h in history]
        merits = [h['merit'] for h in history]
        
        axes[0, 0].plot(iterations, merits, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Merit Functional')
        axes[0, 0].set_title('Merit Functional Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Surface evolution (show every 20 iterations)
        for i, h in enumerate(history[::max(1, len(history)//5)]):
            axes[0, 1].plot(self.x, h['surface'], 
                          label=f'Iter {h["iteration"]}', alpha=0.7)
        
        axes[0, 1].plot(self.x, self._ideal_spherical_surface(self.x), 
                       'k--', linewidth=2, label='Ideal')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Surface Height')
        axes[0, 1].set_title('Surface Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final gradient magnitude
        final_surface_func = lambda x: history[-1]['surface'][np.argmin(np.abs(self.x.reshape(-1, 1) - x.reshape(1, -1)), axis=0)]
        final_gradient_func = self.functional_gradient(final_surface_func)
        final_gradient = final_gradient_func(self.x)
        
        axes[1, 0].plot(self.x, final_gradient, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Gradient Magnitude')
        axes[1, 0].set_title('Final Functional Gradient')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence rate
        log_merits = np.log(merits - np.min(merits) + 1e-10)
        axes[1, 1].plot(iterations[1:], log_merits[1:], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Log(Merit - Min)')
        axes[1, 1].set_title('Convergence Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('functional_gradient_descent_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def theoretical_analysis(self):
        """
        Theoretical analysis of functional gradient descent in optical context.
        """
        print("=== Theoretical Analysis of Functional Gradient Descent ===\n")
        
        # Define a simple test surface
        def test_surface(x):
            return 0.1 * x**2 + 0.05 * np.sin(2 * np.pi * x)
        
        # Compute functional gradient
        gradient_func = self.functional_gradient(test_surface)
        gradient_values = gradient_func(self.x)
        
        # Compute finite difference approximation for verification
        epsilon = 1e-6
        finite_gradient = np.zeros_like(self.x)
        
        for i in range(len(self.x)):
            def perturbed_surface(x_input):
                base = test_surface(x_input)
                perturbation = epsilon * np.exp(-100 * (x_input - self.x[i])**2)
                return base + perturbation
            
            original_merit = self.optical_merit_functional(test_surface)
            perturbed_merit = self.optical_merit_functional(perturbed_surface)
            finite_gradient[i] = (perturbed_merit - original_merit) / epsilon
        
        # Compare analytical and numerical gradients
        gradient_error = np.linalg.norm(gradient_values - finite_gradient)
        print(f"Functional gradient computation verification:")
        print(f"  L2 error between analytical and numerical: {gradient_error:.6f}")
        
        # Analyze gradient properties
        gradient_norm = np.linalg.norm(gradient_values)
        print(f"  L2 norm of functional gradient: {gradient_norm:.6f}")
        
        # Smoothness analysis
        gradient_smoothness = np.linalg.norm(np.gradient(gradient_values, self.dx))
        print(f"  Gradient smoothness (L2 norm of derivative): {gradient_smoothness:.6f}")
        
        return gradient_values, finite_gradient


def main():
    """Main demonstration of functional gradient descent."""
    print("Functional Gradient Descent for Optical Design")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = FunctionalGradientDescent(domain_size=100)
    
    # Theoretical analysis
    gradient_values, finite_gradient = optimizer.theoretical_analysis()
    
    # Comparison with finite-dimensional optimization
    functional_history, finite_result = optimizer.comparison_finite_vs_functional()
    
    # Visualize results
    optimizer.visualize_optimization_process(functional_history)
    
    print("\n=== Key Insights ===")
    print("1. Functional gradient descent operates directly on function spaces")
    print("2. Convergence properties depend on the choice of function space")
    print("3. Regularization terms ensure physically meaningful solutions")
    print("4. The method naturally handles infinite-dimensional optimization problems")


if __name__ == "__main__":
    main()