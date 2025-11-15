"""
Chapter 8: Banach Spaces - Optical Norms and Function Spaces
Functional Analysis for Optical Design
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class OpticalBanachSpaces:
    """
    Banach space concepts in optical design:
    - Lᵖ spaces for different optical metrics
    - Sobolev spaces for smoothness constraints
    - Norm equivalence and completeness
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def lp_norm(self, function, p=2):
        """
        Compute Lᵖ norm of a function:
        ‖f‖ₚ = (∫|f(x)|ᵖdx)^(1/p)
        """
        if p == np.inf:
            return np.max(np.abs(function))
        else:
            return (np.sum(np.abs(function)**p) * self.dx)**(1/p)
    
    def sobolev_norm(self, function, s=1, p=2):
        """
        Compute Sobolev norm Wˢᵖ:
        ‖f‖ₛ,ₚ = ‖f‖ₚ + ‖f'‖ₚ + ... + ‖f⁽ˢ⁾‖ₚ
        """
        derivatives = [function.copy()]
        
        # Compute derivatives up to order s
        for order in range(1, s+1):
            derivative = np.gradient(derivatives[-1], self.dx)
            derivatives.append(derivative)
        
        # Sum of Lᵖ norms of all derivatives
        total_norm = 0
        for derivative in derivatives:
            total_norm += self.lp_norm(derivative, p)**p
        
        return total_norm**(1/p)
    
    def optical_quality_metrics(self, wavefront_error):
        """
        Different optical quality metrics as Banach space norms:
        - RMS: L² norm
        - Peak-to-valley: L∞ norm  
        - Gradient: Sobolev norm
        - Total variation: L¹ norm of derivative
        """
        metrics = {
            'rms_error': self.lp_norm(wavefront_error, 2),
            'peak_valley': self.lp_norm(wavefront_error, np.inf),
            'gradient_energy': self.sobolev_norm(wavefront_error, s=1, p=2),
            'total_variation': self.lp_norm(np.gradient(wavefront_error), 1),
            'smoothness': self.sobolev_norm(wavefront_error, s=2, p=2)
        }
        return metrics
    
    def norm_equivalence_analysis(self, test_functions):
        """
        Analyze equivalence between different norms.
        In finite dimensions, all norms are equivalent.
        """
        results = {}
        
        for func_name, func in test_functions.items():
            l1_norm = self.lp_norm(func, 1)
            l2_norm = self.lp_norm(func, 2)
            linf_norm = self.lp_norm(func, np.inf)
            sobolev_norm = self.sobolev_norm(func, s=1, p=2)
            
            results[func_name] = {
                'L1': l1_norm,
                'L2': l2_norm,
                'L_inf': linf_norm,
                'H1': sobolev_norm,
                'L2/L1_ratio': l2_norm / l1_norm,
                'L_inf/L2_ratio': linf_norm / l2_norm,
                'H1/L2_ratio': sobolev_norm / l2_norm
            }
        
        return results
    
    def completeness_test(self, function_sequence):
        """
        Test completeness (Cauchy sequence convergence).
        A Banach space is complete: every Cauchy sequence converges.
        """
        differences = []
        
        for i in range(1, len(function_sequence)):
            diff = function_sequence[i] - function_sequence[i-1]
            diff_norm = self.lp_norm(diff, 2)
            differences.append(diff_norm)
        
        # Check if sequence is Cauchy
        is_cauchy = all(differences[i] <= differences[0] for i in range(len(differences)))
        
        return {
            'is_cauchy': is_cauchy,
            'differences': differences,
            'final_difference': differences[-1] if differences else 0
        }
    
    def dual_space_analysis(self, original_function, test_linear_functionals):
        """
        Analyze dual space (space of linear functionals).
        For Lᵖ spaces, the dual is Lᵠ where 1/p + 1/q = 1.
        """
        results = {}
        
        for func_name, functional in test_linear_functionals.items():
            # Apply linear functional: ⟨f, g⟩ = ∫f(x)g(x)dx
            dual_value = np.sum(original_function * functional) * self.dx
            
            # Compute dual norm
            functional_norm = self.lp_norm(functional, 2)  # Assuming L² for simplicity
            
            results[func_name] = {
                'dual_value': dual_value,
                'functional_norm': functional_norm,
                'normalized_value': dual_value / functional_norm
            }
        
        return results
    
    def optimization_in_banach_space(self, target_function, initial_guess, norm_order=2):
        """
        Optimization in Banach spaces using different norms.
        """
        def objective(candidate):
            # Distance to target in specified norm
            difference = candidate - target_function
            return self.lp_norm(difference, norm_order)
        
        # Simple gradient descent
        current = initial_guess.copy()
        learning_rate = 0.01
        history = []
        
        for iteration in range(100):
            # Compute gradient
            diff = current - target_function
            if norm_order == 2:
                gradient = 2 * diff * self.dx
            elif norm_order == 1:
                gradient = np.sign(diff) * self.dx
            else:
                gradient = norm_order * np.abs(diff)**(norm_order-1) * np.sign(diff) * self.dx
            
            # Update
            current = current - learning_rate * gradient
            
            current_objective = objective(current)
            history.append(current_objective)
            
            if current_objective < 1e-6:
                break
        
        return current, history


def demonstrate_banach_spaces():
    """Demonstrate Banach space concepts in optical design."""
    print("Banach Spaces in Optical Design")
    print("=" * 35)
    
    # Initialize Banach space analyzer
    banach = OpticalBanachSpaces(grid_size=100)
    
    # Test functions representing different optical aberrations
    test_functions = {
        'defocus': 0.5 * (2 * banach.x**2 - 1),
        'astigmatism': 0.3 * banach.x**2,
        'coma': 0.2 * (3 * banach.x**3 - 2 * banach.x),
        'spherical': 0.1 * (6 * banach.x**4 - 6 * banach.x**2 + 1),
        'random_aberration': 0.1 * np.random.randn(100)
    }
    
    print("Optical quality metrics for different aberrations:")
    for func_name, func in test_functions.items():
        metrics = banach.optical_quality_metrics(func)
        print(f"\n{func_name.title()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
    
    # Norm equivalence analysis
    print(f"\n--- Norm Equivalence Analysis ---")
    equivalence = banach.norm_equivalence_analysis(test_functions)
    
    for func_name, norms in equivalence.items():
        print(f"\n{func_name}:")
        print(f"  L¹: {norms['L1']:.4f}, L²: {norms['L2']:.4f}, L∞: {norms['L_inf']:.4f}")
        print(f"  H¹: {norms['H1']:.4f}")
        print(f"  Ratios - L²/L¹: {norms['L2/L1_ratio']:.4f}, L∞/L²: {norms['L_inf/L2_ratio']:.4f}")
    
    # Completeness test
    print(f"\n--- Completeness Test ---")
    # Create convergent sequence
    convergent_sequence = []
    for n in range(10):
        func = np.exp(-banach.x**2) + np.sin(n * banach.x) / (n + 1)
        convergent_sequence.append(func)
    
    completeness = banach.completeness_test(convergent_sequence)
    print(f"Sequence is Cauchy: {completeness['is_cauchy']}")
    print(f"Final difference: {completeness['final_difference']:.6f}")
    
    # Optimization in different norms
    print(f"\n--- Optimization in Different Norms ---")
    target = np.exp(-banach.x**2)
    initial = np.zeros_like(target)
    
    for p in [1, 2, np.inf]:
        result, history = banach.optimization_in_banach_space(target, initial, p)
        print(f"L{p} norm optimization: final error = {history[-1]:.6f}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Lᵖ spaces provide different metrics for optical quality")
    print("2. Sobolev spaces include derivative information for smoothness")
    print("3. All norms are equivalent in finite dimensions")
    print("4. Banach spaces are complete (all Cauchy sequences converge)")
    print("5. Choice of norm affects optimization behavior")


if __name__ == "__main__":
    demonstrate_banach_spaces()