"""
Chapter 9: Weak Convergence in Optical Systems
Functional Analysis for Optical Design

Weak convergence concepts and applications in optical optimization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class WeakConvergenceOptics:
    """
    Weak convergence concepts in optical systems:
    - Weak vs strong convergence in function spaces
    - Applications in iterative optimization algorithms
    - Convergence analysis in optical design
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def weak_convergence_test(self, function_sequence, test_functionals):
        """
        Test weak convergence: lim ∫fₙ(x)g(x)dx = ∫f(x)g(x)dx
        for all test functionals g in the dual space.
        
        In optical context, this represents convergence of measurements
        or observables rather than pointwise convergence.
        """
        convergence_results = {}
        
        for func_name, functional in test_functionals.items():
            integrals = []
            
            for fn in function_sequence:
                # Compute inner product: ⟨fₙ, g⟩ = ∫fₙ(x)g(x)dx
                integral = np.sum(fn * functional) * self.dx
                integrals.append(integral)
            
            # Check convergence behavior
            final_value = integrals[-1]
            convergence_rate = np.abs(np.array(integrals) - final_value)
            
            # Determine if convergent
            is_convergent = np.all(convergence_rate[-10:] < 1e-6)
            
            convergence_results[func_name] = {
                'integrals': integrals,
                'converged': is_convergent,
                'final_value': final_value,
                'convergence_rate': convergence_rate
            }
        
        return convergence_results
    
    def gradient_descent_weak_convergence(self, initial_function, merit_functional, num_iterations=100):
        """
        Analyze weak convergence in gradient descent optimization.
        
        In function spaces, we care about convergence of the functional
        values rather than strong convergence of the functions themselves.
        """
        current = initial_function.copy()
        history = []
        
        for iteration in range(num_iterations):
            # Compute gradient (represents weak derivative)
            gradient = np.gradient(current, self.dx)
            
            # Merit functional value
            merit = merit_functional(current)
            
            # Weak convergence measures
            strong_norm = np.linalg.norm(gradient)  # Strong norm
            weak_norm = np.max(np.abs(gradient))    # Dual norm (simplified)
            
            # Functional convergence
            if iteration > 0:
                functional_change = abs(history[-1]['merit'] - merit) if history else 1.0
            else:
                functional_change = 1.0
            
            history.append({
                'iteration': iteration,
                'merit': merit,
                'strong_norm': strong_norm,
                'weak_norm': weak_norm,
                'functional_change': functional_change,
                'convergence_rate': strong_norm / (iteration + 1)
            })
            
            # Update (gradient descent step)
            current = current - 0.01 * gradient
            
            # Check for convergence
            if functional_change < 1e-8 and iteration > 10:
                print(f"Weak convergence achieved after {iteration+1} iterations")
                break
        
        return history
    
    def optical_measurement_convergence(self, wavefront_sequence, measurement_apertures):
        """
        Study convergence of optical measurements (weak convergence).
        
        Different aperture functions represent different measurement systems.
        Convergence of measurements doesn't imply pointwise convergence.
        """
        measurement_results = {}
        
        for aperture_name, aperture_func in measurement_apertures.items():
            measurements = []
            
            for wavefront in wavefront_sequence:
                # Apply measurement aperture (weak measurement)
                measurement = np.sum(wavefront * aperture_func) * self.dx
                measurements.append(measurement)
            
            # Analyze convergence
            final_measurement = measurements[-1]
            measurement_variance = np.var(measurements[-10:]) if len(measurements) >= 10 else np.inf
            
            measurement_results[aperture_name] = {
                'measurements': measurements,
                'final_value': final_measurement,
                'variance': measurement_variance,
                'converged': measurement_variance < 1e-6,
                'measurement_error': np.abs(np.array(measurements) - final_measurement)
            }
        
        return measurement_results
    
    def weak_compactness_demo(self, function_sequence):
        """
        Demonstrate weak compactness in function spaces.
        
        In infinite-dimensional spaces, bounded sequences have weakly
        convergent subsequences (Banach-Alaoglu theorem).
        """
        # Check boundedness
        norms = [np.linalg.norm(fn) for fn in function_sequence]
        is_bounded = max(norms) < np.inf
        
        # Extract weakly convergent subsequence (simplified)
        subsequence_size = min(20, len(function_sequence))
        subsequences = []
        
        for start in range(0, len(function_sequence) - subsequence_size + 1, 5):
            subseq = function_sequence[start:start + subsequence_size]
            
            # Test weak convergence with constant functional
            constant_functional = np.ones_like(self.x)
            integrals = [np.sum(fn * constant_functional) * self.dx for fn in subseq]
            
            # Check if this subsequence appears to converge
            final_integral = integrals[-1]
            variance = np.var(integrals[-5:]) if len(integrals) >= 5 else np.inf
            
            subsequences.append({
                'start_index': start,
                'integrals': integrals,
                'appears_convergent': variance < 1e-4,
                'final_integral': final_integral,
                'variance': variance
            })
        
        return {
            'is_bounded': is_bounded,
            'max_norm': max(norms) if norms else 0,
            'subsequences': subsequences,
            'has_convergent_subsequence': any(sub['appears_convergent'] for sub in subsequences)
        }
    
    def weak_vs_strong_convergence_demo(self):
        """
        Demonstrate the difference between weak and strong convergence.
        
        Create a sequence that converges weakly but not strongly.
        """
        # Create oscillating sequence: fₙ(x) = sin(nx)/√n
        sequence = []
        for n in range(1, 21):
            fn = np.sin(n * self.x) / np.sqrt(n)
            sequence.append(fn)
        
        # Test functionals
        test_functionals = {
            'constant': np.ones_like(self.x),
            'linear': self.x,
            'quadratic': self.x**2,
            'sinusoidal': np.sin(self.x)
        }
        
        # Test weak convergence
        weak_results = self.weak_convergence_test(sequence, test_functionals)
        
        # Check strong convergence (norm convergence)
        strong_norms = [np.linalg.norm(fn) for fn in sequence]
        strong_convergent = np.all(np.abs(np.diff(strong_norms[-5:])) < 1e-6)
        
        # This sequence should converge weakly to 0 but not strongly
        expected_weak_limit = np.zeros_like(self.x)
        
        return {
            'sequence_type': 'oscillating',
            'weak_convergence': weak_results,
            'strong_norms': strong_norms,
            'strongly_convergent': strong_convergent,
            'converges_weakly_to_zero': all(abs(result['final_value']) < 1e-4 
                                           for result in weak_results.values()),
            'explanation': 'Oscillating sequence converges weakly to 0 but not strongly'
        }


def demonstrate_weak_convergence():
    """Demonstrate weak convergence concepts in optical systems."""
    print("Weak Convergence in Optical Systems")
    print("=" * 40)
    
    # Initialize weak convergence analyzer
    weak_conv = WeakConvergenceOptics(grid_size=100)
    
    # Test 1: Weak vs Strong Convergence
    print(f"\n--- Test 1: Weak vs Strong Convergence ---")
    
    demo_result = weak_conv.weak_vs_strong_convergence_demo()
    print(f"Oscillating sequence converges weakly to zero: {demo_result['converges_weakly_to_zero']}")
    print(f"Strongly convergent: {demo_result['strongly_convergent']}")
    
    # Test 2: Gradient Descent Weak Convergence
    print(f"\n--- Test 2: Gradient Descent Weak Convergence ---")
    
    # Initial function (parabolic wavefront error)
    initial_function = 0.5 * weak_conv.x**2
    
    def merit_functional(func):
        """Merit functional: RMS wavefront error."""
        return np.sqrt(np.mean(func**2))
    
    convergence_history = weak_conv.gradient_descent_weak_convergence(
        initial_function, merit_functional, num_iterations=50
    )
    
    final_merit = convergence_history[-1]['merit']
    final_functional_change = convergence_history[-1]['functional_change']
    print(f"Final merit value: {final_merit:.6f}")
    print(f"Final functional change: {final_functional_change:.2e}")
    
    # Test 3: Optical Measurement Convergence
    print(f"\n--- Test 3: Optical Measurement Convergence ---")
    
    # Create sequence of wavefronts with decreasing high-frequency components
    wavefront_sequence = []
    for n in range(1, 11):
        # Low frequency component (convergent)
        low_freq = np.sin(weak_conv.x)
        # High frequency component (decreasing amplitude)
        high_freq = 0.1 * np.sin(n * 10 * weak_conv.x) / n
        wavefront = low_freq + high_freq
        wavefront_sequence.append(wavefront)
    
    # Different measurement apertures
    measurement_apertures = {
        'full_aperture': np.ones_like(weak_conv.x),
        'center_spot': np.exp(-weak_conv.x**2 / 0.1**2),
        'edge_sensor': np.abs(weak_conv.x) > 0.8,
        'gradient_sensor': np.gradient(np.sin(weak_conv.x))
    }
    
    measurement_results = weak_conv.optical_measurement_convergence(
        wavefront_sequence, measurement_apertures
    )
    
    print("Measurement convergence results:")
    for aperture_name, result in measurement_results.items():
        print(f"  {aperture_name}: converged = {result['converged']}, "
              f"final value = {result['final_value']:.4f}")
    
    # Test 4: Weak Compactness
    print(f"\n--- Test 4: Weak Compactness ---")
    
    # Create bounded sequence
    bounded_sequence = []
    for n in range(1, 31):
        fn = np.sin(n * weak_conv.x) / np.sqrt(n)
        bounded_sequence.append(fn)
    
    compactness_result = weak_conv.weak_compactness_demo(bounded_sequence)
    
    print(f"Sequence is bounded: {compactness_result['is_bounded']}")
    print(f"Maximum norm: {compactness_result['max_norm']:.4f}")
    print(f"Has convergent subsequence: {compactness_result['has_convergent_subsequence']}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Weak convergence: ∫fₙg → ∫fg for all test functions g")
    print("2. Strong convergence implies weak convergence, but not vice versa")
    print("3. In optimization, functional convergence is often sufficient")
    print("4. Optical measurements represent weak convergence")
    print("5. Bounded sequences in infinite dimensions have weakly convergent subsequences")


if __name__ == "__main__":
    demonstrate_weak_convergence()