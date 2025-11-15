"""
Chapter 10: Distribution Theory in Optical Systems
Functional Analysis for Optical Design

Distribution theory applications including delta functions, Green's functions,
and generalized Fourier transforms in optical contexts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


class DistributionTheoryOptics:
    """
    Distribution theory in optical systems:
    - Delta functions for point sources
    - Green's functions for propagation
    - Principal value integrals
    - Generalized Fourier transforms
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def delta_function_approximation(self, center, width=0.01, method='gaussian'):
        """
        Approximate delta function δ(x-a) using different methods:
        - Gaussian: (1/√(2πσ²))exp(-(x-a)²/(2σ²))
        - Lorentzian: (1/π)ε/((x-a)²+ε²)
        - Sinc: sin(π(x-a)/ε)/(π(x-a))
        """
        if method == 'gaussian':
            return np.exp(-(self.x - center)**2 / (2 * width**2)) / (width * np.sqrt(2 * np.pi))
        elif method == 'lorentzian':
            return (1/np.pi) * width / ((self.x - center)**2 + width**2)
        elif method == 'sinc':
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.sinc((self.x - center) / width) / width
                result[np.isnan(result)] = 1.0 / width  # Handle division by zero
            return result
        else:
            raise ValueError("Method must be 'gaussian', 'lorentzian', or 'sinc'")
    
    def test_delta_properties(self, delta_approx, center=0.0):
        """
        Test properties of delta function approximation:
        1. ∫δ(x-a)dx = 1
        2. ∫f(x)δ(x-a)dx = f(a)
        3. δ(x-a) = 0 for x ≠ a
        """
        # Property 1: Normalization
        integral = np.sum(delta_approx) * self.dx
        
        # Property 2: Sifting property with test function
        test_function = np.sin(2 * np.pi * self.x)
        sifting_integral = np.sum(test_function * delta_approx) * self.dx
        expected_value = np.sin(2 * np.pi * center)
        
        # Property 3: Localization (check maximum near center)
        max_index = np.argmax(delta_approx)
        max_location = self.x[max_index]
        localization_error = abs(max_location - center)
        
        return {
            'normalization': integral,
            'sifting_property': sifting_integral,
            'expected_sifting_value': expected_value,
            'sifting_error': abs(sifting_integral - expected_value),
            'localization_error': localization_error,
            'max_value': np.max(delta_approx),
            'width_estimate': 1.0 / np.max(delta_approx)  # Rough width estimate
        }
    
    def greens_function_helmholtz(self, source_position, wavenumber=2*np.pi/500e-9, dimension=1):
        """
        Green's function for Helmholtz equation: (∇² + k²)G(x,x₀) = -δ(x-x₀)
        
        - 1D: G(x,x₀) = (i/2k)exp(ik|x-x₀|)
        - 2D: G(x,x₀) = (i/4)H₀(k|x-x₀|)  [Hankel function]
        - 3D: G(x,x₀) = exp(ik|x-x₀|)/(4π|x-x₀|)
        """
        r = np.abs(self.x - source_position)
        
        if dimension == 1:
            return (1j / (2 * wavenumber)) * np.exp(1j * wavenumber * r)
        elif dimension == 2:
            # Simplified 2D case (would need scipy.special.hankel1 in practice)
            from scipy.special import hankel1
            return (1j / 4) * hankel1(0, wavenumber * r)
        elif dimension == 3:
            # Simplified 3D case
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.exp(1j * wavenumber * r) / (4 * np.pi * r)
                result[np.isnan(result)] = np.inf  # Handle r=0 case
            return result
        else:
            raise ValueError("Dimension must be 1, 2, or 3")
    
    def point_source_response(self, source_position, observation_points, wavenumber=500e-9, dimension=1):
        """
        Response to point source using Green's function method.
        
        φ(x) = ∫G(x,y)ρ(y)dy where ρ(y) = δ(y-x₀) [point source]
        """
        green_function = self.greens_function_helmholtz(source_position, 2*np.pi/wavenumber, dimension)
        
        # Evaluate at observation points
        responses = []
        for obs_point in observation_points:
            idx = np.argmin(np.abs(self.x - obs_point))
            responses.append(green_function[idx])
        
        return np.array(responses)
    
    def principal_value_integral(self, function, singular_point, method='symmetric'):
        """
        Compute principal value integral: P∫f(x)/(x-a)dx
        
        Methods:
        - 'symmetric': Symmetric limit around singularity
        - 'subtract': Subtract singularity explicitly
        - 'complex': Use complex analysis (residue theorem)
        """
        if method == 'symmetric':
            # Symmetric limit: lim[ε→0] (∫_{-∞}^{a-ε} + ∫_{a+ε}^{∞}) f(x)/(x-a) dx
            eps_values = np.logspace(-3, -1, 10)  # Decreasing epsilon values
            pv_integrals = []
            
            for eps in eps_values:
                # Exclude small interval around singularity
                mask = np.abs(self.x - singular_point) > eps
                safe_x = self.x[mask]
                safe_func = function[mask]
                
                if len(safe_x) > 0:
                    integrand = safe_func / (safe_x - singular_point)
                    integral = np.trapz(integrand, safe_x)
                    pv_integrals.append(integral)
            
            # Extrapolate to ε → 0
            if len(pv_integrals) >= 3:
                # Simple extrapolation using last few values
                return pv_integrals[-1]
            else:
                return 0.0
        
        elif method == 'subtract':
            # Subtract singularity: f(x)/(x-a) = [f(x)-f(a)]/(x-a) + f(a)/(x-a)
            # First term is regular, second term integrates to zero for symmetric intervals
            
            f_at_singularity = np.interp(singular_point, self.x, function)
            
            # Regular part: [f(x)-f(a)]/(x-a)
            with np.errstate(divide='ignore', invalid='ignore'):
                regular_integrand = (function - f_at_singularity) / (self.x - singular_point)
                regular_integrand[np.isnan(regular_integrand)] = 0.0
            
            return np.trapz(regular_integrand, self.x)
        
        else:
            raise ValueError("Method must be 'symmetric', 'subtract', or 'complex'")
    
    def hilbert_transform(self, function):
        """
        Hilbert transform: H[f](x) = (1/π) P∫f(y)/(y-x)dy
        
        Important in signal processing and causal systems.
        """
        hilbert_result = np.zeros_like(function)
        
        for i, x_val in enumerate(self.x):
            # Principal value integral for each point
            pv_integral = self.principal_value_integral(function, x_val, method='subtract')
            hilbert_result[i] = pv_integral / np.pi
        
        return hilbert_transform
    
    def generalized_fourier_transform(self, distribution, regularization_method='gaussian'):
        """
        Compute Fourier transform of distributions/generalized functions.
        
        For tempered distributions, Fourier transform is well-defined.
        """
        # Regularize the distribution
        if regularization_method == 'gaussian':
            # Multiply by Gaussian to ensure integrability
            regularization = np.exp(-self.x**2 / 0.1**2)
            regularized = distribution * regularization
        else:
            regularized = distribution
        
        # Compute Fourier transform
        fft_result = fft(regularized)
        
        # Frequency axis
        frequencies = np.fft.fftfreq(len(self.x), self.dx)
        
        return frequencies, fft_result
    
    def optical_causality_demo(self, time_signal, speed_of_light=3e8):
        """
        Demonstrate causality using distribution theory.
        
        Causal signals have specific analytic properties in frequency domain.
        """
        # Time domain analysis
        time_step = 1.0 / (len(time_signal) * self.dx)
        
        # Check causality: signal must be zero for t < 0
        half_point = len(time_signal) // 2
        is_causal = np.all(np.abs(time_signal[:half_point]) < 1e-10)
        
        # Frequency domain analysis
        frequencies, spectrum = self.generalized_fourier_transform(time_signal)
        
        # Kramers-Kronig relations (causality implies specific relations)
        real_part = np.real(spectrum)
        imaginary_part = np.imag(spectrum)
        
        # Hilbert transform relationship for causal signals
        hilbert_real = self.hilbert_transform(real_part)
        
        return {
            'is_causal': is_causal,
            'spectrum': spectrum,
            'frequencies': frequencies,
            'real_part': real_part,
            'imaginary_part': imaginary_part,
            'hilbert_of_real': hilbert_real,
            'kk_relation_satisfied': np.allclose(imaginary_part, -hilbert_real, atol=1e-2)
        }
    
    def distribution_derivative_demo(self, piecewise_function, discontinuity_points):
        """
        Demonstrate derivatives of discontinuous functions in distribution sense.
        
        For discontinuous functions, derivatives include delta functions
        at discontinuity points.
        """
        # Compute classical derivative where possible
        classical_derivative = np.gradient(piecewise_function, self.dx)
        
        # Find discontinuities (large gradients)
        large_gradients = np.abs(classical_derivative) > 10 * np.mean(np.abs(classical_derivative))
        
        # Distribution derivative includes delta functions at jumps
        delta_contributions = []
        jump_sizes = []
        
        for i, is_large in enumerate(large_gradients):
            if is_large:
                # Estimate jump size
                if i > 0 and i < len(piecewise_function) - 1:
                    jump_size = piecewise_function[i+1] - piecewise_function[i-1]
                    jump_sizes.append(jump_size)
                    
                    # Create delta function at this point
                    delta_func = self.delta_function_approximation(self.x[i], width=3*self.dx)
                    delta_contributions.append(jump_size * delta_func)
        
        # Total distribution derivative
        if delta_contributions:
            total_delta = np.sum(delta_contributions, axis=0)
        else:
            total_delta = np.zeros_like(self.x)
        
        return {
            'classical_derivative': classical_derivative,
            'discontinuity_points': self.x[large_gradients],
            'jump_sizes': jump_sizes,
            'delta_contributions': delta_contributions,
            'distribution_derivative': classical_derivative + total_delta,
            'number_of_deltas': len(delta_contributions)
        }


def demonstrate_distribution_theory():
    """Demonstrate distribution theory concepts in optical systems."""
    print("Distribution Theory in Optical Systems")
    print("=" * 42)
    
    # Initialize distribution theory analyzer
    dist_theory = DistributionTheoryOptics(grid_size=200)
    
    # Test 1: Delta Function Properties
    print(f"\n--- Test 1: Delta Function Properties ---")
    
    center = 0.3
    delta_approx = dist_theory.delta_function_approximation(center, width=0.02, method='gaussian')
    
    delta_properties = dist_theory.test_delta_properties(delta_approx, center)
    print(f"Delta function normalization: {delta_properties['normalization']:.6f}")
    print(f"Sifting property error: {delta_properties['sifting_error']:.2e}")
    print(f"Localization error: {delta_properties['localization_error']:.2e}")
    
    # Test 2: Green's Function for Different Dimensions
    print(f"\n--- Test 2: Green's Function Analysis ---")
    
    source_pos = 0.0
    wavenumber = 2 * np.pi / 500e-9  # 500 nm light
    
    for dim in [1, 2, 3]:
        try:
            green_func = dist_theory.greens_function_helmholtz(source_pos, wavenumber, dimension=dim)
            max_amplitude = np.max(np.abs(green_func))
            print(f"Dimension {dim}D: max amplitude = {max_amplitude:.2e}")
        except Exception as e:
            print(f"Dimension {dim}D: {str(e)[:50]}...")
    
    # Test 3: Principal Value Integral
    print(f"\n--- Test 3: Principal Value Integral ---")
    
    test_function = np.sin(2 * np.pi * dist_theory.x)
    singular_point = 0.0
    
    pv_integral = dist_theory.principal_value_integral(test_function, singular_point, method='subtract')
    print(f"Principal value integral: {pv_integral:.6f}")
    
    # Compare with numerical integration excluding singularity
    eps = 0.01
    mask = np.abs(dist_theory.x - singular_point) > eps
    numerical_pv = np.trapz(test_function[mask] / (dist_theory.x[mask] - singular_point), 
                           dist_theory.x[mask])
    print(f"Numerical PV (eps={eps}): {numerical_pv:.6f}")
    
    # Test 4: Distribution Derivative
    print(f"\n--- Test 4: Distribution Derivative ---")
    
    # Create piecewise function with jump discontinuity
    piecewise_func = np.zeros_like(dist_theory.x)
    jump_location = 0.0
    
    for i, x_val in enumerate(dist_theory.x):
        if x_val < jump_location:
            piecewise_func[i] = 1.0
        else:
            piecewise_func[i] = 2.0
    
    derivative_result = dist_theory.distribution_derivative_demo(piecewise_func, [jump_location])
    
    print(f"Number of delta functions found: {derivative_result['number_of_deltas']}")
    if derivative_result['jump_sizes']:
        print(f"Jump sizes: {derivative_result['jump_sizes']}")
    
    # Test 5: Optical Causality
    print(f"\n--- Test 5: Optical Causality ---")
    
    # Create causal signal (zero for t < 0)
    time_signal = np.zeros_like(dist_theory.x)
    half_point = len(time_signal) // 2
    time_signal[half_point:] = np.exp(-dist_theory.x[half_point:]**2 / 0.1**2)
    
    causality_result = dist_theory.optical_causality_demo(time_signal)
    print(f"Signal is causal: {causality_result['is_causal']}")
    print(f"Kramers-Kronig relations satisfied: {causality_result['kk_relation_satisfied']}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Delta functions represent point sources in optics")
    print("2. Green's functions solve inhomogeneous differential equations")
    print("3. Principal value integrals handle singularities in physics")
    print("4. Distribution derivatives include delta functions at discontinuities")
    print("5. Causal signals have specific analytic properties")
    print("6. Distribution theory provides rigorous framework for generalized functions")


if __name__ == "__main__":
    demonstrate_distribution_theory()