"""
Chapter 7: Nonlinear Operators in Optical Systems
Functional Analysis for Optical Design
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


class NonlinearOptics:
    """
    Nonlinear optical phenomena demonstrating nonlinear operators:
    - Kerr effect (intensity-dependent refractive index)
    - Second harmonic generation
    - Soliton propagation
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-10, 10, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def kerr_nonlinearity(self, electric_field, n2=2.5e-20):
        """
        Kerr nonlinearity: n = n₀ + n₂|E|²
        This is a nonlinear operator: N(E) = n₂|E|²E
        """
        intensity = np.abs(electric_field)**2
        nonlinear_refractive_index = n2 * intensity
        return nonlinear_refractive_index
    
    def nonlinear_wave_equation(self, initial_field, beta2=-1e-26, gamma=1e-3, propagation_distance=1000):
        """
        Solve nonlinear wave equation using split-step Fourier method:
        i∂A/∂z = (β₂/2)∂²A/∂t² - γ|A|²A
        
        This demonstrates solving nonlinear operator equations.
        """
        # Initialize field
        A = initial_field.copy()
        
        # Propagation parameters
        num_steps = 100
        dz = propagation_distance / num_steps
        
        # Frequency domain setup
        omega = 2 * np.pi * fftfreq(len(self.x), self.dx)
        
        for step in range(num_steps):
            # Linear step (frequency domain)
            A_freq = fft(A)
            A_freq *= np.exp(-1j * beta2 * omega**2 * dz / 2)
            A = ifft(A_freq)
            
            # Nonlinear step (time domain)
            nonlinear_phase = gamma * np.abs(A)**2 * dz
            A *= np.exp(1j * nonlinear_phase)
            
            # Linear step (frequency domain)
            A_freq = fft(A)
            A_freq *= np.exp(-1j * beta2 * omega**2 * dz / 2)
            A = ifft(A_freq)
        
        return A
    
    def second_harmonic_generation(self, fundamental_field, chi2=1e-12, interaction_length=1e-3):
        """
        Second harmonic generation: E₂ω ∝ χ²Eω²
        Demonstrates nonlinear frequency conversion.
        """
        # Coupled wave equations (simplified)
        second_harmonic = chi2 * fundamental_field**2 * interaction_length
        return second_harmonic
    
    def nonlinear_susceptibility_tensor(self, electric_fields, chi3=1e-21):
        """
        Third-order nonlinear susceptibility tensor:
        Pᵢ = χᵢⱼₖₗ EⱼEₖEₗ
        Demonstrates tensor nonlinear operators.
        """
        # Simplified scalar case
        E_total = np.sqrt(np.sum(np.abs(electric_fields)**2))
        nonlinear_polarization = chi3 * E_total**3
        return nonlinear_polarization
    
    def soliton_propagation(self, initial_sech_pulse, beta2=-1e-26, gamma=1e-3):
        """
        Soliton propagation - balance between dispersion and nonlinearity.
        Demonstrates stable nonlinear wave solutions.
        """
        # Fundamental soliton condition: N = 1
        # Propagation should maintain shape
        propagated_field = self.nonlinear_wave_equation(initial_sech_pulse, beta2, gamma, 1000)
        return propagated_field
    
    def nonlinear_operator_analysis(self, field_amplitude_range):
        """
        Analyze properties of nonlinear operators.
        """
        results = {}
        
        for amplitude in field_amplitude_range:
            # Test field
            test_field = amplitude * np.exp(-self.x**2)
            
            # Apply nonlinear operators
            kerr_effect = self.kerr_nonlinearity(test_field)
            shg_field = self.second_harmonic_generation(test_field)
            
            # Analyze linearity
            # Test if N(αE) = αN(E) (homogeneity)
            alpha = 2.0
            scaled_field = alpha * test_field
            scaled_kerr = self.kerr_nonlinearity(scaled_field)
            expected_scaled = alpha * kerr_effect
            
            homogeneity_error = np.linalg.norm(scaled_kerr - expected_scaled)
            
            # Test if N(E₁+E₂) = N(E₁)+N(E₂) (additivity)
            field2 = 0.5 * amplitude * np.exp(-(self.x-2)**2)
            sum_field = test_field + field2
            sum_kerr = self.kerr_nonlinearity(sum_field)
            individual_sum = kerr_effect + self.kerr_nonlinearity(field2)
            
            additivity_error = np.linalg.norm(sum_kerr - individual_sum)
            
            results[amplitude] = {
                'homogeneity_error': homogeneity_error,
                'additivity_error': additivity_error,
                'is_linear': homogeneity_error < 1e-10 and additivity_error < 1e-10
            }
        
        return results


def demonstrate_nonlinear_optics():
    """Demonstrate nonlinear optical phenomena."""
    print("Nonlinear Operators in Optical Systems")
    print("=" * 42)
    
    # Initialize nonlinear optics
    nonlinear = NonlinearOptics(grid_size=200)
    
    # Test field: Gaussian pulse
    initial_field = np.exp(-nonlinear.x**2)
    
    print(f"Initial field energy: {np.sum(np.abs(initial_field)**2):.6f}")
    
    # Kerr nonlinearity
    print(f"\n--- Kerr Nonlinearity ---")
    kerr_effect = nonlinear.kerr_nonlinearity(initial_field, n2=1e-20)
    max_nonlinear_index = np.max(kerr_effect)
    print(f"Maximum nonlinear refractive index: {max_nonlinear_index:.2e}")
    
    # Nonlinear wave propagation
    print(f"\n--- Nonlinear Wave Propagation ---")
    propagated_field = nonlinear.nonlinear_wave_equation(
        initial_field, beta2=-1e-27, gamma=1e-4, propagation_distance=100
    )
    final_energy = np.sum(np.abs(propagated_field)**2)
    print(f"Final field energy: {final_energy:.6f}")
    print(f"Energy conservation: {abs(final_energy - np.sum(np.abs(initial_field)**2)) < 1e-10}")
    
    # Second harmonic generation
    print(f"\n--- Second Harmonic Generation ---")
    shg_field = nonlinear.second_harmonic_generation(initial_field, chi2=1e-12)
    shg_efficiency = np.sum(np.abs(shg_field)**2) / np.sum(np.abs(initial_field)**2)
    print(f"SHG efficiency: {shg_efficiency:.2e}")
    
    # Soliton propagation
    print(f"\n--- Soliton Propagation ---")
    # Create sech pulse for fundamental soliton
    sech_pulse = 1.0 / np.cosh(nonlinear.x)
    soliton_field = nonlinear.soliton_propagation(sech_pulse)
    
    # Check if shape is preserved
    shape_error = np.linalg.norm(np.abs(soliton_field) - np.abs(sech_pulse))
    print(f"Soliton shape preservation error: {shape_error:.6f}")
    
    # Nonlinear operator analysis
    print(f"\n--- Nonlinear Operator Analysis ---")
    amplitude_range = np.logspace(-2, 1, 10)
    analysis = nonlinear.nonlinear_operator_analysis(amplitude_range)
    
    # Check first and last amplitude
    first_result = analysis[amplitude_range[0]]
    last_result = analysis[amplitude_range[-1]]
    
    print(f"Kerr operator linearity test:")
    print(f"  Low amplitude: homogeneous={first_result['is_linear']}, error={first_result['homogeneity_error']:.2e}")
    print(f"  High amplitude: homogeneous={last_result['is_linear']}, error={last_result['homogeneity_error']:.2e}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Kerr effect: intensity-dependent refractive index")
    print("2. Nonlinear wave equations require iterative solution methods")
    print("3. Second harmonic generation enables frequency conversion")
    print("4. Solitons are stable nonlinear wave solutions")
    print("5. Nonlinear operators violate superposition principle")


if __name__ == "__main__":
    demonstrate_nonlinear_optics()