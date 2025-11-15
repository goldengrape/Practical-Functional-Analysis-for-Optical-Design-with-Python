"""
Chapter 6: Integral Equations in Optical Systems
Functional Analysis for Optical Design

This module implements integral equations for optical scattering and propagation,
demonstrating Fredholm and Volterra equations in optical contexts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.linalg import solve


class IntegralEquationsOptics:
    """
    Integral equations in optical systems:
    - Fredholm equations for scattering problems
    - Volterra equations for propagation problems
    - Numerical solution methods and applications
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def fredholm_scattering_equation(self, kernel_func, source_func, lambda_param=0.5, num_iterations=100):
        """
        Solve Fredholm integral equation of the second kind:
        φ(x) = f(x) + λ∫K(x,y)φ(y)dy
        
        Applications: Optical scattering, multiple scattering problems
        """
        # Discretize the integral equation
        K = np.zeros((self.grid_size, self.grid_size))
        
        # Build kernel matrix
        for i, x_i in enumerate(self.x):
            for j, y_j in enumerate(self.x):
                K[i, j] = kernel_func(x_i, y_j)
        
        # Source function vector
        f = np.array([source_func(x_i) for x_i in self.x])
        
        # Solve using successive approximations (Neumann series)
        phi = f.copy()  # Initial guess
        convergence_history = []
        
        for iteration in range(num_iterations):
            # Compute integral term: ∫K(x,y)φ(y)dy
            integral_term = np.dot(K, phi) * self.dx
            
            # Update solution: φ_new(x) = f(x) + λ∫K(x,y)φ(y)dy
            phi_new = f + lambda_param * integral_term
            
            # Check convergence
            error = np.linalg.norm(phi_new - phi) / np.linalg.norm(phi_new)
            convergence_history.append(error)
            
            if error < 1e-10:
                print(f"Fredholm equation converged after {iteration+1} iterations")
                break
            
            phi = phi_new
        
        return phi, convergence_history
    
    def volterra_propagation_equation(self, initial_field, propagation_distance, absorption_coeff=0.1):
        """
        Solve Volterra integral equation for light propagation with absorption:
        E(z) = E₀ + ∫₀ᶻ K(z,ζ)E(ζ)dζ
        
        Applications: Propagation in absorbing media, gain/lossy systems
        """
        z_vals = np.linspace(0, propagation_distance, self.grid_size)
        dz = z_vals[1] - z_vals[0]
        
        # Initialize solution array
        E = np.zeros_like(z_vals, dtype=complex)
        E[0] = initial_field
        
        # Solve using trapezoidal rule (predictor-corrector)
        for i in range(1, len(z_vals)):
            z = z_vals[i]
            
            # Integral term using accumulated solution
            integral = 0.0
            for j in range(i):
                zeta = z_vals[j]
                # Kernel for absorption: K(z,ζ) = -α * exp(-α(z-ζ))
                kernel = -absorption_coeff * np.exp(-absorption_coeff * (z - zeta))
                integral += kernel * E[j] * dz
            
            E[i] = initial_field + integral
        
        return z_vals, E
    
    def optical_resolvent_kernel(self, kernel_func, lambda_param=0.5):
        """
        Compute resolvent kernel for Fredholm equation.
        R(x,y;λ) = K(x,y) + λ∫K(x,z)R(z,y;λ)dz
        
        The resolvent kernel allows direct solution: φ(x) = f(x) + ∫R(x,y;λ)f(y)dy
        """
        # Discretize kernel
        K = np.zeros((self.grid_size, self.grid_size))
        for i, x_i in enumerate(self.x):
            for j, y_j in enumerate(self.x):
                K[i, j] = kernel_func(x_i, y_j)
        
        # Compute resolvent kernel using matrix inversion
        I = np.eye(self.grid_size)
        
        # R = (I - λK)⁻¹K
        try:
            resolvent_matrix = np.linalg.solve(I - lambda_param * K * self.dx, K)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            resolvent_matrix = np.linalg.pinv(I - lambda_param * K * self.dx) @ K
        
        return resolvent_matrix
    
    def born_approximation(self, scattering_potential, incident_field, wavenumber=2*np.pi/500e-9):
        """
        Born approximation for weak scattering:
        φ(x) ≈ φ₀(x) + ∫G(x,y)V(y)φ₀(y)dy
        
        Applications: Weak scattering, first-order approximation
        """
        # Green's function for Helmholtz equation in 1D
        def greens_function(x, y):
            return (1j / (2 * wavenumber)) * np.exp(1j * wavenumber * np.abs(x - y))
        
        # First Born approximation
        scattered_field = np.zeros_like(incident_field, dtype=complex)
        
        for i, x_i in enumerate(self.x):
            integral = 0.0
            for j, y_j in enumerate(self.x):
                # Green's function times potential times incident field
                integral += (greens_function(x_i, y_j) * 
                           scattering_potential[j] * 
                           incident_field[j] * self.dx)
            
            scattered_field[i] = incident_field[i] + integral
        
        return scattered_field
    
    def rytov_approximation(self, scattering_potential, incident_field, wavenumber=2*np.pi/500e-9):
        """
        Rytov approximation for phase perturbation:
        φ(x) = φ₀(x)exp(∫G(x,y)V(y)dy)
        
        Applications: Phase objects, smooth refractive index variations
        """
        # Green's function
        def greens_function(x, y):
            return (1j / (2 * wavenumber)) * np.exp(1j * wavenumber * np.abs(x - y))
        
        # Compute phase perturbation
        phase_perturbation = np.zeros_like(incident_field, dtype=complex)
        
        for i, x_i in enumerate(self.x):
            integral = 0.0
            for j, y_j in enumerate(self.x):
                integral += greens_function(x_i, y_j) * scattering_potential[j] * self.dx
            
            phase_perturbation[i] = integral
        
        # Rytov approximation: φ = φ₀ * exp(phase_perturbation/φ₀)
        rytov_field = incident_field * np.exp(phase_perturbation / (incident_field + 1e-10))
        
        return rytov_field
    
    def optical_coherence_integral(self, source_distribution, propagation_distance, coherence_length=1e-6):
        """
        Integral equation for optical coherence propagation:
        Γ(x₁,x₂,z) = ∫∫K(x₁,x₂,y₁,y₂,z)Γ(y₁,y₂,0)dy₁dy₂
        
        Applications: Coherence theory, partially coherent sources
        """
        # Simplified 1D case: mutual coherence function
        coherence_function = np.zeros((self.grid_size, self.grid_size), dtype=complex)
        
        # Initial coherence (assuming incoherent source)
        initial_coherence = np.outer(source_distribution, np.conj(source_distribution))
        
        # Propagation kernel (simplified)
        for i, x1 in enumerate(self.x):
            for j, x2 in enumerate(self.x):
                kernel_sum = 0.0
                
                for k, y1 in enumerate(self.x):
                    for l, y2 in enumerate(self.x):
                        # Propagation kernel (simplified free space)
                        r1 = np.sqrt(propagation_distance**2 + (x1 - y1)**2)
                        r2 = np.sqrt(propagation_distance**2 + (x2 - y2)**2)
                        
                        kernel = (np.exp(1j * 2*np.pi/500e-9 * (r1 - r2)) / 
                                (r1 * r2)) * np.exp(-((y1 - y2)/coherence_length)**2)
                        
                        kernel_sum += (kernel * 
                                     initial_coherence[k, l] * 
                                     self.dx * self.dx)
                
                coherence_function[i, j] = kernel_sum
        
        return coherence_function
    
    def iterative_solution_convergence(self, kernel_func, source_func, lambda_range):
        """
        Analyze convergence of iterative solutions for different λ values.
        """
        convergence_analysis = {}
        
        for lambda_val in lambda_range:
            phi, convergence_history = self.fredholm_scattering_equation(
                kernel_func, source_func, lambda_param=lambda_val, num_iterations=200
            )
            
            convergence_analysis[lambda_val] = {
                'convergence_history': convergence_history,
                'final_error': convergence_history[-1] if convergence_history else 1.0,
                'num_iterations': len(convergence_history),
                'converged': convergence_history[-1] < 1e-8 if convergence_history else False
            }
        
        return convergence_analysis


def demonstrate_integral_equations():
    """Demonstrate integral equations in optical systems."""
    print("Integral Equations in Optical Systems")
    print("=" * 42)
    
    # Initialize integral equation solver
    integral_eq = IntegralEquationsOptics(grid_size=50)
    
    # Test 1: Fredholm equation with Gaussian kernel
    print(f"\n--- Test 1: Fredholm Equation with Gaussian Kernel ---")
    
    def gaussian_kernel(x, y):
        """Gaussian kernel for scattering."""
        sigma = 0.2
        return np.exp(-(x - y)**2 / (2 * sigma**2))
    
    def source_function(x):
        """Source function."""
        return np.sin(2 * np.pi * x)
    
    # Solve Fredholm equation
    solution, convergence = integral_eq.fredholm_scattering_equation(
        gaussian_kernel, source_function, lambda_param=0.3
    )
    
    print(f"Fredholm equation solved with {len(convergence)} iterations")
    print(f"Final convergence error: {convergence[-1]:.2e}")
    
    # Test 2: Volterra equation for propagation
    print(f"\n--- Test 2: Volterra Equation for Propagation ---")
    
    initial_field = 1.0 + 0.0j
    propagation_distance = 1.0
    absorption_coeff = 0.5
    
    z_vals, propagated_field = integral_eq.volterra_propagation_equation(
        initial_field, propagation_distance, absorption_coeff
    )
    
    # Check energy conservation
    initial_energy = np.abs(initial_field)**2
    final_energy = np.abs(propagated_field[-1])**2
    energy_loss = (initial_energy - final_energy) / initial_energy
    
    print(f"Propagation distance: {propagation_distance} m")
    print(f"Energy loss: {energy_loss:.4f} ({energy_loss*100:.2f}%)")
    
    # Test 3: Born approximation
    print(f"\n--- Test 3: Born Approximation ---")
    
    # Weak scattering potential
    scattering_potential = 0.1 * np.exp(-integral_eq.x**2 / 0.1**2)
    incident_field = np.exp(1j * 2*np.pi/500e-9 * integral_eq.x)
    
    born_field = integral_eq.born_approximation(scattering_potential, incident_field)
    
    # Compare with incident field
    field_difference = np.linalg.norm(born_field - incident_field) / np.linalg.norm(incident_field)
    print(f"Born approximation field difference: {field_difference:.4f}")
    
    # Test 4: Convergence analysis
    print(f"\n--- Test 4: Convergence Analysis ---")
    
    lambda_range = np.linspace(0.1, 0.9, 9)
    convergence_analysis = integral_eq.iterative_solution_convergence(
        gaussian_kernel, source_function, lambda_range
    )
    
    print("Convergence analysis for different λ values:")
    for lam, result in convergence_analysis.items():
        print(f"  λ = {lam:.1f}: converged = {result['converged']}, "
              f"iterations = {result['num_iterations']}, "
              f"final error = {result['final_error']:.2e}")
    
    # Test 5: Resolvent kernel
    print(f"\n--- Test 5: Resolvent Kernel ---")
    
    resolvent_matrix = integral_eq.optical_resolvent_kernel(gaussian_kernel, lambda_param=0.3)
    
    # Verify resolvent property: R = K + λKR
    K_matrix = np.zeros((50, 50))
    for i, x_i in enumerate(integral_eq.x):
        for j, y_j in enumerate(integral_eq.x):
            K_matrix[i, j] = gaussian_kernel(x_i, y_j) * integral_eq.dx
    
    # Check resolvent equation
    identity_check = resolvent_matrix - (K_matrix + 0.3 * K_matrix @ resolvent_matrix)
    resolvent_error = np.linalg.norm(identity_check)
    print(f"Resolvent kernel error: {resolvent_error:.2e}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Fredholm equations model scattering problems with fixed integration limits")
    print("2. Volterra equations model propagation with variable upper limits")
    print("3. Born approximation is first-order solution for weak scattering")
    print("4. Resolvent kernel provides direct solution method")
    print("5. Convergence depends on the magnitude of λ (Neumann series)")
    print("6. Integral equations naturally handle nonlocal interactions in optics")


if __name__ == "__main__":
    demonstrate_integral_equations()