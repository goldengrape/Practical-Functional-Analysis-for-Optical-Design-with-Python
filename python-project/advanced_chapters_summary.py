"""
Advanced Chapters Summary: Chapters 6-13
Functional Analysis for Optical Design

This file provides key concepts and implementations for the advanced chapters
of the textbook, focusing on practical applications in optical design.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import pandas as pd


# Chapter 6: Integral Equations in Optics
class IntegralEquationsOptics:
    """
    Integral equations in optical systems:
    - Fredholm equations for scattering
    - Volterra equations for propagation
    - Numerical solution methods
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def fredholm_scattering_equation(self, kernel_func, source_func, num_iterations=50):
        """
        Solve Fredholm integral equation for optical scattering:
        φ(x) = f(x) + λ∫K(x,y)φ(y)dy
        """
        # Discretize the equation
        K = np.zeros((self.grid_size, self.grid_size))
        for i, x in enumerate(self.x):
            for j, y in enumerate(self.x):
                K[i, j] = kernel_func(x, y)
        
        # Source function
        f = np.array([source_func(x) for x in self.x])
        
        # Solve using Neumann series
        phi = f.copy()
        lambda_val = 0.5
        
        for iteration in range(num_iterations):
            integral_term = lambda_val * np.dot(K, phi) * self.dx
            phi_new = f + integral_term
            
            # Check convergence
            error = np.linalg.norm(phi_new - phi)
            if error < 1e-10:
                print(f"Converged after {iteration+1} iterations")
                break
            
            phi = phi_new
        
        return phi
    
    def volterra_propagation_equation(self, initial_field, propagation_distance, absorption_coeff):
        """
        Solve Volterra integral equation for light propagation with absorption:
        E(z) = E₀ + ∫₀ᶻ K(z,ζ)E(ζ)dζ
        """
        z_vals = np.linspace(0, propagation_distance, self.grid_size)
        dz = z_vals[1] - z_vals[0]
        
        # Initialize solution array
        E = np.zeros_like(z_vals, dtype=complex)
        E[0] = initial_field
        
        # Solve using trapezoidal rule
        for i in range(1, len(z_vals)):
            z = z_vals[i]
            
            # Integral term using accumulated solution
            integral = 0
            for j in range(i):
                zeta = z_vals[j]
                # Kernel for absorption: K(z,ζ) = -α * exp(-α(z-ζ))
                kernel = -absorption_coeff * np.exp(-absorption_coeff * (z - zeta))
                integral += kernel * E[j] * dz
            
            E[i] = initial_field + integral
        
        return z_vals, E


# Chapter 7: Nonlinear Operators in Optics
class NonlinearOptics:
    """
    Nonlinear optical phenomena:
    - Kerr effect
    - Second harmonic generation
    - Soliton propagation
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-10, 10, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def kerr_nonlinearity(self, field, n2=2.5e-20):
        """
        Kerr nonlinearity: n = n₀ + n₂|E|²
        """
        intensity = np.abs(field)**2
        refractive_index = 1.0 + n2 * intensity
        return refractive_index
    
    def nonlinear_schrodinger_equation(self, initial_pulse, beta2=-1e-26, gamma=1e-3, distance=1000):
        """
        Solve nonlinear Schrödinger equation for pulse propagation:
        i∂A/∂z = (β₂/2)∂²A/∂t² - γ|A|²A
        """
        # Split-step Fourier method
        dt = self.x[1] - self.x[0]
        dz = distance / 100  # 100 steps
        
        A = initial_pulse.copy()
        
        for step in range(100):
            # Linear step (frequency domain)
            A_freq = fft(A)
            omega = 2 * np.pi * fft(np.fft.fftfreq(len(self.x), dt))
            A_freq *= np.exp(-1j * beta2 * omega**2 * dz / 4)
            A = ifft(A_freq)
            
            # Nonlinear step (time domain)
            A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)
            
            # Linear step (frequency domain)
            A_freq = fft(A)
            A_freq *= np.exp(-1j * beta2 * omega**2 * dz / 4)
            A = ifft(A_freq)
        
        return A
    
    def second_harmonic_generation(self, fundamental_field, chi2=1e-12, interaction_length=1e-3):
        """
        Second harmonic generation: E₂ω ∝ χ²Eω²
        """
        # Coupled wave equations (simplified)
        second_harmonic = chi2 * fundamental_field**2 * interaction_length
        return second_harmonic


# Chapter 8: Banach Spaces in Optical Design
class BanachSpaceOptics:
    """
    Banach space concepts in optical design:
    - Lᵖ spaces for different optical metrics
    - Norm equivalence
    - Completeness properties
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def lp_norm(self, function, p=2):
        """Compute Lᵖ norm of a function."""
        return (np.sum(np.abs(function)**p) * self.dx)**(1/p)
    
    def sobolev_norm(self, function, s=1):
        """Compute Sobolev norm Hˢ (includes derivatives)."""
        # Simple approximation using finite differences
        derivative = np.gradient(function, self.dx)
        
        # H¹ norm: ‖f‖² = ‖f‖²₂ + ‖f'‖²₂
        h1_norm_squared = self.lp_norm(function, 2)**2 + self.lp_norm(derivative, 2)**2
        return np.sqrt(h1_norm_squared)
    
    def optical_quality_metrics(self, wavefront_error):
        """
        Different optical quality metrics as Banach space norms:
        - RMS: L² norm
        - Peak-to-valley: L∞ norm
        - Gradient: Sobolev norm
        """
        metrics = {
            'rms_error': self.lp_norm(wavefront_error, 2),
            'peak_valley': np.max(wavefront_error) - np.min(wavefront_error),
            'gradient_energy': self.sobolev_norm(wavefront_error, 1),
            'total_variation': self.lp_norm(np.gradient(wavefront_error), 1)
        }
        return metrics


# Chapter 9: Weak Convergence in Optical Optimization
class WeakConvergenceOptics:
    """
    Weak convergence concepts in optical optimization:
    - Weak vs strong convergence
    - Applications in iterative algorithms
    - Convergence rates
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def weak_convergence_test(self, sequence_functions, test_function):
        """
        Test weak convergence: ∫fₙ(x)g(x)dx → ∫f(x)g(x)dx
        """
        integrals = []
        for fn in sequence_functions:
            integral = np.sum(fn * test_function) * self.dx
            integrals.append(integral)
        
        return np.array(integrals)
    
    def gradient_descent_convergence(self, initial_guess, merit_function, gradient_function, num_iterations=100):
        """
        Analyze convergence of gradient descent in function spaces.
        """
        x = initial_guess.copy()
        convergence_history = []
        
        for iteration in range(num_iterations):
            # Current merit
            current_merit = merit_function(x)
            
            # Gradient
            grad = gradient_function(x)
            
            # Update
            x_new = x - 0.01 * grad
            
            # Convergence measures
            strong_convergence = np.linalg.norm(x_new - x)
            weak_convergence = np.sum(grad * (x_new - x)) * self.dx
            
            convergence_history.append({
                'iteration': iteration,
                'merit': current_merit,
                'strong_convergence': strong_convergence,
                'weak_convergence': weak_convergence,
                'gradient_norm': np.linalg.norm(grad)
            })
            
            x = x_new
        
        return convergence_history


# Chapter 10: Distribution Theory in Optics
class DistributionTheoryOptics:
    """
    Distribution theory applications in optics:
    - Delta functions for point sources
    - Green's functions for propagation
    - Generalized Fourier transforms
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def delta_function_approximation(self, x0, width=0.01):
        """
        Approximate delta function using Gaussian.
        """
        return np.exp(-(self.x - x0)**2 / (2 * width**2)) / (width * np.sqrt(2 * np.pi))
    
    def greens_function_propagation(self, source_position, wavenumber):
        """
        Green's function for 1D wave propagation.
        G(x,x₀) = exp(ik|x-x₀|)
        """
        return np.exp(1j * wavenumber * np.abs(self.x - source_position))
    
    def point_source_response(self, source_position, propagation_distance, wavelength=500e-9):
        """
        Response to point source using Green's function method.
        """
        k = 2 * np.pi / wavelength
        
        # Green's function for free propagation
        green_function = self.greens_function_propagation(source_position, k)
        
        # Apply propagation phase factor
        propagated = green_function * np.exp(1j * k * propagation_distance)
        
        return propagated


# Chapter 11: Advanced Optimization Algorithms
class AdvancedOptimizationOptics:
    """
    Advanced optimization algorithms for optical design:
    - Newton methods in function spaces
    - Trust region methods
    - Multi-objective optimization
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def newton_method_functional(self, initial_guess, merit_function, gradient_function, hessian_function, num_iterations=20):
        """
        Newton's method in function spaces.
        xₙ₊₁ = xₙ - [H(xₙ)]⁻¹∇f(xₙ)
        """
        x = initial_guess.copy()
        history = []
        
        for iteration in range(num_iterations):
            # Compute gradient and Hessian
            grad = gradient_function(x)
            hessian = hessian_function(x)
            
            # Solve Newton system: HΔx = -∇f
            try:
                # For large systems, use iterative solvers
                delta_x = -np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Hessian is singular
                delta_x = -0.01 * grad
            
            # Update
            x_new = x + delta_x
            
            # Store history
            history.append({
                'iteration': iteration,
                'merit': merit_function(x),
                'gradient_norm': np.linalg.norm(grad),
                'step_size': np.linalg.norm(delta_x)
            })
            
            x = x_new
        
        return history
    
    def multi_objective_optimization(self, initial_design, objectives, weights):
        """
        Multi-objective optimization for optical systems.
        """
        def combined_objective(design):
            total_objective = 0
            for i, (objective, weight) in enumerate(zip(objectives, weights)):
                total_objective += weight * objective(design)
            return total_objective
        
        # Optimize using scipy
        result = minimize(combined_objective, initial_design, method='BFGS')
        
        return result


# Chapter 12: Uncertainty Quantification
class UncertaintyQuantificationOptics:
    """
    Uncertainty quantification in optical systems:
    - Monte Carlo methods
    - Polynomial chaos expansion
    - Sensitivity analysis
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def monte_carlo_optical_simulation(self, optical_model, parameter_distributions):
        """
        Monte Carlo simulation for optical performance analysis.
        """
        results = []
        
        for sample in range(self.num_samples):
            # Sample parameters from distributions
            sampled_params = {}
            for param, distribution in parameter_distributions.items():
                if distribution['type'] == 'normal':
                    sampled_params[param] = np.random.normal(
                        distribution['mean'], distribution['std']
                    )
                elif distribution['type'] == 'uniform':
                    sampled_params[param] = np.random.uniform(
                        distribution['min'], distribution['max']
                    )
            
            # Run optical simulation
            result = optical_model(sampled_params)
            results.append(result)
        
        return np.array(results)
    
    def sensitivity_analysis(self, base_model, parameter_ranges):
        """
        Sensitivity analysis using finite differences.
        """
        sensitivities = {}
        
        # Base case
        base_result = base_model()
        
        for param, (min_val, max_val) in parameter_ranges.items():
            # Perturb parameter
            delta = (max_val - min_val) * 0.01
            
            # Forward difference
            def perturbed_model():
                perturbed_params = {param: base_model.__defaults__[0] + delta}
                return base_model(**perturbed_params)
            
            perturbed_result = perturbed_model()
            
            # Sensitivity
            sensitivity = (perturbed_result - base_result) / delta
            sensitivities[param] = sensitivity
        
        return sensitivities


# Chapter 13: AI Integration in Optical Design
class AIIntegrationOptics:
    """
    AI and machine learning integration in optical design:
    - Neural networks for inverse design
    - Surrogate modeling
    - Reinforcement learning for optimization
    """
    
    def __init__(self):
        pass
    
    def neural_network_surrogate(self, training_data, architecture=[64, 32, 16]):
        """
        Simple neural network surrogate model for optical simulation.
        """
        # This is a placeholder - in practice, use TensorFlow/PyTorch
        print(f"Training neural network with architecture: {architecture}")
        print(f"Training data size: {len(training_data)}")
        
        return "Neural network model trained"
    
    def inverse_design_optimization(self, target_performance, forward_model, initial_design):
        """
        Inverse design using optimization-based approach.
        """
        def objective(design):
            performance = forward_model(design)
            return np.sum((performance - target_performance)**2)
        
        # Optimize design to match target performance
        result = minimize(objective, initial_design, method='BFGS')
        
        return result.x
    
    def reinforcement_learning_optics(self, environment, agent, num_episodes=1000):
        """
        Reinforcement learning for optical system control.
        """
        rewards = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            
            while not environment.is_done():
                action = agent.select_action(state)
                next_state, reward, done = environment.step(action)
                
                agent.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.3f}")
        
        return rewards


def main():
    """Demonstrate advanced concepts from chapters 6-13."""
    print("Advanced Functional Analysis for Optical Design")
    print("Chapters 6-13: Key Concepts and Applications")
    print("=" * 55)
    
    # Chapter 6: Integral Equations
    print("\n--- Chapter 6: Integral Equations ---")
    integral_eq = IntegralEquationsOptics()
    
    # Simple scattering kernel
    def scattering_kernel(x, y):
        return np.exp(-abs(x - y))
    
    def source_function(x):
        return np.sin(x)
    
    solution = integral_eq.fredholm_scattering_equation(scattering_kernel, source_function)
    print(f"Fredholm equation solution length: {len(solution)}")
    
    # Chapter 7: Nonlinear Operators
    print("\n--- Chapter 7: Nonlinear Operators ---")
    nonlinear = NonlinearOptics()
    
    # Test pulse propagation
    initial_pulse = np.exp(-np.linspace(-10, 10, 100)**2)
    final_pulse = nonlinear.nonlinear_schrodinger_equation(initial_pulse)
    print(f"Nonlinear propagation completed")
    
    # Chapter 8: Banach Spaces
    print("\n--- Chapter 8: Banach Spaces ---")
    banach = BanachSpaceOptics()
    
    test_function = np.sin(np.linspace(-1, 1, 100))
    l2_norm = banach.lp_norm(test_function, 2)
    sobolev_norm = banach.sobolev_norm(test_function, 1)
    print(f"L² norm: {l2_norm:.4f}")
    print(f"Sobolev norm: {sobolev_norm:.4f}")
    
    # Chapter 9: Weak Convergence
    print("\n--- Chapter 9: Weak Convergence ---")
    weak_conv = WeakConvergenceOptics()
    
    # Test sequence: fₙ(x) = sin(nx)/n
    test_sequence = [np.sin(n * np.linspace(-1, 1, 100)) / n for n in range(1, 11)]
    test_function = np.cos(np.linspace(-1, 1, 100))
    
    weak_integrals = weak_conv.weak_convergence_test(test_sequence, test_function)
    print(f"Weak convergence test: first integral = {weak_integrals[0]:.6f}, last integral = {weak_integrals[-1]:.6f}")
    
    # Chapter 10: Distribution Theory
    print("\n--- Chapter 10: Distribution Theory ---")
    dist_theory = DistributionTheoryOptics()
    
    delta_approx = dist_theory.delta_function_approximation(0.5, 0.01)
    print(f"Delta function approximation integral: {np.sum(delta_approx) * 0.02:.6f}")
    
    # Chapter 11: Advanced Optimization
    print("\n--- Chapter 11: Advanced Optimization ---")
    adv_opt = AdvancedOptimizationOptics()
    
    # Simple quadratic function
    def merit_func(x):
        return np.sum(x**2)
    
    def grad_func(x):
        return 2 * x
    
    def hessian_func(x):
        return np.eye(len(x)) * 2
    
    initial = np.random.randn(100)
    newton_history = adv_opt.newton_method_functional(initial, merit_func, grad_func, hessian_func)
    print(f"Newton method: {len(newton_history)} iterations")
    
    # Chapter 12: Uncertainty Quantification
    print("\n--- Chapter 12: Uncertainty Quantification ---")
    uq = UncertaintyQuantificationOptics()
    
    # Simple optical model
    def simple_optical_model(params):
        return params['focal_length'] * params['aperture']
    
    parameter_distributions = {
        'focal_length': {'type': 'normal', 'mean': 0.1, 'std': 0.01},
        'aperture': {'type': 'uniform', 'min': 0.01, 'max': 0.02}
    }
    
    results = uq.monte_carlo_optical_simulation(simple_optical_model, parameter_distributions)
    print(f"Monte Carlo simulation: mean = {np.mean(results):.6f}, std = {np.std(results):.6f}")
    
    # Chapter 13: AI Integration
    print("\n--- Chapter 13: AI Integration ---")
    ai_optics = AIIntegrationOptics()
    
    # Generate training data
    training_data = [(np.random.randn(10), np.random.randn(5)) for _ in range(100)]
    model = ai_optics.neural_network_surrogate(training_data)
    print(f"AI surrogate model: {model}")
    
    print("\n=== Summary of Advanced Concepts ===")
    print("1. Integral equations model scattering and propagation")
    print("2. Nonlinear operators capture intensity-dependent effects")
    print("3. Banach spaces provide different metrics for optimization")
    print("4. Weak convergence important for iterative algorithms")
    print("5. Distribution theory handles point sources and Green's functions")
    print("6. Advanced optimization accelerates convergence")
    print("7. Uncertainty quantification essential for robust design")
    print("8. AI integration enables inverse design and acceleration")


if __name__ == "__main__":
    main()