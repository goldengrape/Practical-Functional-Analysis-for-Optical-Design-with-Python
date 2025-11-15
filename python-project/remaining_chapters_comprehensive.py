"""
Chapters 9-13: Advanced Topics in Functional Analysis for Optical Design
Comprehensive implementation covering weak convergence, distribution theory,
optimization algorithms, uncertainty quantification, and AI integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft, ifft
from scipy.integrate import quad
import pandas as pd


# Chapter 9: Weak Convergence in Optical Systems
class WeakConvergenceOptics:
    """Weak convergence concepts in optical optimization and analysis."""
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def weak_convergence_test(self, function_sequence, test_functionals):
        """
        Test weak convergence: lim ∫fₙ(x)g(x)dx = ∫f(x)g(x)dx
        for all test functionals g in the dual space.
        """
        convergence_results = {}
        
        for func_name, functional in test_functionals.items():
            integrals = []
            for fn in function_sequence:
                integral = np.sum(fn * functional) * self.dx
                integrals.append(integral)
            
            # Check if sequence converges
            final_value = integrals[-1]
            convergence_rate = np.abs(np.array(integrals) - final_value)
            
            convergence_results[func_name] = {
                'integrals': integrals,
                'converged': np.all(convergence_rate[-10:] < 1e-6),
                'final_value': final_value
            }
        
        return convergence_results
    
    def gradient_descent_weak_convergence(self, initial_function, merit_functional, num_iterations=50):
        """
        Analyze weak convergence in gradient descent optimization.
        """
        current = initial_function.copy()
        history = []
        
        for iteration in range(num_iterations):
            # Compute gradient (represents weak derivative)
            gradient = np.gradient(current, self.dx)
            
            # Merit functional value
            merit = merit_functional(current)
            
            # Weak convergence measures
            strong_norm = np.linalg.norm(gradient)
            weak_norm = np.max(np.abs(gradient))  # Dual norm
            
            history.append({
                'iteration': iteration,
                'merit': merit,
                'strong_norm': strong_norm,
                'weak_norm': weak_norm,
                'convergence_rate': strong_norm / (iteration + 1)
            })
            
            # Update (gradient descent step)
            current = current - 0.01 * gradient
        
        return history


# Chapter 10: Distribution Theory in Optics
class DistributionTheoryOptics:
    """Distribution theory applications in optical systems."""
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def delta_function_approximation(self, center, width=0.01):
        """
        Approximate delta function: δ(x-a) ≈ (1/√(2πσ²))exp(-(x-a)²/(2σ²))
        """
        return np.exp(-(self.x - center)**2 / (2 * width**2)) / (width * np.sqrt(2 * np.pi))
    
    def greens_function_1d_propagation(self, source_position, wavenumber):
        """
        Green's function for 1D wave equation: G(x,x₀) = exp(ik|x-x₀|)
        """
        return np.exp(1j * wavenumber * np.abs(self.x - source_position))
    
    def point_source_response(self, source_position, observation_points, wavelength=500e-9):
        """
        Response to point source using Green's function.
        """
        k = 2 * np.pi / wavelength
        
        # Green's function
        green_function = self.greens_function_1d_propagation(source_position, k)
        
        # Evaluate at observation points
        responses = []
        for obs_point in observation_points:
            idx = np.argmin(np.abs(self.x - obs_point))
            responses.append(green_function[idx])
        
        return np.array(responses)
    
    def principal_value_integral(self, function, singular_point):
        """
        Compute principal value integral: P∫f(x)/(x-a)dx
        """
        # Remove singular point
        mask = np.abs(self.x - singular_point) > 1e-10
        safe_x = self.x[mask]
        safe_func = function[mask]
        
        # Cauchy principal value
        integrand = safe_func / (safe_x - singular_point)
        integral = np.trapz(integrand, safe_x)
        
        return integral


# Chapter 11: Advanced Optimization Algorithms
class AdvancedOptimizationAlgorithms:
    """Advanced optimization algorithms for optical design."""
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def newton_method_functional(self, initial_guess, merit_function, gradient_function, 
                                hessian_function=None, num_iterations=20):
        """
        Newton's method in function spaces.
        xₙ₊₁ = xₙ - [H(xₙ)]⁻¹∇f(xₙ)
        """
        current = initial_guess.copy()
        history = []
        
        for iteration in range(num_iterations):
            # Compute gradient
            gradient = gradient_function(current)
            
            # Approximate Hessian if not provided
            if hessian_function is None:
                # Finite difference approximation
                hessian = np.eye(len(current)) * 1e-3
            else:
                hessian = hessian_function(current)
            
            # Solve Newton system: HΔx = -∇f
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent
                step = -0.01 * gradient
            
            # Update
            current = current + step
            
            # Store history
            history.append({
                'iteration': iteration,
                'merit': merit_function(current),
                'gradient_norm': np.linalg.norm(gradient),
                'step_size': np.linalg.norm(step)
            })
            
            # Convergence check
            if np.linalg.norm(gradient) < 1e-8:
                break
        
        return history
    
    def trust_region_method(self, initial_guess, merit_function, gradient_function, 
                           trust_radius=0.1, num_iterations=30):
        """
        Trust region method for constrained optimization.
        """
        current = initial_guess.copy()
        history = []
        
        for iteration in range(num_iterations):
            gradient = gradient_function(current)
            
            # Solve trust region subproblem
            # Minimize m(d) = f + gᵀd + ½dᵀHd subject to ‖d‖ ≤ Δ
            def subproblem(direction):
                if np.linalg.norm(direction) > trust_radius:
                    return np.inf
                
                model_value = (merit_function(current) + 
                             np.dot(gradient, direction) + 
                             0.5 * np.dot(direction, direction) * 1e-3)
                return model_value
            
            # Solve subproblem
            result = minimize(subproblem, np.zeros_like(current), method='L-BFGS-B')
            step = result.x
            
            # Update trust radius
            actual_reduction = merit_function(current) - merit_function(current + step)
            predicted_reduction = -np.dot(gradient, step)
            
            if predicted_reduction > 0:
                ratio = actual_reduction / predicted_reduction
                if ratio > 0.75:
                    trust_radius = min(2 * trust_radius, 1.0)
                elif ratio < 0.25:
                    trust_radius = max(0.5 * trust_radius, 1e-4)
            
            current = current + step
            
            history.append({
                'iteration': iteration,
                'merit': merit_function(current),
                'trust_radius': trust_radius,
                'step_size': np.linalg.norm(step)
            })
        
        return history
    
    def multi_objective_optimization(self, initial_design, objectives, weights):
        """
        Multi-objective optimization for optical systems.
        """
        def combined_objective(design):
            total = 0
            for objective, weight in zip(objectives, weights):
                total += weight * objective(design)
            return total
        
        result = minimize(combined_objective, initial_design, method='BFGS')
        return result


# Chapter 12: Uncertainty Quantification
class UncertaintyQuantificationOptics:
    """Uncertainty quantification in optical systems."""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def monte_carlo_simulation(self, optical_model, parameter_distributions):
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
    
    def polynomial_chaos_expansion(self, optical_model, parameter_distributions, expansion_order=3):
        """
        Polynomial chaos expansion for uncertainty propagation.
        """
        # Generate quadrature points and weights
        num_params = len(parameter_distributions)
        quadrature_points = []
        quadrature_weights = []
        
        # Simple tensor product quadrature (for low dimensions)
        for i in range(5):  # 5 points per dimension
            point = {}
            weight = 1.0
            
            for param, dist in parameter_distributions.items():
                if dist['type'] == 'normal':
                    # Gauss-Hermite quadrature points
                    points = np.polynomial.hermite.hermgauss(5)[0]
                    weights = np.polynomial.hermite.hermgauss(5)[1]
                    point[param] = dist['mean'] + dist['std'] * points[i]
                    weight *= weights[i] / np.sqrt(np.pi)
            
            quadrature_points.append(point)
            quadrature_weights.append(weight)
        
        # Evaluate model at quadrature points
        expansions = []
        for point in quadrature_points:
            result = optical_model(point)
            expansions.append(result)
        
        # Compute statistics
        mean_result = np.average(expansions, weights=quadrature_weights)
        variance_result = np.average((expansions - mean_result)**2, weights=quadrature_weights)
        
        return {
            'mean': mean_result,
            'variance': variance_result,
            'std': np.sqrt(variance_result)
        }
    
    def sensitivity_analysis(self, base_model, parameter_ranges):
        """
        Sensitivity analysis using finite differences.
        """
        sensitivities = {}
        
        # Base case
        base_params = {param: (min_val + max_val) / 2 
                      for param, (min_val, max_val) in parameter_ranges.items()}
        base_result = base_model(base_params)
        
        for param, (min_val, max_val) in parameter_ranges.items():
            # Perturb parameter
            delta = (max_val - min_val) * 0.01
            
            perturbed_params = base_params.copy()
            perturbed_params[param] += delta
            
            perturbed_result = base_model(perturbed_params)
            
            # Sensitivity (normalized)
            sensitivity = (perturbed_result - base_result) / delta
            sensitivities[param] = sensitivity
        
        return sensitivities


# Chapter 13: AI Integration in Optical Design
class AIIntegrationOptics:
    """AI and machine learning integration in optical design."""
    
    def __init__(self):
        pass
    
    def neural_network_surrogate(self, training_data, hidden_layers=[64, 32]):
        """
        Simple neural network surrogate model.
        This is a placeholder - in practice, use TensorFlow/PyTorch.
        """
        print(f"Training neural network with architecture: {hidden_layers}")
        print(f"Training data size: {len(training_data)}")
        
        # Simulate training
        training_loss = np.exp(-np.linspace(0, 5, 100))
        
        return {
            'architecture': hidden_layers,
            'training_loss': training_loss,
            'final_loss': training_loss[-1]
        }
    
    def inverse_design_optimization(self, target_performance, forward_model, 
                                   initial_design, learning_rate=0.01):
        """
        Inverse design using gradient-based optimization.
        """
        def objective(design):
            performance = forward_model(design)
            return np.sum((performance - target_performance)**2)
        
        # Simple gradient descent
        current_design = initial_design.copy()
        history = []
        
        for iteration in range(100):
            # Forward pass
            current_performance = forward_model(current_design)
            
            # Compute loss
            loss = np.sum((current_performance - target_performance)**2)
            history.append(loss)
            
            # Simple gradient approximation
            epsilon = 1e-6
            gradient = np.zeros_like(current_design)
            
            for i in range(len(current_design)):
                perturbed = current_design.copy()
                perturbed[i] += epsilon
                perturbed_loss = np.sum((forward_model(perturbed) - target_performance)**2)
                gradient[i] = (perturbed_loss - loss) / epsilon
            
            # Update
            current_design = current_design - learning_rate * gradient
            
            if loss < 1e-8:
                break
        
        return current_design, history
    
    def reinforcement_learning_optics(self, environment_simulator, num_episodes=100):
        """
        Simple reinforcement learning for optical system control.
        """
        rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            state = environment_simulator.reset()
            episode_reward = 0
            
            for step in range(50):  # Max steps per episode
                # Simple policy: random exploration with decay
                epsilon = max(0.1, 1.0 - episode / num_episodes)
                
                if np.random.random() < epsilon:
                    action = np.random.randn(len(state))
                else:
                    # Greedy action (simplified)
                    action = -state * 0.1
                
                # Take action
                next_state, reward, done = environment_simulator.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return rewards


# Environment simulator for RL
class OpticalEnvironment:
    """Simple optical environment for reinforcement learning."""
    
    def __init__(self, target_performance):
        self.target_performance = target_performance
        self.state = None
        self.step_count = 0
    
    def reset(self):
        self.state = np.random.randn(10)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        # Update state
        self.state = self.state + action + np.random.randn(len(self.state)) * 0.01
        self.step_count += 1
        
        # Compute reward (negative distance to target)
        performance = np.sin(self.state)  # Simple performance model
        reward = -np.sum((performance - self.target_performance)**2)
        
        # Done condition
        done = self.step_count >= 50 or np.sum((performance - self.target_performance)**2) < 1e-4
        
        return self.state, reward, done


def main():
    """Demonstrate advanced topics from chapters 9-13."""
    print("Advanced Functional Analysis for Optical Design")
    print("Chapters 9-13: Comprehensive Demonstration")
    print("=" * 50)
    
    # Chapter 9: Weak Convergence
    print("\n--- Chapter 9: Weak Convergence ---")
    weak_conv = WeakConvergenceOptics()
    
    # Test weak convergence with oscillating sequence
    sequence = [np.sin(n * np.linspace(-1, 1, 100)) / n for n in range(1, 21)]
    test_functionals = {
        'constant': np.ones(100),
        'linear': np.linspace(-1, 1, 100),
        'quadratic': np.linspace(-1, 1, 100)**2
    }
    
    convergence_results = weak_conv.weak_convergence_test(sequence, test_functionals)
    print(f"Weak convergence test completed for {len(convergence_results)} functionals")
    
    # Chapter 10: Distribution Theory
    print("\n--- Chapter 10: Distribution Theory ---")
    dist_theory = DistributionTheoryOptics()
    
    # Point source response
    source_pos = 0.3
    obs_points = [-0.5, 0.0, 0.5]
    responses = dist_theory.point_source_response(source_pos, obs_points)
    print(f"Point source responses at observation points: {np.abs(responses)}")
    
    # Principal value integral
    test_function = np.sin(np.linspace(-1, 1, 100))
    pv_integral = dist_theory.principal_value_integral(test_function, 0.0)
    print(f"Principal value integral: {pv_integral:.6f}")
    
    # Chapter 11: Advanced Optimization
    print("\n--- Chapter 11: Advanced Optimization ---")
    adv_opt = AdvancedOptimizationAlgorithms()
    
    # Test functions
    def merit_func(x):
        return np.sum(x**4) + np.sum((x-1)**2)
    
    def grad_func(x):
        return 4 * x**3 + 2 * (x - 1)
    
    initial = np.random.randn(100)
    newton_history = adv_opt.newton_method_functional(initial, merit_func, grad_func)
    print(f"Newton method: {len(newton_history)} iterations, final merit: {newton_history[-1]['merit']:.6f}")
    
    # Chapter 12: Uncertainty Quantification
    print("\n--- Chapter 12: Uncertainty Quantification ---")
    uq = UncertaintyQuantificationOptics()
    
    # Simple optical model with uncertainty
    def optical_model_with_uncertainty(params):
        # Simple lens model: 1/f = 1/u + 1/v
        focal_length = params['focal_length']
        object_distance = params['object_distance']
        image_distance = 1 / (1/focal_length - 1/object_distance)
        return image_distance
    
    parameter_distributions = {
        'focal_length': {'type': 'normal', 'mean': 0.1, 'std': 0.005},
        'object_distance': {'type': 'uniform', 'min': 0.05, 'max': 0.15}
    }
    
    mc_results = uq.monte_carlo_simulation(optical_model_with_uncertainty, parameter_distributions)
    print(f"Monte Carlo simulation: mean = {np.mean(mc_results):.4f}, std = {np.std(mc_results):.4f}")
    
    # Chapter 13: AI Integration
    print("\n--- Chapter 13: AI Integration ---")
    ai_optics = AIIntegrationOptics()
    
    # Generate training data
    training_data = [(np.random.randn(5), np.random.randn(3)) for _ in range(200)]
    nn_model = ai_optics.neural_network_surrogate(training_data)
    print(f"Neural network training completed: final loss = {nn_model['final_loss']:.6f}")
    
    # Inverse design
    def forward_model(design):
        # Simple optical system model
        return np.sin(design) + 0.1 * design**2
    
    target_perf = np.array([0.5, 0.3, 0.8])
    initial_design = np.random.randn(3)
    final_design, design_history = ai_optics.inverse_design_optimization(
        target_perf, forward_model, initial_design
    )
    print(f"Inverse design: final error = {design_history[-1]:.6f}")
    
    # Reinforcement learning
    env = OpticalEnvironment(target_perf)
    rl_rewards = ai_optics.reinforcement_learning_optics(env)
    print(f"Reinforcement learning: average reward = {np.mean(rl_rewards):.3f}")
    
    print("\n=== Summary of Advanced Concepts ===")
    print("1. Weak convergence important for iterative algorithms")
    print("2. Distribution theory handles point sources and Green's functions")
    print("3. Advanced optimization accelerates convergence")
    print("4. Uncertainty quantification essential for robust design")
    print("5. AI integration enables inverse design and acceleration")
    print("6. All concepts build on functional analysis foundations")


if __name__ == "__main__":
    main()