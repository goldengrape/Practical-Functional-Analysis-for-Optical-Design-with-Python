"""
Chapter 11: Advanced Optimization Algorithms in Optical Design
Functional Analysis for Optical Design

Advanced optimization algorithms including Newton methods, trust region methods,
and multi-objective optimization for optical systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import solve, inv


class AdvancedOptimizationOptics:
    """
    Advanced optimization algorithms for optical design:
    - Newton methods in function spaces
    - Trust region methods
    - Multi-objective optimization
    - Global optimization techniques
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
    
    def newton_method_functional(self, initial_guess, merit_function, gradient_function, 
                                hessian_function=None, num_iterations=20, line_search=True):
        """
        Newton's method in function spaces with optional line search.
        
        xₙ₊₁ = xₙ - [H(xₙ)]⁻¹∇f(xₙ)
        
        For large systems, we use iterative linear solvers instead of direct inversion.
        """
        current = initial_guess.copy()
        history = []
        
        for iteration in range(num_iterations):
            # Compute gradient
            gradient = gradient_function(current)
            
            # Compute or approximate Hessian
            if hessian_function is None:
                # Finite difference approximation (simplified)
                hessian = np.eye(len(current)) * 1e-3
                # Add diagonal regularization
                hessian += np.eye(len(current)) * 1e-6
            else:
                hessian = hessian_function(current)
            
            # Solve Newton system: HΔx = -∇f
            try:
                # Use iterative solver for large systems
                step = -solve(hessian, gradient, assume_a='pos')
            except np.linalg.LinAlgError:
                # Fallback to gradient descent with regularization
                regularization = 1e-6 * np.eye(len(current))
                step = -solve(hessian + regularization, gradient)
            
            # Line search (optional)
            if line_search:
                # Simple backtracking line search
                alpha = 1.0
                current_merit = merit_function(current)
                
                for ls_iter in range(10):
                    trial_point = current + alpha * step
                    trial_merit = merit_function(trial_point)
                    
                    if trial_merit < current_merit:
                        break
                    
                    alpha *= 0.5
                
                step = alpha * step
            
            # Update
            current = current + step
            
            # Store history
            history.append({
                'iteration': iteration,
                'merit': merit_function(current),
                'gradient_norm': np.linalg.norm(gradient),
                'step_size': np.linalg.norm(step),
                'alpha': alpha if line_search else 1.0
            })
            
            # Convergence check
            if np.linalg.norm(gradient) < 1e-8:
                print(f"Newton method converged after {iteration+1} iterations")
                break
        
        return history
    
    def trust_region_method(self, initial_guess, merit_function, gradient_function, 
                           hessian_function=None, trust_radius=0.1, max_iterations=30):
        """
        Trust region method for constrained optimization.
        
        Solves: min m(d) = f + gᵀd + ½dᵀHd subject to ‖d‖ ≤ Δ
        """
        current = initial_guess.copy()
        history = []
        delta = trust_radius
        
        for iteration in range(max_iterations):
            # Compute gradient and Hessian
            gradient = gradient_function(current)
            
            if hessian_function is None:
                # Approximate Hessian
                hessian = np.eye(len(current)) * 1e-3
            else:
                hessian = hessian_function(current)
            
            # Solve trust region subproblem
            def subproblem(direction):
                if np.linalg.norm(direction) > delta:
                    return np.inf
                
                # Quadratic model
                model_value = (merit_function(current) + 
                             np.dot(gradient, direction) + 
                             0.5 * np.dot(direction, hessian @ direction))
                return model_value
            
            # Solve subproblem using dogleg method (simplified)
            # Cauchy point (gradient descent)
            cauchy_direction = -delta * gradient / (np.linalg.norm(gradient) + 1e-10)
            
            # Newton direction
            try:
                newton_direction = -solve(hessian, gradient)
                # Project onto trust region if necessary
                if np.linalg.norm(newton_direction) > delta:
                    newton_direction = delta * newton_direction / np.linalg.norm(newton_direction)
            except:
                newton_direction = cauchy_direction
            
            # Choose direction (simplified dogleg)
            if np.linalg.norm(newton_direction) <= delta:
                step = newton_direction
            else:
                step = cauchy_direction
            
            # Evaluate actual vs predicted reduction
            current_merit = merit_function(current)
            trial_point = current + step
            trial_merit = merit_function(trial_point)
            
            actual_reduction = current_merit - trial_merit
            predicted_reduction = -(np.dot(gradient, step) + 0.5 * np.dot(step, hessian @ step))
            
            # Update trust radius
            if predicted_reduction > 0:
                rho = actual_reduction / predicted_reduction
                if rho > 0.75:
                    delta = min(2 * delta, 1.0)  # Increase trust region
                elif rho < 0.25:
                    delta = max(0.5 * delta, 1e-4)  # Decrease trust region
            
            # Accept or reject step
            if actual_reduction > 0:
                current = trial_point
                step_accepted = True
            else:
                step_accepted = False
            
            history.append({
                'iteration': iteration,
                'merit': current_merit,
                'trust_radius': delta,
                'step_size': np.linalg.norm(step),
                'step_accepted': step_accepted,
                'rho': rho if predicted_reduction > 0 else 0
            })
            
            # Convergence check
            if np.linalg.norm(gradient) < 1e-8:
                print(f"Trust region method converged after {iteration+1} iterations")
                break
        
        return history
    
    def multi_objective_optimization(self, initial_design, objectives, weights, constraints=None):
        """
        Multi-objective optimization for optical systems.
        
        Combines multiple objectives using weighted sum or other methods.
        """
        def combined_objective(design):
            total = 0
            for objective, weight in zip(objectives, weights):
                total += weight * objective(design)
            return total
        
        def combined_gradient(design):
            total_gradient = np.zeros_like(design)
            for objective, weight in zip(objectives, weights):
                # Numerical gradient (simplified)
                epsilon = 1e-6
                gradient = np.zeros_like(design)
                for i in range(len(design)):
                    perturbed = design.copy()
                    perturbed[i] += epsilon
                    gradient[i] = (objective(perturbed) - objective(design)) / epsilon
                total_gradient += weight * gradient
            return total_gradient
        
        # Optimize using different methods
        methods = ['BFGS', 'L-BFGS-B', 'Nelder-Mead']
        results = {}
        
        for method in methods:
            try:
                if method in ['BFGS', 'L-BFGS-B']:
                    result = minimize(combined_objective, initial_design, 
                                    method=method, jac=combined_gradient,
                                    constraints=constraints)
                else:
                    result = minimize(combined_objective, initial_design, method=method)
                
                results[method] = {
                    'success': result.success,
                    'x': result.x,
                    'fun': result.fun,
                    'nit': result.nit if hasattr(result, 'nit') else 0
                }
            except Exception as e:
                results[method] = {'success': False, 'error': str(e)}
        
        return results
    
    def pareto_optimization(self, objectives, bounds, population_size=50, generations=100):
        """
        Pareto optimization for multi-objective problems.
        
        Finds Pareto frontier instead of single weighted solution.
        """
        def multi_objective_function(x):
            return [obj(x) for obj in objectives]
        
        # Use differential evolution for global Pareto optimization
        result = differential_evolution(
            lambda x: np.sum(np.array(multi_objective_function(x))**2),  # Temporary objective
            bounds=bounds,
            popsize=population_size,
            maxiter=generations,
            seed=42
        )
        
        # Generate Pareto points (simplified)
        pareto_points = []
        for i in range(20):  # Sample 20 points
            # Random weights
            weights = np.random.dirichlet(np.ones(len(objectives)))
            
            def weighted_objective(x):
                obj_values = multi_objective_function(x)
                return np.sum(weights * obj_values)
            
            try:
                point_result = minimize(weighted_objective, result.x, method='BFGS')
                if point_result.success:
                    obj_values = multi_objective_function(point_result.x)
                    pareto_points.append({
                        'design': point_result.x,
                        'objectives': obj_values,
                        'weights': weights
                    })
            except:
                continue
        
        return {
            'pareto_points': pareto_points,
            'num_points': len(pareto_points),
            'representative_solution': result.x
        }
    
    def global_optimization_demo(self, objective_function, bounds, methods=['differential_evolution', 'basinhopping']):
        """
        Demonstrate global optimization techniques for optical design.
        """
        results = {}
        
        # Differential Evolution
        if 'differential_evolution' in methods:
            try:
                from scipy.optimize import differential_evolution
                result_de = differential_evolution(
                    objective_function, 
                    bounds=bounds,
                    popsize=30,
                    maxiter=200,
                    seed=42
                )
                results['differential_evolution'] = {
                    'success': result_de.success,
                    'x': result_de.x,
                    'fun': result_de.fun,
                    'nit': result_de.nit
                }
            except Exception as e:
                results['differential_evolution'] = {'error': str(e)}
        
        # Basin Hopping
        if 'basinhopping' in methods:
            try:
                from scipy.optimize import basinhopping
                
                # Need to convert bounds for basinhopping
                def bounds_wrapper(x):
                    # Penalty for violating bounds
                    penalty = 0
                    for i, (xi, bound) in enumerate(zip(x, bounds)):
                        if xi < bound[0] or xi > bound[1]:
                            penalty += 1e6 * (abs(xi - bound[0]) + abs(xi - bound[1]))
                    return objective_function(x) + penalty
                
                result_bh = basinhopping(bounds_wrapper, np.mean(bounds, axis=1), 
                                       niter=50, seed=42)
                results['basinhopping'] = {
                    'success': True,
                    'x': result_bh.x,
                    'fun': result_bh.fun,
                    'nit': result_bh.nit
                }
            except Exception as e:
                results['basinhopping'] = {'error': str(e)}
        
        return results


def demonstrate_advanced_optimization():
    """Demonstrate advanced optimization algorithms in optical design."""
    print("Advanced Optimization Algorithms in Optical Design")
    print("=" * 52)
    
    # Initialize optimizer
    optimizer = AdvancedOptimizationOptics(grid_size=50)
    
    # Test 1: Newton Method vs Gradient Descent
    print(f"\n--- Test 1: Newton Method Comparison ---")
    
    # Rosenbrock-like function (common in optical optimization)
    def rosenbrock_objective(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rosenbrock_gradient(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad
    
    def rosenbrock_hessian(x):
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n-1):
            hessian[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
            hessian[i, i+1] = -400 * x[i]
            hessian[i+1, i] = -400 * x[i]
            hessian[i+1, i+1] = 200
        
        return hessian
    
    # Initial guess
    initial = np.random.randn(20) * 0.1
    
    # Newton method with Hessian
    newton_history = optimizer.newton_method_functional(
        initial, rosenbrock_objective, rosenbrock_gradient, 
        rosenbrock_hessian, num_iterations=50
    )
    
    # Newton method without Hessian (approximated)
    newton_approx_history = optimizer.newton_method_functional(
        initial, rosenbrock_objective, rosenbrock_gradient,
        hessian_function=None, num_iterations=50
    )
    
    print(f"Newton with Hessian: {len(newton_history)} iterations, final merit: {newton_history[-1]['merit']:.6f}")
    print(f"Newton approximated: {len(newton_approx_history)} iterations, final merit: {newton_approx_history[-1]['merit']:.6f}")
    
    # Test 2: Trust Region Method
    print(f"\n--- Test 2: Trust Region Method ---")
    
    trust_history = optimizer.trust_region_method(
        initial, rosenbrock_objective, rosenbrock_gradient,
        trust_radius=0.5, max_iterations=50
    )
    
    print(f"Trust region: {len(trust_history)} iterations, final merit: {trust_history[-1]['merit']:.6f}")
    print(f"Final trust radius: {trust_history[-1]['trust_radius']:.4f}")
    
    # Test 3: Multi-objective Optimization
    print(f"\n--- Test 3: Multi-objective Optimization ---")
    
    # Optical design objectives
    def wavefront_error_objective(design):
        """Minimize wavefront error (RMS)."""
        return np.sqrt(np.mean(design**2))
    
    def smoothness_objective(design):
        """Maximize smoothness (minimize curvature)."""
        curvature = np.gradient(np.gradient(design))
        return np.mean(curvature**2)
    
    def manufacturing_objective(design):
        """Minimize manufacturing complexity."""
        gradient = np.gradient(design)
        return np.mean(np.abs(gradient))
    
    objectives = [wavefront_error_objective, smoothness_objective, manufacturing_objective]
    weights = [0.5, 0.3, 0.2]  # Different priorities
    
    multi_results = optimizer.multi_objective_optimization(
        initial, objectives, weights
    )
    
    print("Multi-objective optimization results:")
    for method, result in multi_results.items():
        if result['success']:
            print(f"  {method}: success = True, iterations = {result['nit']}, final value = {result['fun']:.6f}")
        else:
            print(f"  {method}: success = False")
    
    # Test 4: Pareto Optimization
    print(f"\n--- Test 4: Pareto Optimization ---")
    
    # Simple bounds for design parameters
    bounds = [(-1, 1)] * 20
    
    pareto_result = optimizer.pareto_optimization(
        [wavefront_error_objective, smoothness_objective],
        bounds, population_size=30, generations=50
    )
    
    print(f"Pareto optimization: {pareto_result['num_points']} Pareto points found")
    
    # Test 5: Global Optimization
    print(f"\n--- Test 5: Global Optimization ---")
    
    # Multi-modal objective function (common in optical design)
    def multimodal_objective(x):
        """Multi-modal function with many local minima."""
        return (np.sin(5 * x[0]) * np.cos(5 * x[1]) + 
                0.1 * np.sum(x**2) + 
                0.5 * np.sin(10 * x[0]) * np.sin(10 * x[1]))
    
    # 2D bounds for visualization
    bounds_2d = [(-2, 2), (-2, 2)]
    initial_2d = np.array([0.5, 0.5])
    
    global_results = optimizer.global_optimization_demo(
        multimodal_objective, bounds_2d, methods=['differential_evolution']
    )
    
    print("Global optimization results:")
    for method, result in global_results.items():
        if 'error' not in result:
            print(f"  {method}: success = {result['success']}, final value = {result['fun']:.6f}")
        else:
            print(f"  {method}: error = {result['error'][:30]}...")
    
    print(f"\n=== Key Concepts ===")
    print("1. Newton methods use second-order information for faster convergence")
    print("2. Trust region methods balance local model accuracy and step size")
    print("3. Multi-objective optimization handles competing design criteria")
    print("4. Pareto optimization finds trade-off solutions rather than single optimum")
    print("5. Global optimization techniques avoid local minima in complex landscapes")
    print("6. Advanced algorithms are essential for complex optical design problems")


if __name__ == "__main__":
    demonstrate_advanced_optimization()