"""
Chapter 2: Calculus of Variations - Shortest Path vs Brachistochrone
Practice Project: Classical Variational Problems in Optics Context
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad, odeint
import sympy as sp


class VariationalProblems:
    """Solve classical variational problems: shortest path and brachistochrone"""
    
    def __init__(self):
        self.setup_parameters()
    
    def setup_parameters(self):
        """Set up problem parameters"""
        self.g = 9.81  # gravitational acceleration (m/s^2)
        self.n_points = 100  # number of points for discretization
    
    def shortest_path_problem(self, x_start=0, y_start=0, x_end=1, y_end=1):
        """Solve the shortest path problem (straight line)"""
        print("=== Shortest Path Problem ===")
        
        # Analytical solution: straight line
        def straight_line(x):
            m = (y_end - y_start) / (x_end - x_start)
            return y_start + m * (x - x_start)
        
        # Arc length functional
        def arc_length_functional(y_func, x_range):
            """Compute arc length of a function"""
            def integrand(x):
                # Numerical derivative
                h = 1e-8
                dy_dx = (y_func(x + h) - y_func(x - h)) / (2 * h)
                return np.sqrt(1 + dy_dx**2)
            
            length, _ = quad(integrand, x_range[0], x_range[1])
            return length
        
        # Verify straight line is shortest
        x_range = [x_start, x_end]
        straight_length = arc_length_functional(straight_line, x_range)
        
        print(f"Straight line length: {straight_length:.6f}")
        
        # Compare with other paths
        def parabolic_path(x):
            return y_start + (y_end - y_start) * ((x - x_start) / (x_end - x_start))**2
        
        def sinusoidal_path(x):
            return y_start + (y_end - y_start) * (x - x_start) / (x_end - x_start) + \
                   0.1 * np.sin(2 * np.pi * (x - x_start) / (x_end - x_start))
        
        parabolic_length = arc_length_functional(parabolic_path, x_range)
        sinusoidal_length = arc_length_functional(sinusoidal_path, x_range)
        
        print(f"Parabolic path length: {parabolic_length:.6f}")
        print(f"Sinusoidal path length: {sinusoidal_length:.6f}")
        
        # Visualize paths
        self.visualize_shortest_paths(x_range, straight_line, parabolic_path, sinusoidal_path)
        
        return {
            'straight': (straight_line, straight_length),
            'parabolic': (parabolic_path, parabolic_length),
            'sinusoidal': (sinusoidal_path, sinusoidal_length)
        }
    
    def visualize_shortest_paths(self, x_range, *paths):
        """Visualize different paths"""
        x = np.linspace(x_range[0], x_range[1], 100)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange']
        labels = ['Straight Line', 'Parabolic', 'Sinusoidal', 'Optimized']
        
        for i, (path_func, color, label) in enumerate(zip(paths, colors, labels)):
            y = path_func(x)
            plt.plot(x, y, color=color, linewidth=2, label=label)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Shortest Path Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    def brachistochrone_problem(self, x_start=0, y_start=0, x_end=1, y_end=-0.5):
        """Solve the brachistochrone problem (curve of fastest descent)"""
        print("\n=== Brachistochrone Problem ===")
        
        # Analytical solution: cycloid
        def cycloid_solution(x, a=None):
            """Parametric cycloid solution"""
            if a is None:
                # Find optimal parameter a
                a = self.find_optimal_cycloid_parameter(x_start, y_start, x_end, y_end)
            
            # Parametric equations for cycloid
            t = np.arccos(1 - (x - x_start) / a)
            y = y_start - a * (t - np.sin(t))
            return y
        
        # Numerical optimization approach
        def time_functional(y_values, x_values):
            """Compute descent time for a given path"""
            # Interpolate path
            from scipy.interpolate import interp1d
            y_func = interp1d(x_values, y_values, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # Compute descent time
            def time_integrand(x):
                h = 1e-6
                dy_dx = (y_func(x + h) - y_func(x - h)) / (2 * h)
                y_val = y_func(x)
                # Time = ∫√(1 + (dy/dx)²)/√(2g(y_start - y)) dx
                return np.sqrt((1 + dy_dx**2) / (2 * self.g * (y_start - y_val)))
            
            # Integrate over the path
            time, _ = quad(time_integrand, x_values[0], x_values[-1])
            return time
        
        # Find optimal cycloid parameter
        optimal_a = self.find_optimal_cycloid_parameter(x_start, y_start, x_end, y_end)
        
        # Create comparison paths
        x_range = np.linspace(x_start, x_end, 100)
        
        # Cycloid (optimal)
        cycloid_y = np.array([cycloid_solution(x, optimal_a) for x in x_range])
        cycloid_time = time_functional(cycloid_y, x_range)
        
        # Straight line
        straight_y = y_start + (y_end - y_start) * (x_range - x_start) / (x_end - x_start)
        straight_time = time_functional(straight_y, x_range)
        
        # Parabolic path
        parabolic_y = y_start + (y_end - y_start) * ((x_range - x_start) / (x_end - x_start))**2
        parabolic_time = time_functional(parabolic_y, x_range)
        
        print(f"Cycloid descent time: {cycloid_time:.6f} seconds")
        print(f"Straight line time: {straight_time:.6f} seconds")
        print(f"Parabolic path time: {parabolic_time:.6f} seconds")
        
        # Visualize results
        self.visualize_brachistochrone_paths(x_range, cycloid_y, straight_y, parabolic_y, 
                                            [cycloid_time, straight_time, parabolic_time])
        
        return {
            'cycloid': (cycloid_y, cycloid_time),
            'straight': (straight_y, straight_time),
            'parabolic': (parabolic_y, parabolic_time),
            'optimal_parameter': optimal_a
        }
    
    def find_optimal_cycloid_parameter(self, x_start, y_start, x_end, y_end):
        """Find optimal cycloid parameter for given endpoints"""
        def endpoint_error(a):
            # Find t value at x_end
            t_end = np.arccos(1 - (x_end - x_start) / a)
            y_calculated = y_start - a * (t_end - np.sin(t_end))
            return (y_calculated - y_end)**2
        
        # Minimize endpoint error
        result = minimize_scalar(endpoint_error, bounds=(0.01, 10), method='bounded')
        return result.x
    
    def visualize_brachistochrone_paths(self, x_range, cycloid_y, straight_y, parabolic_y, times):
        """Visualize brachistochrone paths"""
        plt.figure(figsize=(12, 8))
        
        # Plot paths
        plt.plot(x_range, cycloid_y, 'b-', linewidth=3, label=f'Cycloid (optimal) - {times[0]:.4f}s')
        plt.plot(x_range, straight_y, 'r--', linewidth=2, label=f'Straight line - {times[1]:.4f}s')
        plt.plot(x_range, parabolic_y, 'g:', linewidth=2, label=f'Parabolic - {times[2]:.4f}s')
        
        # Add start and end points
        plt.plot(x_range[0], cycloid_y[0], 'ko', markersize=8, label='Start')
        plt.plot(x_range[-1], cycloid_y[-1], 'ko', markersize=8, label='End')
        
        plt.xlabel('Horizontal Distance (m)')
        plt.ylabel('Vertical Distance (m)')
        plt.title('Brachistochrone Problem: Fastest Descent Paths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to show descent
        plt.show()
    
    def derive_euler_lagrange_equation(self):
        """Derive Euler-Lagrange equation for brachistochrone problem"""
        print("\n=== Euler-Lagrange Equation Derivation ===")
        
        # Symbolic derivation using SymPy
        x, y, dy_dx = sp.symbols('x y dy_dx', real=True)
        g = sp.symbols('g', real=True, positive=True)
        
        # Lagrangian for brachistochrone: L = sqrt((1 + (dy/dx)^2)/(2g(y_start - y)))
        L = sp.sqrt((1 + dy_dx**2) / (2 * g * (0 - y)))  # y_start = 0
        
        print("Lagrangian for brachistochrone problem:")
        print(f"L = {L}")
        
        # Euler-Lagrange equation: d/dx(∂L/∂y') - ∂L/∂y = 0
        dL_dy = sp.diff(L, y)
        dL_dydx = sp.diff(L, dy_dx)
        
        print(f"\n∂L/∂y = {dL_dy}")
        print(f"∂L/∂y' = {dL_dydx}")
        
        # For the complete derivation, we would need to compute d/dx(∂L/∂y')
        print("\nEuler-Lagrange equation leads to the cycloid solution.")
        print("This demonstrates how variational principles yield optimal paths.")
        
        return L, dL_dy, dL_dydx
    
    def numerical_optimization_approach(self):
        """Solve variational problems using numerical optimization"""
        print("\n=== Numerical Optimization Approach ===")
        
        # Discretize the problem
        n_points = 50
        x_points = np.linspace(0, 1, n_points)
        
        def total_time_function(y_points):
            """Total time for a given discrete path"""
            # Use trapezoidal integration
            total_time = 0
            for i in range(1, len(x_points)):
                dx = x_points[i] - x_points[i-1]
                dy = y_points[i] - y_points[i-1]
                y_avg = (y_points[i] + y_points[i-1]) / 2
                
                # Time increment for this segment
                if y_avg < 0:  # Ensure positive height difference
                    dt = np.sqrt((dx**2 + dy**2) / (2 * self.g * abs(y_avg)))
                    total_time += dt
            
            return total_time
        
        # Initial guess: straight line
        y_initial = -0.5 * x_points  # Straight line from (0,0) to (1,-0.5)
        
        # Optimize the path
        from scipy.optimize import minimize
        
        # Constraints: fix endpoints
        constraints = [
            {'type': 'eq', 'fun': lambda y: y[0]},  # y(0) = 0
            {'type': 'eq', 'fun': lambda y: y[-1] + 0.5}  # y(1) = -0.5
        ]
        
        # Bounds for intermediate points
        bounds = [(None, 0.1)] * n_points  # y <= 0.1 (slightly above start)
        bounds[0] = (0, 0)  # Fix start point
        bounds[-1] = (-0.5, -0.5)  # Fix end point
        
        # Optimize
        result = minimize(total_time_function, y_initial, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_y = result.x
        optimal_time = result.fun
        
        print(f"Numerical optimization result:")
        print(f"Optimal time: {optimal_time:.6f} seconds")
        print(f"Number of iterations: {result.nit}")
        
        # Compare with analytical solution
        cycloid_time = 0.5 * np.pi * np.sqrt(0.5 / self.g)  # Approximate analytical result
        print(f"Analytical cycloid time: {cycloid_time:.6f} seconds")
        print(f"Numerical accuracy: {abs(optimal_time - cycloid_time) / cycloid_time * 100:.2f}%")
        
        # Visualize the optimized path
        plt.figure(figsize=(10, 6))
        plt.plot(x_points, optimal_y, 'b-', linewidth=2, label='Numerical Optimal')
        plt.plot(x_points, y_initial, 'r--', linewidth=2, label='Initial Guess')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Numerical Optimization Result')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        plt.show()
        
        return optimal_y, optimal_time


def main():
    """Main function to demonstrate calculus of variations"""
    print("Chapter 2: Calculus of Variations - Shortest Path vs Brachistochrone")
    print("=" * 75)
    
    # Initialize the variational problems class
    variational = VariationalProblems()
    
    # Solve shortest path problem
    shortest_results = variational.shortest_path_problem()
    
    # Solve brachistochrone problem
    brachistochrone_results = variational.brachistochrone_problem()
    
    # Derive Euler-Lagrange equation
    L, dL_dy, dL_dydx = variational.derive_euler_lagrange_equation()
    
    # Numerical optimization approach
    optimal_path, optimal_time = variational.numerical_optimization_approach()
    
    print("\n=== Key Insights ===")
    print("1. Shortest path is always a straight line (Euclidean geometry)")
    print("2. Brachistochrone is a cycloid (optimal for time minimization)")
    print("3. Euler-Lagrange equation provides analytical solution method")
    print("4. Numerical optimization can solve complex variational problems")
    
    print("\n=== Applications to Optics ===")
    print("1. Fermat's principle: light follows path of least time")
    print("2. Lens design: minimize aberrations using variational principles")
    print("3. Ray tracing: apply calculus of variations to optical systems")
    print("4. Wavefront optimization: use functional minimization")
    
    print("\n=== Practice Exercises ===")
    print("1. Solve the catenary problem (hanging chain)")
    print("2. Apply Fermat's principle to derive Snell's law")
    print("3. Extend to 3D variational problems")
    print("4. Implement constrained optimization for optical systems")


if __name__ == "__main__":
    main()