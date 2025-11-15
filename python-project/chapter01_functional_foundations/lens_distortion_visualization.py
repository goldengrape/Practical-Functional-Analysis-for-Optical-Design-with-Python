"""
Chapter 1: Functional Foundations - Spherical Lens Edge Distortion
Practice Project: Visualizing Continuous vs Discrete Thinking in Optics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize


class LensDistortionAnalyzer:
    """Analyze edge distortion in spherical lenses using functional approach"""
    
    def __init__(self):
        self.setup_parameters()
    
    def setup_parameters(self):
        """Set up lens parameters"""
        self.lens_radius = 25.0  # mm
        self.lens_thickness = 2.0  # mm
        self.refractive_index = 1.5
        self.wavelength = 0.55  # μm (green light)
    
    def spherical_surface(self, x, y, radius_of_curvature):
        """Define spherical surface sag"""
        r_squared = x**2 + y**2
        sag = r_squared / (2 * radius_of_curvature)
        return sag
    
    def ray_trace_through_lens(self, ray_position, ray_angle, surface_sag_func):
        """Simple ray tracing through a lens surface"""
        # This is a simplified ray tracing model
        x, y = ray_position
        
        # Find intersection with lens surface
        def height_error(z):
            surface_height = surface_sag_func(x, y)
            return z - surface_height
        
        # Solve for intersection
        from scipy.optimize import fsolve
        z_intersect = fsolve(height_error, 0)[0]
        
        # Apply Snell's law (simplified)
        surface_normal = self.compute_surface_normal(x, y, surface_sag_func)
        
        # Compute transmitted ray direction
        transmitted_angle = self.apply_snells_law(ray_angle, surface_normal)
        
        return z_intersect, transmitted_angle
    
    def compute_surface_normal(self, x, y, surface_sag_func):
        """Compute surface normal at point (x, y)"""
        # Numerical gradient
        h = 1e-6
        dz_dx = (surface_sag_func(x + h, y) - surface_sag_func(x - h, y)) / (2 * h)
        dz_dy = (surface_sag_func(x, y + h) - surface_sag_func(x, y - h)) / (2 * h)
        
        # Normal vector (pointing in -z direction for convex surface)
        normal = np.array([-dz_dx, -dz_dy, 1])
        return normal / np.linalg.norm(normal)
    
    def apply_snells_law(self, incident_angle, normal, n1=1.0, n2=1.5):
        """Apply Snell's law for ray refraction"""
        # Simplified 2D version
        cos_theta1 = np.cos(incident_angle)
        sin_theta1 = np.sin(incident_angle)
        
        # Snell's law: n1*sin(theta1) = n2*sin(theta2)
        sin_theta2 = (n1 / n2) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2)
        
        return np.arctan2(sin_theta2, cos_theta2)
    
    def compute_optical_path_difference(self, ray_positions, surface_func):
        """Compute optical path difference for a set of rays"""
        opd_values = []
        
        for pos in ray_positions:
            x, y = pos
            
            # Compute path through lens
            sag = surface_func(x, y)
            
            # Optical path difference (simplified)
            opd = sag * (self.refractive_index - 1)
            opd_values.append(opd)
        
        return np.array(opd_values)
    
    def functional_approach_vs_discrete(self):
        """Compare functional approach vs discrete parameter approach"""
        print("=== Functional vs Discrete Approach Comparison ===")
        
        # Discrete approach: optimize a few parameters
        def discrete_merit_function(params):
            r1, r2, thickness = params
            
            # Create surface functions
            def front_surface(x, y):
                return self.spherical_surface(x, y, r1)
            
            def back_surface(x, y):
                return self.spherical_surface(x, y, r2) + thickness
            
            # Sample rays
            ray_positions = np.array([
                [0, 0], [5, 0], [10, 0], [15, 0], [20, 0]
            ])
            
            # Compute aberrations
            opd_front = self.compute_optical_path_difference(ray_positions, front_surface)
            opd_back = self.compute_optical_path_difference(ray_positions, back_surface)
            
            # Merit function: minimize variance of OPD
            total_opd = opd_front + opd_back
            merit = np.var(total_opd)
            
            return merit
        
        # Functional approach: optimize entire surface
        def functional_merit_function(surface_coefficients):
            # Create surface function from coefficients
            def surface_func(x, y):
                # Polynomial expansion
                z = 0
                for i, coeff in enumerate(surface_coefficients):
                    # Even powers of r for rotational symmetry
                    power = 2 * (i + 1)
                    r = np.sqrt(x**2 + y**2)
                    z += coeff * (r**power)
                return z
            
            # Dense sampling of rays
            n_rays = 50
            angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
            ray_positions = 20 * np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Compute aberrations
            opd = self.compute_optical_path_difference(ray_positions, surface_func)
            
            # Merit function
            merit = np.var(opd) + 0.1 * np.max(np.abs(opd))
            
            return merit
        
        # Optimize discrete approach
        print("Optimizing discrete parameters...")
        initial_discrete = [25.0, -25.0, 2.0]  # r1, r2, thickness
        result_discrete = optimize.minimize(discrete_merit_function, initial_discrete)
        
        print(f"Discrete optimization result:")
        print(f"  Optimal parameters: r1={result_discrete.x[0]:.2f}, r2={result_discrete.x[1]:.2f}, t={result_discrete.x[2]:.2f}")
        print(f"  Final merit: {result_discrete.fun:.6f}")
        
        # Optimize functional approach
        print("\nOptimizing functional approach...")
        initial_functional = [0.02, -0.001, 0.0001]  # polynomial coefficients
        result_functional = optimize.minimize(functional_merit_function, initial_functional)
        
        print(f"Functional optimization result:")
        print(f"  Optimal coefficients: {result_functional.x}")
        print(f"  Final merit: {result_functional.fun:.6f}")
        
        return result_discrete, result_functional
    
    def visualize_lens_distortion(self):
        """Visualize lens edge distortion"""
        print("\n=== Lens Edge Distortion Visualization ===")
        
        # Create coordinate system
        N = 100
        x = np.linspace(-30, 30, N)
        y = np.linspace(-30, 30, N)
        X, Y = np.meshgrid(x, y)
        
        # Create different lens surfaces
        surfaces = {
            'Spherical (R=25mm)': self.spherical_surface(X, Y, 25.0),
            'Spherical (R=50mm)': self.spherical_surface(X, Y, 50.0),
            'Optimized Functional': self.optimized_functional_surface(X, Y),
            'Discrete Optimized': self.discrete_optimized_surface(X, Y)
        }
        
        # Create 3D visualization
        fig = plt.figure(figsize=(16, 12))
        
        for i, (name, surface) in enumerate(surfaces.items(), 1):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            
            # Plot surface
            surf = ax.plot_surface(X, Y, surface, cmap='viridis', 
                                 alpha=0.8, linewidth=0, antialiased=True)
            
            # Add lens aperture boundary
            theta = np.linspace(0, 2*np.pi, 100)
            x_aperture = 25 * np.cos(theta)
            y_aperture = 25 * np.sin(theta)
            z_aperture = np.interp(np.sqrt(x_aperture**2 + y_aperture**2), 
                                 np.sqrt(X[0,:]**2 + Y[:,0]**2), 
                                 surface[N//2, :])
            
            ax.plot(x_aperture, y_aperture, z_aperture, 'r-', linewidth=3)
            
            # Customize plot
            ax.set_title(f'{name}')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Sag (mm)')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            
            # Print surface statistics
            aperture_mask = np.sqrt(X**2 + Y**2) <= 25
            rms_error = np.std(surface[aperture_mask])
            print(f"{name}:")
            print(f"  RMS deviation: {rms_error:.3f} mm")
        
        plt.tight_layout()
        plt.show()
        
        return surfaces
    
    def optimized_functional_surface(self, X, Y):
        """Create optimized functional surface"""
        # Use coefficients from optimization (example)
        coeffs = [0.02, -0.0008, 0.00002]
        
        R = np.sqrt(X**2 + Y**2)
        surface = 0
        for i, coeff in enumerate(coeffs):
            power = 2 * (i + 1)
            surface += coeff * (R**power)
        
        return surface
    
    def discrete_optimized_surface(self, X, Y):
        """Create discrete optimized surface"""
        # Simple spherical surface with optimized parameters
        R = 30.0  # Optimized radius
        return self.spherical_surface(X, Y, R)
    
    def demonstrate_continuous_thinking(self):
        """Demonstrate continuous thinking in optical design"""
        print("\n=== Continuous Thinking Demonstration ===")
        
        # Create a set of field points
        field_angles = np.linspace(0, 20, 21)  # degrees
        
        # Discrete approach: optimize each field point separately
        discrete_performance = []
        for angle in field_angles:
            # Optimize for this specific angle
            def angle_merit(params):
                # Simple merit function
                return (params[0] - angle)**2 + 0.1 * angle**2
            
            result = optimize.minimize(angle_merit, [angle])
            discrete_performance.append(result.fun)
        
        # Functional approach: optimize entire field simultaneously
        def functional_field_merit(params):
            # Polynomial coefficients for field-dependent optimization
            a, b, c = params
            total_merit = 0
            
            for angle in field_angles:
                # Predicted performance across field
                predicted_performance = a + b*angle + c*angle**2
                # Merit includes deviation from ideal and field variation
                total_merit += (predicted_performance - 0)**2 + 0.01 * angle**2
            
            return total_merit
        
        # Optimize functional approach
        result_functional = optimize.minimize(functional_field_merit, [0, 0, 0])
        a_opt, b_opt, c_opt = result_functional.x
        
        # Compare results
        functional_performance = []
        for angle in field_angles:
            functional_performance.append(a_opt + b_opt*angle + c_opt*angle**2)
        
        # Visualize comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(field_angles, discrete_performance, 'bo-', label='Discrete Approach')
        plt.plot(field_angles, functional_performance, 'r-', label='Functional Approach')
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Performance Metric')
        plt.title('Discrete vs Functional Approach Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(field_angles, np.array(discrete_performance) - np.array(functional_performance), 'g-')
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Performance Difference')
        plt.title('Performance Improvement with Functional Approach')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Functional approach improvement: {np.mean(discrete_performance) - np.mean(functional_performance):.4f}")


def main():
    """Main function to demonstrate functional foundations"""
    print("Chapter 1: Functional Foundations - Lens Edge Distortion Analysis")
    print("=" * 70)
    
    # Initialize the analyzer
    analyzer = LensDistortionAnalyzer()
    
    # Run demonstrations
    result_discrete, result_functional = analyzer.functional_approach_vs_discrete()
    surfaces = analyzer.visualize_lens_distortion()
    analyzer.demonstrate_continuous_thinking()
    
    print("\n=== Key Insights ===")
    print("1. Functional approach allows optimization of entire surfaces")
    print("2. Continuous thinking handles edge effects better than discrete sampling")
    print("3. Functional optimization can achieve better performance with fewer parameters")
    print("4. Ray tracing demonstrates the power of function-based representations")
    
    print("\n=== Practice Exercises ===")
    print("1. Implement aspheric surface functions")
    print("2. Add chromatic aberration to the ray tracing model")
    print("3. Compare computational efficiency of different approaches")
    print("4. Extend the model to include multiple lens elements")


if __name__ == "__main__":
    main()