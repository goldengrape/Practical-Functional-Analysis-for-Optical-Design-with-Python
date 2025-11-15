"""
Chapter 2: Calculus of Variations - Fermat Principle and Snell's Law
Practice Project: Deriving Optical Laws from Variational Principles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad
import sympy as sp


class FermatPrincipleOptics:
    """Derive optical laws from Fermat's principle of least time"""
    
    def __init__(self):
        self.setup_constants()
    
    def setup_constants(self):
        """Set up physical constants"""
        self.c = 3e8  # speed of light in vacuum (m/s)
        self.n_air = 1.0
        self.n_glass = 1.5
        self.n_water = 1.33
    
    def fermat_principle_refraction(self):
        """Derive Snell's law from Fermat's principle"""
        print("=== Fermat's Principle: Refraction ===")
        
        # Setup: light ray from point A to point B through interface
        # A = (0, h1), B = (d, -h2), interface at y = 0
        
        h1 = 1.0  # height above interface
        h2 = 1.0  # height below interface  
        d = 2.0   # horizontal distance
        
        def total_time(x_intersect):
            """Total time for light ray intersecting interface at x_intersect"""
            # Distance in medium 1 (air)
            d1 = np.sqrt(x_intersect**2 + h1**2)
            
            # Distance in medium 2 (glass)
            d2 = np.sqrt((d - x_intersect)**2 + h2**2)
            
            # Time = distance / speed
            t1 = d1 / (self.c / self.n_air)
            t2 = d2 / (self.c / self.n_glass)
            
            return t1 + t2
        
        # Find optimal intersection point
        result = minimize_scalar(total_time, bounds=(0, d), method='bounded')
        x_opt = result.x
        
        # Compute angles
        theta1 = np.arctan(x_opt / h1)  # angle of incidence
        theta2 = np.arctan((d - x_opt) / h2)  # angle of refraction
        
        print(f"Optimal intersection point: x = {x_opt:.4f}")
        print(f"Angle of incidence: θ₁ = {np.degrees(theta1):.2f}°")
        print(f"Angle of refraction: θ₂ = {np.degrees(theta2):.2f}°")
        
        # Verify Snell's law
        n1_sin_theta1 = self.n_air * np.sin(theta1)
        n2_sin_theta2 = self.n_glass * np.sin(theta2)
        
        print(f"n₁ sin(θ₁) = {n1_sin_theta1:.6f}")
        print(f"n₂ sin(θ₂) = {n2_sin_theta2:.6f}")
        print(f"Snell's law satisfied: {abs(n1_sin_theta1 - n2_sin_theta2) < 1e-6}")
        
        # Visualize the refraction
        self.visualize_refraction(h1, h2, d, x_opt, theta1, theta2)
        
        return x_opt, theta1, theta2
    
    def visualize_refraction(self, h1, h2, d, x_opt, theta1, theta2):
        """Visualize light refraction at interface"""
        plt.figure(figsize=(12, 8))
        
        # Setup coordinates
        x_air = np.linspace(0, x_opt, 50)
        y_air = h1 - (h1 / x_opt) * x_air
        
        x_glass = np.linspace(x_opt, d, 50)
        y_glass = -(h2 / (d - x_opt)) * (x_glass - x_opt)
        
        # Plot ray paths
        plt.plot(x_air, y_air, 'r-', linewidth=3, label='Incident ray')
        plt.plot(x_glass, y_glass, 'b-', linewidth=3, label='Refracted ray')
        
        # Plot interface
        plt.axhline(y=0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        
        # Add normal line at intersection
        plt.plot([x_opt, x_opt], [h1, -h2], 'k--', alpha=0.5, label='Normal')
        
        # Add angle arcs
        arc1 = plt.Circle((x_opt, 0), 0.2, theta1=90, theta2=90-np.degrees(theta1), 
                         color='red', fill=False, linewidth=2)
        arc2 = plt.Circle((x_opt, 0), 0.3, theta1=270, theta2=270+np.degrees(theta2), 
                         color='blue', fill=False, linewidth=2)
        
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        
        # Labels
        plt.text(x_opt - 0.3, 0.4, f'θ₁ = {np.degrees(theta1):.1f}°', 
                fontsize=12, color='red', ha='center')
        plt.text(x_opt + 0.3, -0.4, f'θ₂ = {np.degrees(theta2):.1f}°', 
                fontsize=12, color='blue', ha='center')
        
        # Medium labels
        plt.text(0.2, h1/2, f'Air (n = {self.n_air})', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.text(0.2, -h2/2, f'Glass (n = {self.n_glass})', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Light Refraction at Interface (Fermat\'s Principle)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def lens_optimization_problem(self):
        """Optimize lens shape using Fermat's principle"""
        print("\n=== Lens Shape Optimization ===")
        
        # Thin lens with variable curvature
        # Goal: focus parallel rays to a single point
        
        focal_length = 10.0  # desired focal length
        lens_aperture = 5.0  # lens diameter
        n_lens = self.n_glass
        
        def lens_surface(x, curvature_params):
            """Define lens surface with variable curvature"""
            # Polynomial curvature: z = a*x² + b*x⁴ + c*x⁶
            a, b, c = curvature_params
            return a * x**2 + b * x**4 + c * x**6
        
        def ray_trace_time(x_ray, curvature_params):
            """Compute time for ray to reach focal point"""
            # Parallel ray from infinity
            y_ray = x_ray  # ray height
            
            # Intersection with lens surface
            z_lens = lens_surface(x_ray, curvature_params)
            
            # Apply refraction at lens surface
            # Simplified: compute angle after refraction
            h = 1e-6
            dz_dx = (lens_surface(x_ray + h, curvature_params) - 
                    lens_surface(x_ray - h, curvature_params)) / (2 * h)
            
            # Surface normal
            normal_angle = np.arctan(dz_dx)
            
            # Incident angle (parallel ray)
            incident_angle = 0
            
            # Refracted angle (Snell's law)
            sin_refracted = np.sin(incident_angle - normal_angle) / n_lens
            refracted_angle = np.arcsin(sin_refracted) + normal_angle
            
            # Path to focal point
            focal_point = np.array([0, 0, focal_length])
            ray_direction = np.array([np.sin(refracted_angle), 0, np.cos(refracted_angle)])
            
            # Intersection with focal plane
            # Simplified: assume ray reaches focal plane
            path_length = focal_length / np.cos(refracted_angle)
            
            # Total time
            time_lens = np.sqrt(x_ray**2 + z_lens**2) / (self.c / n_lens)
            time_air = path_length / self.c
            
            return time_lens + time_air
        
        def lens_merit_function(curvature_params):
            """Merit function for lens optimization"""
            # Sample rays across aperture
            ray_heights = np.linspace(-lens_aperture/2, lens_aperture/2, 10)
            total_time = 0
            
            for y_ray in ray_heights:
                total_time += ray_trace_time(y_ray, curvature_params)
            
            return total_time
        
        # Optimize lens curvature
        initial_params = [0.01, 0, 0]  # Start with simple spherical
        result = minimize(lens_merit_function, initial_params, method='BFGS')
        
        optimal_params = result.x
        print(f"Optimal curvature parameters: a={optimal_params[0]:.6f}, b={optimal_params[1]:.6f}, c={optimal_params[2]:.6f}")
        print(f"Optimization success: {result.success}")
        
        # Visualize optimized lens
        self.visualize_optimized_lens(lens_surface, optimal_params, focal_length, lens_aperture)
        
        return optimal_params
    
    def visualize_optimized_lens(self, lens_surface_func, params, focal_length, aperture):
        """Visualize the optimized lens"""
        plt.figure(figsize=(12, 8))
        
        # Create lens profile
        x_lens = np.linspace(-aperture/2, aperture/2, 100)
        z_lens = [lens_surface_func(x, params) for x in x_lens]
        
        # Plot lens profile
        plt.plot(x_lens, z_lens, 'b-', linewidth=3, label='Optimized lens surface')
        plt.fill_between(x_lens, 0, z_lens, alpha=0.3, color='lightblue', label='Lens material')
        
        # Plot focal point
        plt.plot(0, focal_length, 'ro', markersize=10, label='Focal point')
        
        # Plot some ray paths
        ray_heights = [-1.5, -1.0, 0, 1.0, 1.5]
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for y_ray, color in zip(ray_heights, colors):
            # Simplified ray path
            plt.plot([y_ray, 0], [0, focal_length], color=color, linewidth=2, 
                    alpha=0.7, linestyle='--')
        
        plt.xlabel('Distance from optical axis')
        plt.ylabel('Distance along optical axis')
        plt.title('Optimized Lens Profile (Fermat\'s Principle)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def optical_path_length_analysis(self):
        """Analyze optical path length in different media"""
        print("\n=== Optical Path Length Analysis ===")
        
        # Compare path lengths in different media
        distances = np.linspace(0, 10, 100)  # physical distances in meters
        
        # Optical path length = n × physical distance
        opl_air = self.n_air * distances
        opl_glass = self.n_glass * distances
        opl_water = self.n_water * distances
        
        # Time for light to travel these distances
        time_air = opl_air / self.c
        time_glass = opl_glass / self.c
        time_water = opl_water / self.c
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(distances, opl_air, 'b-', linewidth=2, label='Air (n=1.0)')
        plt.plot(distances, opl_glass, 'r-', linewidth=2, label='Glass (n=1.5)')
        plt.plot(distances, opl_water, 'g-', linewidth=2, label='Water (n=1.33)')
        plt.xlabel('Physical Distance (m)')
        plt.ylabel('Optical Path Length')
        plt.title('Optical Path Length vs Physical Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(distances, time_air * 1e9, 'b-', linewidth=2, label='Air')
        plt.plot(distances, time_glass * 1e9, 'r-', linewidth=2, label='Glass')
        plt.plot(distances, time_water * 1e9, 'g-', linewidth=2, label='Water')
        plt.xlabel('Physical Distance (m)')
        plt.ylabel('Travel Time (ns)')
        plt.title('Light Travel Time in Different Media')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Demonstrate Fermat's principle with a practical example
        plt.subplot(2, 2, 3)
        
        # Scenario: light from air to water
        h1, h2 = 2.0, 1.5  # heights in air and water
        d_total = 4.0  # total horizontal distance
        
        # Find optimal path
        def total_time_practical(x_intersect):
            d1 = np.sqrt(x_intersect**2 + h1**2)
            d2 = np.sqrt((d_total - x_intersect)**2 + h2**2)
            return (d1 / (self.c/self.n_air)) + (d2 / (self.c/self.n_water))
        
        result_practical = minimize_scalar(total_time_practical, bounds=(0, d_total), method='bounded')
        x_opt_practical = result_practical.x
        
        # Plot time vs intersection point
        x_test = np.linspace(0, d_total, 100)
        times = [total_time_practical(x) * 1e9 for x in x_test]
        
        plt.plot(x_test, times, 'b-', linewidth=2, label='Total time')
        plt.axvline(x=x_opt_practical, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal x = {x_opt_practical:.2f} m')
        plt.xlabel('Intersection Point (m)')
        plt.ylabel('Total Time (ns)')
        plt.title('Fermat\'s Principle: Time Minimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show the optimal path
        plt.subplot(2, 2, 4)
        
        # Air path
        x_air = np.linspace(0, x_opt_practical, 50)
        y_air = h1 - (h1 / x_opt_practical) * x_air
        plt.plot(x_air, y_air, 'b-', linewidth=3, label='Air path')
        
        # Water path
        x_water = np.linspace(x_opt_practical, d_total, 50)
        y_water = -(h2 / (d_total - x_opt_practical)) * (x_water - x_opt_practical)
        plt.plot(x_water, y_water, 'g-', linewidth=3, label='Water path')
        
        # Interface
        plt.axhline(y=0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        
        # Labels
        plt.text(0.2, h1/2, f'Air (n={self.n_air})', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.text(0.2, -h2/2, f'Water (n={self.n_water})', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Optimal Light Path (Air to Water)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        return x_opt_practical


def main():
    """Main function to demonstrate Fermat's principle applications"""
    print("Chapter 2: Calculus of Variations - Fermat Principle and Snell's Law")
    print("=" * 75)
    
    # Initialize the Fermat principle class
    fermat = FermatPrincipleOptics()
    
    # Demonstrate refraction from Fermat's principle
    x_opt, theta1, theta2 = fermat.fermat_principle_refraction()
    
    # Optimize lens shape
    optimal_params = fermat.lens_optimization_problem()
    
    # Analyze optical path lengths
    x_opt_practical = fermat.optical_path_length_analysis()
    
    print("\n=== Key Insights ===")
    print("1. Snell's law emerges naturally from Fermat's principle")
    print("2. Light always takes the path of least time")
    print("3. Lens optimization can be formulated as a variational problem")
    print("4. Optical path length accounts for refractive index variations")
    
    print("\n=== Applications ===")
    print("1. Lens design: optimize surface shapes for focusing")
    print("2. Optical fibers: minimize signal dispersion")
    print("3. Gradient index optics: design materials with varying n")
    print("4. Atmospheric optics: understand mirages and refraction")
    
    print("\n=== Practice Exercises ===")
    print("1. Extend to multiple interfaces (multi-layer coatings)")
    print("2. Include dispersion (wavelength-dependent refractive index)")
    print("3. Implement full ray tracing through complex optical systems")
    print("4. Optimize for different criteria (minimum aberration, etc.)")


if __name__ == "__main__":
    main()