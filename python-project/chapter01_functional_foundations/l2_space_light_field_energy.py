"""
Chapter 1: Functional Foundations - L2 Space and Light Field Energy
Practice Project: Understanding Function Spaces in Optics Context
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate


class L2SpaceOptics:
    """Explore L2 space concepts in optical applications"""
    
    def __init__(self):
        self.setup_domains()
    
    def setup_domains(self):
        """Set up spatial and frequency domains"""
        self.x_range = np.linspace(-10, 10, 1000)
        self.y_range = np.linspace(-10, 10, 1000)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        
        # Spatial frequency domain
        self.fx_range = np.fft.fftshift(np.fft.fftfreq(len(self.x_range), 
                                                       d=self.x_range[1] - self.x_range[0]))
        self.fy_range = np.fft.fftshift(np.fft.fftfreq(len(self.y_range), 
                                                       d=self.y_range[1] - self.y_range[0]))
        self.Fx, self.Fy = np.meshgrid(self.fx_range, self.fy_range)
    
    def l2_norm_function(self, f):
        """Compute L2 norm of a function"""
        # For discrete representation: sqrt(sum(|f|^2 * dx * dy))
        dx = self.x_range[1] - self.x_range[0]
        dy = self.y_range[1] - self.y_range[0]
        
        l2_norm = np.sqrt(np.sum(np.abs(f)**2) * dx * dy)
        return l2_norm
    
    def l2_inner_product(self, f, g):
        """Compute L2 inner product of two functions"""
        dx = self.x_range[1] - self.x_range[0]
        dy = self.y_range[1] - self.y_range[0]
        
        inner_product = np.sum(f * np.conj(g)) * dx * dy
        return inner_product
    
    def light_field_representations(self):
        """Explore different representations of light fields"""
        print("=== Light Field Representations in L2 Space ===")
        
        # 1. Spatial domain representation
        def spatial_field(x, y):
            # Gaussian beam
            w0 = 2.0  # beam waist
            return np.exp(-(x**2 + y**2) / w0**2)
        
        spatial_field_values = spatial_field(self.X, self.Y)
        
        # 2. Frequency domain representation (Fourier transform)
        spatial_field_norm = self.l2_norm_function(spatial_field_values)
        print(f"Spatial field L2 norm: {spatial_field_norm:.4f}")
        
        # Fourier transform
        freq_field = np.fft.fftshift(np.fft.fft2(spatial_field_values))
        freq_field_norm = self.l2_norm_function(freq_field)
        print(f"Frequency field L2 norm: {freq_field_norm:.4f}")
        
        # Parseval's theorem: norms should be equal (up to scaling)
        print(f"Parseval's theorem check: {spatial_field_norm:.4f} ≈ {freq_field_norm / np.sqrt(len(self.x_range) * len(self.y_range)):.4f}")
        
        return spatial_field_values, freq_field
    
    def optical_energy_computation(self):
        """Compute optical energy using L2 norm"""
        print("\n=== Optical Energy Computation ===")
        
        # Different light field patterns
        patterns = {
            'Gaussian Beam': lambda x, y: np.exp(-(x**2 + y**2) / 4),
            'Plane Wave': lambda x, y: np.ones_like(x),
            'Spherical Wave': lambda x, y: np.exp(1j * np.sqrt(x**2 + y**2)),
            'Aberrated Wave': lambda x, y: np.exp(-(x**2 + y**2) / 4) * (1 + 0.1 * np.sin(2*x) * np.cos(3*y))
        }
        
        energies = {}
        
        for name, pattern_func in patterns.items():
            field = pattern_func(self.X, self.Y)
            energy = self.l2_norm_function(field)**2  # Energy ∝ |E|^2
            energies[name] = energy
            print(f"{name} energy: {energy:.4f}")
        
        return energies
    
    def orthogonality_in_optics(self):
        """Demonstrate orthogonality concepts in optical functions"""
        print("\n=== Orthogonality in Optical Functions ===")
        
        # Create orthogonal basis functions (simplified)
        def basis_function_1(x, y):
            return np.exp(-(x**2 + y**2) / 4)  # Gaussian
        
        def basis_function_2(x, y):
            return x * np.exp(-(x**2 + y**2) / 4)  # First-order Hermite-Gauss
        
        def basis_function_3(x, y):
            return y * np.exp(-(x**2 + y**2) / 4)  # First-order Hermite-Gauss (rotated)
        
        basis_functions = [basis_function_1, basis_function_2, basis_function_3]
        basis_names = ['Gaussian', 'Hermite-Gauss (x)', 'Hermite-Gauss (y)']
        
        # Compute inner products
        n_basis = len(basis_functions)
        inner_product_matrix = np.zeros((n_basis, n_basis))
        
        for i in range(n_basis):
            for j in range(n_basis):
                f_i = basis_functions[i](self.X, self.Y)
                f_j = basis_functions[j](self.X, self.Y)
                inner_product_matrix[i, j] = self.l2_inner_product(f_i, f_j)
        
        print("Inner product matrix (should be diagonal for orthogonal basis):")
        print(inner_product_matrix)
        
        # Check orthogonality
        print("\nOrthogonality check:")
        for i in range(n_basis):
            for j in range(i+1, n_basis):
                inner_prod = inner_product_matrix[i, j]
                print(f"<{basis_names[i]}, {basis_names[j]}> = {inner_prod:.6f}")
        
        return inner_product_matrix
    
    def function_approximation_demo(self):
        """Demonstrate function approximation in L2 space"""
        print("\n=== Function Approximation in L2 Space ===")
        
        # Target function (complex wavefront)
        def target_wavefront(x, y):
            return np.exp(-(x**2 + y**2) / 9) * (1 + 0.3 * np.sin(x) * np.cos(y) + 
                                               0.2 * (x**2 - y**2) / 4)
        
        target = target_wavefront(self.X, self.Y)
        
        # Approximate using basis functions
        basis_functions = [
            lambda x, y: np.exp(-(x**2 + y**2) / 4),
            lambda x, y: x * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: y * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: (x**2 - y**2) * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: x * y * np.exp(-(x**2 + y**2) / 4)
        ]
        
        # Compute optimal coefficients (orthogonal projection)
        coefficients = []
        for basis_func in basis_functions:
            basis_field = basis_func(self.X, self.Y)
            coeff = self.l2_inner_product(target, basis_field) / self.l2_inner_product(basis_field, basis_field)
            coefficients.append(coeff)
        
        print("Optimal coefficients for approximation:")
        for i, coeff in enumerate(coefficients):
            print(f"  Basis {i+1}: {coeff:.4f}")
        
        # Reconstruct approximation
        approximation = np.zeros_like(target)
        for i, (basis_func, coeff) in enumerate(zip(basis_functions, coefficients)):
            approximation += coeff * basis_func(self.X, self.Y)
        
        # Compute approximation error
        error = target - approximation
        error_norm = self.l2_norm_function(error)
        target_norm = self.l2_norm_function(target)
        relative_error = error_norm / target_norm
        
        print(f"Approximation error (L2 norm): {error_norm:.4f}")
        print(f"Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        
        # Visualize results
        self.visualize_approximation(target, approximation, error)
        
        return target, approximation, error, coefficients
    
    def visualize_approximation(self, target, approximation, error):
        """Visualize function approximation results"""
        fig = plt.figure(figsize=(15, 5))
        
        # Target function
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(self.X, self.Y, np.real(target), cmap='viridis')
        ax1.set_title('Target Wavefront')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Phase')
        
        # Approximation
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(self.X, self.Y, np.real(approximation), cmap='viridis')
        ax2.set_title('Approximation')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Phase')
        
        # Error
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        surf3 = ax3.plot_surface(self.X, self.Y, np.real(error), cmap='RdBu')
        ax3.set_title('Approximation Error')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Error')
        
        plt.tight_layout()
        plt.show()
    
    def convergence_analysis(self):
        """Analyze convergence of function approximations"""
        print("\n=== Convergence Analysis ===")
        
        # Target function
        def target_wavefront(x, y):
            return np.exp(-(x**2 + y**2) / 4) * np.sin(2*x) * np.cos(3*y)
        
        target = target_wavefront(self.X, self.Y)
        
        # Build basis set incrementally
        basis_functions = [
            lambda x, y: np.exp(-(x**2 + y**2) / 4),
            lambda x, y: x * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: y * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: x**2 * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: y**2 * np.exp(-(x**2 + y**2) / 4),
            lambda x, y: x*y * np.exp(-(x**2 + y**2) / 4),
        ]
        
        errors = []
        num_basis_functions = []
        
        for n_basis in range(1, len(basis_functions) + 1):
            current_basis = basis_functions[:n_basis]
            
            # Compute approximation
            approximation = np.zeros_like(target)
            for basis_func in current_basis:
                basis_field = basis_func(self.X, self.Y)
                coeff = self.l2_inner_product(target, basis_field) / self.l2_inner_product(basis_field, basis_field)
                approximation += coeff * basis_field
            
            # Compute error
            error = self.l2_norm_function(target - approximation)
            errors.append(error)
            num_basis_functions.append(n_basis)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(num_basis_functions, errors, 'bo-')
        plt.xlabel('Number of Basis Functions')
        plt.ylabel('L2 Error')
        plt.title('Approximation Convergence')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(num_basis_functions, errors, 'ro-')
        plt.xlabel('Number of Basis Functions')
        plt.ylabel('L2 Error')
        plt.title('Convergence (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final error with {len(basis_functions)} basis functions: {errors[-1]:.6f}")
        
        return errors, num_basis_functions


def main():
    """Main function to demonstrate L2 space concepts"""
    print("Chapter 1: Functional Foundations - L2 Space and Light Field Energy")
    print("=" * 70)
    
    # Initialize the L2 space optics class
    l2_optics = L2SpaceOptics()
    
    # Run demonstrations
    spatial_field, freq_field = l2_optics.light_field_representations()
    energies = l2_optics.optical_energy_computation()
    inner_product_matrix = l2_optics.orthogonality_in_optics()
    target, approximation, error, coefficients = l2_optics.function_approximation_demo()
    errors, num_basis = l2_optics.convergence_analysis()
    
    print("\n=== Key Insights ===")
    print("1. L2 norm represents optical energy/intensity")
    print("2. Parseval's theorem connects spatial and frequency domains")
    print("3. Orthogonal basis functions simplify function approximation")
    print("4. Function approximation error decreases with more basis functions")
    
    print("\n=== Practice Exercises ===")
    print("1. Implement different orthogonal basis sets (Zernike, Hermite-Gauss)")
    print("2. Compare convergence rates for different target functions")
    print("3. Explore non-orthogonal basis functions and their properties")
    print("4. Implement weighted L2 spaces for optical applications")


if __name__ == "__main__":
    main()