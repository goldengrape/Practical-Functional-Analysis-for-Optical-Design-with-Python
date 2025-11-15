"""
Chapter 0: Bridge Week - Python Scientific Computing
Practice Project: NumPy Vectorization and Matplotlib 3D Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class PythonScientificComputing:
    """Demonstrate NumPy vectorization and Matplotlib 3D visualization"""
    
    def __init__(self):
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up matplotlib plotting style"""
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
    
    def demonstrate_vectorization(self):
        """Demonstrate NumPy vectorization vs Python loops"""
        print("=== NumPy Vectorization Demo ===")
        
        # Create large arrays for timing comparison
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            print(f"\nArray size: {size}x{size}")
            
            # Create test data
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # Method 1: Python loops (slow)
            start_time = time.time()
            C_loops = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    C_loops[i, j] = A[i, j] * B[i, j] + np.sin(A[i, j])
            loop_time = time.time() - start_time
            
            # Method 2: NumPy vectorization (fast)
            start_time = time.time()
            C_vectorized = A * B + np.sin(A)
            vector_time = time.time() - start_time
            
            # Verify results are the same
            assert np.allclose(C_loops, C_vectorized)
            
            print(f"  Python loops: {loop_time:.4f} seconds")
            print(f"  NumPy vectorization: {vector_time:.4f} seconds")
            print(f"  Speedup: {loop_time/vector_time:.1f}x")
    
    def demonstrate_broadcasting(self):
        """Demonstrate NumPy broadcasting"""
        print("\n=== NumPy Broadcasting Demo ===")
        
        # Example 1: Adding a vector to a matrix
        print("\n1. Adding vector to matrix:")
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vector = np.array([10, 20, 30])
        result = matrix + vector
        print(f"Matrix:\n{matrix}")
        print(f"Vector: {vector}")
        print(f"Result (broadcasted):\n{result}")
        
        # Example 2: Outer product using broadcasting
        print("\n2. Outer product using broadcasting:")
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        outer = a[:, np.newaxis] * b[np.newaxis, :]
        print(f"Vector a: {a}")
        print(f"Vector b: {b}")
        print(f"Outer product:\n{outer}")
        
        # Example 3: Broadcasting in optical context
        print("\n3. Optical surface generation using broadcasting:")
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create a lens surface using broadcasting
        R = 10.0  # Radius of curvature
        sag = (X**2 + Y**2) / (2 * R)  # Spherical surface
        
        print(f"Generated spherical surface with R = {R} mm")
        print(f"Surface sag range: [{np.min(sag):.3f}, {np.max(sag):.3f}] mm")
    
    def create_3d_lens_surface(self):
        """Create and visualize a 3D lens surface"""
        print("\n=== 3D Lens Surface Visualization ===")
        
        # Define coordinate system
        N = 100
        x = np.linspace(-10, 10, N)
        y = np.linspace(-10, 10, N)
        X, Y = np.meshgrid(x, y)
        
        # Create different lens surfaces
        surfaces = {
            'Spherical': self.spherical_surface(X, Y, R=20.0),
            'Aspheric': self.aspheric_surface(X, Y),
            'Cylindrical': self.cylindrical_surface(X, Y, R=15.0),
            'Freeform': self.freeform_surface(X, Y)
        }
        
        # Create 3D visualization
        fig = plt.figure(figsize=(16, 12))
        
        for i, (name, surface) in enumerate(surfaces.items(), 1):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            
            # Plot surface
            surf = ax.plot_surface(X, Y, surface, cmap='viridis', 
                                 alpha=0.8, linewidth=0, antialiased=True)
            
            # Customize plot
            ax.set_title(f'{name} Lens Surface')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Sag (mm)')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            
            # Print surface statistics
            print(f"\n{name} surface:")
            print(f"  Sag range: [{np.min(surface):.3f}, {np.max(surface):.3f}] mm")
            print(f"  RMS deviation: {np.std(surface):.3f} mm")
        
        plt.tight_layout()
        plt.show()
        
        return surfaces
    
    def spherical_surface(self, X, Y, R):
        """Generate spherical surface"""
        r_squared = X**2 + Y**2
        sag = r_squared / (2 * R)
        return sag
    
    def aspheric_surface(self, X, Y):
        """Generate aspheric surface"""
        r_squared = X**2 + Y**2
        r = np.sqrt(r_squared)
        
        # Aspheric equation: z = cr² / (1 + √(1 - (1+k)c²r²))
        c = 1/20.0  # Curvature
        k = -0.5    # Conic constant
        
        sag = (c * r_squared) / (1 + np.sqrt(1 - (1 + k) * c**2 * r_squared))
        return sag
    
    def cylindrical_surface(self, X, Y, R):
        """Generate cylindrical surface"""
        sag = X**2 / (2 * R)  # Curvature only in X direction
        return sag
    
    def freeform_surface(self, X, Y):
        """Generate freeform surface with multiple aberrations"""
        # Combination of different polynomial terms
        sag = (0.01 * (X**2 + Y**2) + 
               0.001 * (X**3 - 3*X*Y**2) +  # Trefoil
               0.0005 * (X**4 - 6*X**2*Y**2 + Y**4) +  # Tetrafoil
               0.002 * X*Y)  # Astigmatism
        return sag
    
    def demonstrate_optical_computations(self):
        """Demonstrate optical computations using NumPy"""
        print("\n=== Optical Computations Demo ===")
        
        # Create a simple lens system
        x = np.linspace(-5, 5, 1000)
        y = np.linspace(-5, 5, 1000)
        X, Y = np.meshgrid(x, y)
        
        # Generate wavefront error
        wavefront = self.generate_wavefront_error(X, Y)
        
        # Compute Zernike coefficients (simplified)
        zernike_coeffs = self.compute_zernike_coefficients(wavefront, X, Y)
        
        # Compute optical metrics
        rms_error = np.sqrt(np.mean(wavefront**2))
        peak_to_valley = np.max(wavefront) - np.min(wavefront)
        
        print(f"Wavefront Error Analysis:")
        print(f"  RMS Error: {rms_error:.3f} μm")
        print(f"  Peak-to-Valley: {peak_to_valley:.3f} μm")
        print(f"  Strehl Ratio: {np.exp(-(2*np.pi*rms_error)**2):.3f}")
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original wavefront
        im1 = axes[0].imshow(wavefront, extent=[-5, 5, -5, 5], cmap='RdBu')
        axes[0].set_title('Original Wavefront Error')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0])
        
        # Zernike reconstruction
        reconstruction = self.reconstruct_from_zernike(zernike_coeffs, X, Y)
        im2 = axes[1].imshow(reconstruction, extent=[-5, 5, -5, 5], cmap='RdBu')
        axes[1].set_title('Zernike Reconstruction')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[1])
        
        # Residual error
        residual = wavefront - reconstruction
        im3 = axes[2].imshow(residual, extent=[-5, 5, -5, 5], cmap='RdBu')
        axes[2].set_title('Residual Error')
        axes[2].set_xlabel('X (mm)')
        axes[2].set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
    
    def generate_wavefront_error(self, X, Y):
        """Generate realistic wavefront error"""
        # Combination of different aberrations
        defocus = 0.5 * (X**2 + Y**2)
        astigmatism = 0.3 * (X**2 - Y**2)
        coma = 0.2 * (X**2 + Y**2) * X
        spherical = 0.1 * (X**2 + Y**2)**2
        
        wavefront = defocus + astigmatism + coma + spherical
        return wavefront
    
    def compute_zernike_coefficients(self, wavefront, X, Y):
        """Compute simplified Zernike coefficients"""
        # This is a simplified version - real Zernike polynomials are more complex
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Normalize to unit circle
        r_norm = r / np.max(r)
        
        # Compute coefficients for first few Zernike terms
        coeffs = {}
        
        # Defocus (Z4)
        mask = r_norm <= 1.0
        z4 = 2 * r_norm**2 - 1
        coeffs['defocus'] = np.sum(wavefront[mask] * z4[mask]) / np.sum(z4[mask]**2)
        
        # Astigmatism (Z5, Z6)
        z5 = r_norm**2 * np.cos(2*theta)
        z6 = r_norm**2 * np.sin(2*theta)
        coeffs['astigmatism_0'] = np.sum(wavefront[mask] * z5[mask]) / np.sum(z5[mask]**2)
        coeffs['astigmatism_45'] = np.sum(wavefront[mask] * z6[mask]) / np.sum(z6[mask]**2)
        
        return coeffs
    
    def reconstruct_from_zernike(self, coeffs, X, Y):
        """Reconstruct wavefront from Zernike coefficients"""
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        r_norm = r / np.max(r)
        
        reconstruction = np.zeros_like(X)
        
        # Add defocus
        z4 = 2 * r_norm**2 - 1
        reconstruction += coeffs['defocus'] * z4
        
        # Add astigmatism
        z5 = r_norm**2 * np.cos(2*theta)
        z6 = r_norm**2 * np.sin(2*theta)
        reconstruction += coeffs['astigmatism_0'] * z5
        reconstruction += coeffs['astigmatism_45'] * z6
        
        return reconstruction


def main():
    """Main function to demonstrate Python scientific computing"""
    print("Chapter 0: Bridge Week - Python Scientific Computing")
    print("=" * 60)
    
    # Initialize the scientific computing class
    sci_comp = PythonScientificComputing()
    
    # Run demonstrations
    sci_comp.demonstrate_vectorization()
    sci_comp.demonstrate_broadcasting()
    
    # Create and visualize lens surfaces
    surfaces = sci_comp.create_3d_lens_surface()
    
    # Demonstrate optical computations
    sci_comp.demonstrate_optical_computations()
    
    print("\n=== Key Takeaways ===")
    print("1. NumPy vectorization provides significant speedup over Python loops")
    print("2. Broadcasting allows operations between arrays of different shapes")
    print("3. Matplotlib can create professional 3D visualizations")
    print("4. These tools are essential for optical design computations")
    
    print("\n=== Practice Exercises ===")
    print("1. Implement finite difference gradient computation")
    print("2. Create a toric lens surface (different curvatures in X and Y)")
    print("3. Add more Zernike polynomial terms to the reconstruction")
    print("4. Compare computation times for different array sizes")


if __name__ == "__main__":
    main()