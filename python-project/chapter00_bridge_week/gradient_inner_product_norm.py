"""
Chapter 0: Bridge Week - Mathematical Foundations
Practice Project: Gradient, Inner Product, and Norm in Optics Context
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OpticalMathematics:
    """Mathematical foundations for optical design"""
    
    def __init__(self):
        self.x_range = np.linspace(-5, 5, 100)
        self.y_range = np.linspace(-5, 5, 100)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
    
    def wavefront_error_surface(self, x, y):
        """Simulate a wavefront error surface"""
        # Combination of defocus and astigmatism
        z = 0.3 * (x**2 + y**2) + 0.2 * (x**2 - y**2)
        return z
    
    def compute_gradient(self, scalar_field):
        """Compute gradient of a scalar field"""
        # Using numpy's gradient function
        grad_y, grad_x = np.gradient(scalar_field, self.y_range, self.x_range)
        return grad_x, grad_y
    
    def compute_inner_product(self, field1, field2):
        """Compute L2 inner product between two fields"""
        # Integral of field1 * field2 over the domain
        inner_product = np.trapz(np.trapz(field1 * field2, self.x_range), self.y_range)
        return inner_product
    
    def compute_norm(self, field, norm_type='L2'):
        """Compute different types of norms"""
        if norm_type == 'L2':
            # L2 norm: sqrt(integral of field^2)
            norm = np.sqrt(np.trapz(np.trapz(field**2, self.x_range), self.y_range))
        elif norm_type == 'L1':
            # L1 norm: integral of |field|
            norm = np.trapz(np.trapz(np.abs(field), self.x_range), self.y_range)
        elif norm_type == 'L_inf':
            # L-infinity norm: max absolute value
            norm = np.max(np.abs(field))
        return norm
    
    def visualize_mathematical_concepts(self):
        """Visualize gradient, inner product, and norm concepts"""
        # Create wavefront error surface
        Z = self.wavefront_error_surface(self.X, self.Y)
        
        # Compute gradient
        grad_x, grad_y = self.compute_gradient(Z)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Wavefront error surface
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(self.X, self.Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_title('Wavefront Error Surface\n(Scalar Field)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Error')
        fig.colorbar(surf1, ax=ax1)
        
        # Plot 2: Gradient field (quiver plot)
        ax2 = fig.add_subplot(2, 3, 2)
        # Subsample for clearer visualization
        skip = 5
        ax2.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip], 
                  grad_x[::skip, ::skip], grad_y[::skip, ::skip])
        ax2.set_title('Gradient Field\n(Direction of Steepest Ascent)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient magnitude
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        surf3 = ax3.plot_surface(self.X, self.Y, grad_magnitude, cmap='hot')
        ax3.set_title('Gradient Magnitude\n(How Steep?)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('|∇Z|')
        fig.colorbar(surf3, ax=ax3)
        
        # Plot 4: Different wavefront patterns for inner product demo
        ax4 = fig.add_subplot(2, 3, 4)
        # Create two different aberration patterns
        pattern1 = 0.5 * (self.X**2 + self.Y**2)  # Defocus
        pattern2 = 0.3 * (self.X**2 - self.Y**2)  # Astigmatism
        
        # Compute inner product
        inner_prod = self.compute_inner_product(pattern1, pattern2)
        
        ax4.contourf(self.X, self.Y, pattern1, alpha=0.5, levels=20, cmap='Blues')
        ax4.contour(self.X, self.Y, pattern2, levels=10, colors='red', alpha=0.7)
        ax4.set_title(f'Inner Product Demo\n<Pattern1, Pattern2> = {inner_prod:.2f}')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        
        # Plot 5: Norm comparison
        ax5 = fig.add_subplot(2, 3, 5)
        # Create a field with different characteristics
        test_field = np.exp(-(self.X**2 + self.Y**2)/8) * np.sin(self.X/2)
        
        l2_norm = self.compute_norm(test_field, 'L2')
        l1_norm = self.compute_norm(test_field, 'L1')
        l_inf_norm = self.compute_norm(test_field, 'L_inf')
        
        norms = ['L2', 'L1', 'L∞']
        norm_values = [l2_norm, l1_norm, l_inf_norm]
        
        bars = ax5.bar(norms, norm_values, color=['blue', 'green', 'red'], alpha=0.7)
        ax5.set_title('Different Norms of Test Field')
        ax5.set_ylabel('Norm Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, norm_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 6: Physical interpretation
        ax6 = fig.add_subplot(2, 3, 6)
        # Simulate optical quality metrics
        wavefront_error = self.wavefront_error_surface(self.X, self.Y)
        rms_error = self.compute_norm(wavefront_error, 'L2')
        
        # Create a simple quality metric
        quality = 1 / (1 + rms_error)
        
        ax6.text(0.1, 0.8, f'RMS Wavefront Error: {rms_error:.3f} μm', 
                fontsize=12, transform=ax6.transAxes)
        ax6.text(0.1, 0.6, f'Optical Quality Score: {quality:.3f}', 
                fontsize=12, transform=ax6.transAxes)
        ax6.text(0.1, 0.4, f'Gradient Max: {np.max(grad_magnitude):.3f}', 
                fontsize=12, transform=ax6.transAxes)
        ax6.text(0.1, 0.2, f'Inner Product: {inner_prod:.3f}', 
                fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Physical Interpretation')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("=== Mathematical Foundations Summary ===")
        print(f"Wavefront RMS Error: {rms_error:.4f}")
        print(f"L2 Norm of Wavefront: {self.compute_norm(wavefront_error, 'L2'):.4f}")
        print(f"Maximum Gradient: {np.max(grad_magnitude):.4f}")
        print(f"Inner Product (Defocus, Astigmatism): {inner_prod:.4f}")


def main():
    """Main function to demonstrate mathematical concepts"""
    print("Chapter 0: Bridge Week - Mathematical Foundations")
    print("=" * 50)
    
    # Initialize the optical mathematics class
    optics_math = OpticalMathematics()
    
    # Run visualization
    optics_math.visualize_mathematical_concepts()
    
    # Additional demonstrations
    print("\n=== Additional Demonstrations ===")
    
    # Demonstrate gradient in optimization context
    print("\n1. Gradient as Optimization Direction:")
    print("   - Gradient points to steepest ascent")
    print("   - Negative gradient points to steepest descent")
    print("   - In optical design: gradient tells us how to modify lens surface")
    
    # Demonstrate inner product applications
    print("\n2. Inner Product Applications:")
    print("   - Measures similarity between wavefront patterns")
    print("   - Used in Zernike polynomial decomposition")
    print("   - Orthogonality condition for basis functions")
    
    # Demonstrate norm applications
    print("\n3. Norm Applications:")
    print("   - L2 norm: RMS wavefront error")
    print("   - L1 norm: Total absolute error")
    print("   - L∞ norm: Peak-to-valley error")
    
    print("\n=== Practice Exercises ===")
    print("1. Modify the wavefront_error_surface function to include higher-order aberrations")
    print("2. Implement numerical gradient computation using finite differences")
    print("3. Compare different norm types for various error distributions")
    print("4. Visualize the relationship between gradient magnitude and surface smoothness")


if __name__ == "__main__":
    main()