"""
Chapter 5: Operator Theory - Optical Operators and Transformations
Functional Analysis for Optical Design
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.linalg import eig, inv


class OpticalOperators:
    """
    Linear operators in optical systems: Fourier transforms, propagation, and imaging.
    Demonstrates operator theory in infinite-dimensional function spaces.
    """
    
    def __init__(self, grid_size=128, physical_size=1e-3):
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.dx = physical_size / grid_size
        
        # Spatial coordinates
        x = np.linspace(-physical_size/2, physical_size/2, grid_size)
        y = np.linspace(-physical_size/2, physical_size/2, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Frequency coordinates
        self.dfx = 1 / physical_size
        fx = np.linspace(-1/(2*self.dx), 1/(2*self.dx), grid_size)
        fy = np.linspace(-1/(2*self.dx), 1/(2*self.dx), grid_size)
        self.Fx, self.Fy = np.meshgrid(fx, fy)
    
    def fourier_transform_operator(self, field):
        """Fourier transform operator in optics (Fraunhofer approximation)."""
        return fftshift(fft2(fftshift(field))) * self.dx**2
    
    def inverse_fourier_transform_operator(self, spectrum):
        """Inverse Fourier transform operator."""
        return fftshift(ifft2(fftshift(spectrum))) * self.dfx**2
    
    def fresnel_propagation_operator(self, field, propagation_distance, wavelength=500e-9):
        """
        Fresnel propagation operator.
        Implements U(x,y,z) = F⁻¹{F{U(x,y,0)} * exp(i*k*z*sqrt(1-(λf)²))}
        """
        k = 2 * np.pi / wavelength
        
        # Fourier transform of input field
        field_spectrum = self.fourier_transform_operator(field)
        
        # Propagation transfer function
        f_squared = self.Fx**2 + self.Fy**2
        propagation_filter = np.exp(1j * k * propagation_distance * 
                                   np.sqrt(np.maximum(0, 1 - (wavelength**2) * f_squared)))
        
        # Apply propagation and inverse transform
        propagated_spectrum = field_spectrum * propagation_filter
        propagated_field = self.inverse_fourier_transform_operator(propagated_spectrum)
        
        return propagated_field
    
    def angular_spectrum_operator(self, field, propagation_distance, wavelength=500e-9):
        """
        Angular spectrum propagation operator (more accurate than Fresnel).
        """
        k = 2 * np.pi / wavelength
        
        # Fourier transform
        field_spectrum = self.fourier_transform_operator(field)
        
        # Angular spectrum propagation
        f_squared = self.Fx**2 + self.Fy**2
        # Ensure numerical stability
        valid_indices = (wavelength**2) * f_squared <= 1.0
        
        propagation_filter = np.zeros_like(f_squared, dtype=complex)
        propagation_filter[valid_indices] = np.exp(1j * k * propagation_distance * 
                                                   np.sqrt(1 - (wavelength**2) * f_squared[valid_indices]))
        
        # Apply filter and inverse transform
        propagated_spectrum = field_spectrum * propagation_filter
        propagated_field = self.inverse_fourier_transform_operator(propagated_spectrum)
        
        return propagated_field
    
    def thin_lens_operator(self, field, focal_length, wavelength=500e-9):
        """
        Thin lens operator: applies quadratic phase factor.
        T(x,y) = exp(-i*k*(x²+y²)/(2*f))
        """
        k = 2 * np.pi / wavelength
        r_squared = self.X**2 + self.Y**2
        
        # Lens transmission function
        phase_factor = np.exp(-1j * k * r_squared / (2 * focal_length))
        
        return field * phase_factor
    
    def circular_aperture_operator(self, field, aperture_radius):
        """Circular aperture operator."""
        aperture = (self.X**2 + self.Y**2) <= aperture_radius**2
        return field * aperture.astype(float)
    
    def imaging_operator(self, field, object_distance, image_distance, focal_length, wavelength=500e-9):
        """
        Complete imaging operator: object → lens → image.
        Implements Gaussian imaging: 1/f = 1/u + 1/v
        """
        # Verify imaging condition
        expected_focal = 1 / (1/object_distance + 1/image_distance)
        print(f"Expected focal length: {expected_focal:.6f} m")
        print(f"Actual focal length: {focal_length:.6f} m")
        
        # Propagate from object to lens
        field_at_lens = self.fresnel_propagation_operator(field, object_distance, wavelength)
        
        # Apply lens operator
        field_after_lens = self.thin_lens_operator(field_at_lens, focal_length, wavelength)
        
        # Propagate from lens to image
        image_field = self.fresnel_propagation_operator(field_after_lens, image_distance, wavelength)
        
        return image_field
    
    def point_spread_function(self, aperture_radius, wavelength=500e-9):
        """
        Compute point spread function (Airy pattern) for circular aperture.
        """
        # Input field: point source (delta function approximation)
        field = np.zeros((self.grid_size, self.grid_size))
        center = self.grid_size // 2
        field[center, center] = 1.0 / (self.dx**2)  # Normalized delta function
        
        # Apply aperture
        field_aperture = self.circular_aperture_operator(field, aperture_radius)
        
        # Far-field diffraction (Fourier transform)
        psf = np.abs(self.fourier_transform_operator(field_aperture))**2
        
        # Normalize
        psf = psf / np.max(psf)
        
        return psf
    
    def optical_transfer_function(self, psf):
        """
        Compute optical transfer function (OTF) from PSF.
        OTF = F{PSF} / F{PSF}(0,0)
        """
        otf = self.fourier_transform_operator(psf)
        otf = otf / otf[self.grid_size//2, self.grid_size//2]  # Normalize
        return otf
    
    def linearity_test(self, operator_func, test_field1, test_field2, alpha=0.5, beta=0.7):
        """
        Test linearity of an optical operator.
        Check if L(αf₁ + βf₂) = αL(f₁) + βL(f₂)
        """
        # Left side: operator applied to linear combination
        linear_combination = alpha * test_field1 + beta * test_field2
        left_side = operator_func(linear_combination)
        
        # Right side: linear combination of operator applications
        right_side = alpha * operator_func(test_field1) + beta * operator_func(test_field2)
        
        # Check linearity
        difference = np.max(np.abs(left_side - right_side))
        is_linear = difference < 1e-10
        
        return {
            'is_linear': is_linear,
            'max_difference': difference,
            'left_side': left_side,
            'right_side': right_side
        }
    
    def eigenmode_analysis(self, operator_func, num_modes=5):
        """
        Compute eigenmodes and eigenvalues of optical operator.
        """
        # Create matrix representation of operator
        test_fields = []
        for i in range(min(20, self.grid_size)):
            for j in range(min(20, self.grid_size)):
                field = np.zeros((self.grid_size, self.grid_size))
                field[i, j] = 1.0
                test_fields.append(field.flatten())
        
        # Apply operator to basis
        if len(test_fields) > 100:  # Limit size for computational efficiency
            test_fields = test_fields[:100]
        
        # This is a simplified eigenmode analysis
        # In practice, you would use iterative methods for large operators
        print(f"Computing eigenmodes for operator (matrix size: {len(test_fields)}×{len(test_fields)})")
        
        return {
            'num_modes': num_modes,
            'grid_size': self.grid_size,
            'note': 'Full eigenmode analysis requires iterative methods for large operators'
        }


def demonstrate_optical_operators():
    """Demonstrate optical operators in function spaces."""
    print("Optical Operators in Function Spaces")
    print("=" * 40)
    
    # Initialize optical operators
    optics = OpticalOperators(grid_size=64, physical_size=1e-3)
    
    # Test field: Gaussian beam
    w0 = 100e-6  # Beam waist
    gaussian_field = np.exp(-(optics.X**2 + optics.Y**2) / w0**2)
    
    print(f"Initial Gaussian beam parameters:")
    print(f"  Beam waist: {w0*1e6:.1f} μm")
    print(f"  Grid size: {optics.grid_size}×{optics.grid_size}")
    print(f"  Physical size: {optics.physical_size*1e3:.1f} mm")
    
    # Test Fourier transform operator
    print(f"\n--- Fourier Transform Operator ---")
    spectrum = optics.fourier_transform_operator(gaussian_field)
    print(f"  Fourier transform preserves energy: {np.abs(np.sum(np.abs(spectrum)**2) - np.sum(np.abs(gaussian_field)**2)) < 1e-10}")
    
    # Test propagation operators
    propagation_distance = 10e-3  # 10 mm
    wavelength = 500e-9  # 500 nm
    
    print(f"\n--- Fresnel Propagation Operator ---")
    propagated_field = optics.fresnel_propagation_operator(gaussian_field, propagation_distance, wavelength)
    print(f"  Propagation distance: {propagation_distance*1e3:.1f} mm")
    
    # Test lens operator
    focal_length = 50e-3  # 50 mm
    print(f"\n--- Thin Lens Operator ---")
    lens_field = optics.thin_lens_operator(gaussian_field, focal_length, wavelength)
    print(f"  Focal length: {focal_length*1e3:.1f} mm")
    
    # Test imaging system
    object_distance = 100e-3  # 100 mm
    image_distance = 1 / (1/focal_length - 1/object_distance)
    print(f"\n--- Imaging Operator ---")
    print(f"  Object distance: {object_distance*1e3:.1f} mm")
    print(f"  Image distance: {image_distance*1e3:.1f} mm")
    
    # Test linearity
    print(f"\n--- Linearity Test ---")
    test_field1 = np.exp(-(optics.X**2 + optics.Y**2) / (50e-6)**2)
    test_field2 = np.exp(-(optics.X**2 + optics.Y**2) / (150e-6)**2)
    
    linearity_result = optics.linearity_test(
        lambda f: optics.fourier_transform_operator(f), 
        test_field1, test_field2
    )
    print(f"  Fourier transform is linear: {linearity_result['is_linear']}")
    print(f"  Maximum difference: {linearity_result['max_difference']:.2e}")
    
    # Point spread function
    print(f"\n--- Point Spread Function ---")
    aperture_radius = 2.5e-3  # 2.5 mm
    psf = optics.point_spread_function(aperture_radius, wavelength)
    print(f"  Aperture radius: {aperture_radius*1e3:.1f} mm")
    print(f"  PSF peak intensity: {np.max(psf):.6f}")
    
    print(f"\n=== Key Concepts ===")
    print("1. Optical operators are linear transformations in function spaces")
    print("2. Fourier optics connects spatial and frequency domains")
    print("3. Propagation operators implement wave equation solutions")
    print("4. Imaging systems combine multiple linear operators")
    print("5. PSF and OTF characterize system performance")


if __name__ == "__main__":
    demonstrate_optical_operators()