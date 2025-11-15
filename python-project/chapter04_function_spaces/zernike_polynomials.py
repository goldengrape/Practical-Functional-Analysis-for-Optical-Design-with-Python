"""
Chapter 4: Function Spaces - Zernike Polynomials for Optical Wavefront Analysis
Functional Analysis for Optical Design
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_jacobi
import pandas as pd


class ZernikePolynomials:
    """
    Zernike polynomials for optical wavefront analysis.
    Demonstrates orthogonal function bases in circular domains.
    """
    
    def __init__(self, max_n: int = 8, grid_size: int = 100):
        self.max_n = max_n
        self.grid_size = grid_size
        self.x, self.y = np.meshgrid(
            np.linspace(-1, 1, grid_size),
            np.linspace(-1, 1, grid_size)
        )
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
        
        # Create circular mask
        self.mask = self.r <= 1.0
    
    def zernike_radial(self, n: int, m: int, r: np.ndarray) -> np.ndarray:
        """Compute radial component of Zernike polynomial."""
        if (n - m) % 2 != 0:
            return np.zeros_like(r)
        
        l = (n - m) // 2
        radial = np.zeros_like(r)
        
        for k in range(l + 1):
            coeff = ((-1)**k * np.math.factorial(n - k)) / \
                   (np.math.factorial(k) * np.math.factorial(l - k) * np.math.factorial((n + m) // 2 - k) * np.math.factorial((n - m) // 2 - k))
            radial += coeff * r**(n - 2*k)
        
        return radial
    
    def zernike_polynomial(self, n: int, m: int) -> np.ndarray:
        """Compute full Zernike polynomial."""
        if m >= 0:
            radial = self.zernike_radial(n, abs(m), self.r)
            zernike = radial * np.cos(m * self.theta)
        else:
            radial = self.zernike_radial(n, abs(m), self.r)
            zernike = radial * np.sin(abs(m) * self.theta)
        
        return zernike * self.mask
    
    def zernike_basis(self) -> dict:
        """Generate complete Zernike basis up to max_n."""
        basis = {}
        j = 1  # Fringe indexing
        
        for n in range(self.max_n + 1):
            for m in range(-n, n + 1, 2):  # m has same parity as n
                if abs(m) <= n:
                    basis[f"Z_{j} (n={n}, m={m})"] = self.zernike_polynomial(n, m)
                    j += 1
        
        return basis
    
    def wavefront_decomposition(self, wavefront: np.ndarray) -> dict:
        """
        Decompose wavefront into Zernike coefficients.
        Demonstrates function space projection.
        """
        basis = self.zernike_basis()
        coefficients = {}
        
        # Apply circular mask to wavefront
        masked_wavefront = wavefront * self.mask
        
        for name, zernike in basis.items():
            # Compute inner product (projection)
            inner_product = np.sum(masked_wavefront * zernike) / np.sum(self.mask)
            
            # Compute norm for normalization
            zernike_norm = np.sqrt(np.sum(zernike**2) / np.sum(self.mask))
            
            # Normalized coefficient
            coefficient = inner_product / zernike_norm if zernike_norm > 0 else 0
            coefficients[name] = coefficient
        
        return coefficients
    
    def reconstruct_wavefront(self, coefficients: dict, max_terms: int = None) -> np.ndarray:
        """Reconstruct wavefront from Zernike coefficients."""
        basis = self.zernike_basis()
        reconstructed = np.zeros_like(self.r)
        
        if max_terms is None:
            max_terms = len(coefficients)
        
        for i, (name, coeff) in enumerate(coefficients.items()):
            if i >= max_terms:
                break
            if name in basis:
                reconstructed += coeff * basis[name]
        
        return reconstructed * self.mask
    
    def analyze_optical_aberrations(self, wavefront: np.ndarray) -> dict:
        """
        Analyze optical aberrations using Zernike decomposition.
        """
        coefficients = self.wavefront_decomposition(wavefront)
        
        # Group coefficients by aberration type
        aberrations = {
            'piston': 0.0,
            'tilt_x': 0.0,
            'tilt_y': 0.0,
            'defocus': 0.0,
            'astigmatism_0': 0.0,
            'astigmatism_45': 0.0,
            'coma_x': 0.0,
            'coma_y': 0.0,
            'spherical': 0.0,
            'trefoil_x': 0.0,
            'trefoil_y': 0.0
        }
        
        # Map Zernike coefficients to aberrations
        for name, coeff in coefficients.items():
            if 'n=0, m=0' in name:
                aberrations['piston'] = coeff
            elif 'n=1, m=-1' in name:
                aberrations['tilt_y'] = coeff
            elif 'n=1, m=1' in name:
                aberrations['tilt_x'] = coeff
            elif 'n=2, m=0' in name:
                aberrations['defocus'] = coeff
            elif 'n=2, m=-2' in name:
                aberrations['astigmatism_45'] = coeff
            elif 'n=2, m=2' in name:
                aberrations['astigmatism_0'] = coeff
            elif 'n=3, m=-1' in name:
                aberrations['coma_y'] = coeff
            elif 'n=3, m=1' in name:
                aberrations['coma_x'] = coeff
            elif 'n=4, m=0' in name:
                aberrations['spherical'] = coeff
            elif 'n=3, m=-3' in name:
                aberrations['trefoil_y'] = coeff
            elif 'n=3, m=3' in name:
                aberrations['trefoil_x'] = coeff
        
        return aberrations
    
    def generate_test_wavefront(self, aberration_type: str = 'mixed', 
                               amplitude: float = 1.0) -> np.ndarray:
        """Generate test wavefront with specific aberrations."""
        if aberration_type == 'defocus':
            # Defocus: n=2, m=0
            wavefront = amplitude * (2 * self.r**2 - 1)
        elif aberration_type == 'astigmatism':
            # Astigmatism: n=2, m=2
            wavefront = amplitude * self.r**2 * np.cos(2 * self.theta)
        elif aberration_type == 'coma':
            # Coma: n=3, m=1
            wavefront = amplitude * (3 * self.r**3 - 2 * self.r) * np.cos(self.theta)
        elif aberration_type == 'spherical':
            # Spherical: n=4, m=0
            wavefront = amplitude * (6 * self.r**4 - 6 * self.r**2 + 1)
        elif aberration_type == 'mixed':
            # Mixed aberrations
            wavefront = (0.5 * amplitude * (2 * self.r**2 - 1) +  # Defocus
                        0.3 * amplitude * self.r**2 * np.cos(2 * self.theta) +  # Astigmatism
                        0.2 * amplitude * (3 * self.r**3 - 2 * self.r) * np.cos(self.theta))  # Coma
        else:
            wavefront = np.zeros_like(self.r)
        
        return wavefront * self.mask
    
    def orthogonality_check(self) -> dict:
        """Verify orthogonality of Zernike basis."""
        basis = self.zernike_basis()
        n_terms = len(basis)
        
        # Compute inner product matrix
        inner_product_matrix = np.zeros((n_terms, n_terms))
        
        basis_names = list(basis.keys())
        
        for i, name_i in enumerate(basis_names):
            for j, name_j in enumerate(basis_names):
                zernike_i = basis[name_i]
                zernike_j = basis[name_j]
                
                # Inner product over unit disk
                inner_product = np.sum(zernike_i * zernike_j) / np.sum(self.mask)
                inner_product_matrix[i, j] = inner_product
        
        # Check orthogonality
        off_diagonal = inner_product_matrix - np.diag(np.diag(inner_product_matrix))
        max_off_diagonal = np.max(np.abs(off_diagonal))
        
        return {
            'inner_product_matrix': inner_product_matrix,
            'max_off_diagonal': max_off_diagonal,
            'is_orthogonal': max_off_diagonal < 1e-10
        }


def demonstrate_zernike_analysis():
    """Demonstrate Zernike polynomial analysis for optical wavefronts."""
    print("Zernike Polynomials for Optical Wavefront Analysis")
    print("=" * 55)
    
    # Initialize Zernike analyzer
    zernike = ZernikePolynomials(max_n=6, grid_size=100)
    
    # Test different aberrations
    test_cases = ['defocus', 'astigmatism', 'coma', 'spherical', 'mixed']
    
    for aberration in test_cases:
        print(f"\n--- Analyzing {aberration.title()} Aberration ---")
        
        # Generate test wavefront
        wavefront = zernike.generate_test_wavefront(aberration, amplitude=1.0)
        
        # Decompose into Zernike coefficients
        coefficients = zernike.wavefront_decomposition(wavefront)
        
        # Analyze aberrations
        aberration_analysis = zernike.analyze_optical_aberrations(wavefront)
        
        print(f"Key aberration coefficients:")
        for key, value in aberration_analysis.items():
            if abs(value) > 0.01:  # Only show significant coefficients
                print(f"  {key}: {value:.4f}")
        
        # Reconstruction error
        reconstructed = zernike.reconstruct_wavefront(coefficients)
        error = wavefront - reconstructed
        rms_error = np.sqrt(np.mean(error**2))
        print(f"RMS reconstruction error: {rms_error:.6f}")
    
    # Orthogonality verification
    print(f"\n--- Orthogonality Check ---")
    orthogonality = zernike.orthogonality_check()
    print(f"Maximum off-diagonal inner product: {orthogonality['max_off_diagonal']:.2e}")
    print(f"Basis is orthogonal: {orthogonality['is_orthogonal']}")
    
    # Function space properties
    print(f"\n--- Function Space Properties ---")
    print(f"Zernike polynomials form an orthogonal basis on the unit disk")
    print(f"They are complete in L²(D) where D is the unit disk")
    print(f"Each polynomial represents a specific optical aberration")
    print(f"The decomposition provides physical insight into wavefront errors")


if __name__ == "__main__":
    demonstrate_zernike_analysis()