"""
Chapter 4: Function Spaces - Zernike Polynomials (Simplified Version)
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleZernike:
    def __init__(self, n_points=50):
        self.n_points = n_points
        x = np.linspace(-1, 1, n_points)
        y = np.linspace(-1, 1, n_points)
        self.X, self.Y = np.meshgrid(x, y)
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.theta = np.arctan2(self.Y, self.X)
        self.mask = self.r <= 1.0
    
    def zernike(self, n, m):
        """Simple Zernike polynomial."""
        if abs(m) > n or (n - abs(m)) % 2 != 0:
            return np.zeros_like(self.r)
        
        radial = np.zeros_like(self.r)
        for k in range((n - abs(m)) // 2 + 1):
            radial += ((-1)**k * np.math.factorial(n - k)) / \
                     (np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * 
                      np.math.factorial((n - abs(m)) // 2 - k)) * self.r**(n - 2*k)
        
        if m >= 0:
            return radial * np.cos(abs(m) * self.theta) * self.mask
        else:
            return radial * np.sin(abs(m) * self.theta) * self.mask
    
    def analyze_wavefront(self, wavefront):
        """Decompose wavefront into Zernike coefficients."""
        coefficients = {}
        for n in range(5):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0 and abs(m) <= n:
                    z = self.zernike(n, m)
                    coeff = np.sum(wavefront * z) / np.sum(self.mask)
                    coefficients[f"Z_{n},{m}"] = coeff
        return coefficients


if __name__ == "__main__":
    zernike = SimpleZernike()
    
    # Test defocus
    defocus = 2 * zernike.r**2 - 1
    coeffs = zernike.analyze_wavefront(defocus)
    
    print("Zernike Analysis Results:")
    for name, coeff in coeffs.items():
        if abs(coeff) > 0.01:
            print(f"{name}: {coeff:.4f}")