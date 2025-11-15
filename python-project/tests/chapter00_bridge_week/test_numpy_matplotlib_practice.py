"""
Tests for numpy_matplotlib_practice.py - Chapter 00 Bridge Week
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch

# Add the chapter directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'chapter00_bridge_week'))

from numpy_matplotlib_practice import PythonScientificComputing


class TestPythonScientificComputing:
    """Test class for PythonScientificComputing class"""
    
    @pytest.fixture
    def sci_comp(self):
        """Create an instance of PythonScientificComputing for testing."""
        return PythonScientificComputing()
    
    def test_initialization(self, sci_comp):
        """Test proper initialization of PythonScientificComputing."""
        assert hasattr(sci_comp, 'setup_plotting_style')
        assert callable(sci_comp.setup_plotting_style)
    
    def test_vectorization_correctness(self, sci_comp):
        """Test that vectorization produces correct results."""
        # Create small test arrays
        size = 10
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # Method 1: Python loops (reference implementation)
        C_loops = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                C_loops[i, j] = A[i, j] * B[i, j] + np.sin(A[i, j])
        
        # Method 2: NumPy vectorization
        C_vectorized = A * B + np.sin(A)
        
        # Results should be identical
        np.testing.assert_allclose(C_loops, C_vectorized)
    
    def test_broadcasting_examples(self, sci_comp):
        """Test broadcasting examples."""
        # Example 1: Adding a vector to a matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vector = np.array([10, 20, 30])
        result = matrix + vector
        
        expected = np.array([[11, 22, 33], [14, 25, 36], [17, 28, 39]])
        np.testing.assert_array_equal(result, expected)
        
        # Example 2: Outer product using broadcasting
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        outer = a[:, np.newaxis] * b[np.newaxis, :]
        
        expected_outer = np.array([[4, 5], [8, 10], [12, 15]])
        np.testing.assert_array_equal(outer, expected_outer)
    
    def test_spherical_surface(self, sci_comp):
        """Test spherical surface generation."""
        # Create test coordinates
        X = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        Y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        R = 10.0
        
        sag = sci_comp.spherical_surface(X, Y, R)
        
        # Expected: sag = (X² + Y²) / (2R)
        expected = (X**2 + Y**2) / (2 * R)
        np.testing.assert_allclose(sag, expected)
    
    def test_cylindrical_surface(self, sci_comp):
        """Test cylindrical surface generation."""
        # Create test coordinates
        X = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        Y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        R = 15.0
        
        sag = sci_comp.cylindrical_surface(X, Y, R)
        
        # Expected: sag = X² / (2R) (curvature only in X direction)
        expected = X**2 / (2 * R)
        np.testing.assert_allclose(sag, expected)
        
        # Y values should not affect the result
        Y_different = np.array([[5, 5, 5], [10, 10, 10], [15, 15, 15]])
        sag_different = sci_comp.cylindrical_surface(X, Y_different, R)
        np.testing.assert_allclose(sag_different, expected)
    
    def test_freeform_surface(self, sci_comp):
        """Test freeform surface generation."""
        # Create small test coordinates
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[0.5, 1.0], [1.5, 2.0]])
        
        sag = sci_comp.freeform_surface(X, Y)
        
        # Expected: combination of polynomial terms
        expected = (0.01 * (X**2 + Y**2) + 
                   0.001 * (X**3 - 3*X*Y**2) +  # Trefoil
                   0.0005 * (X**4 - 6*X**2*Y**2 + Y**4) +  # Tetrafoil
                   0.002 * X*Y)  # Astigmatism
        
        np.testing.assert_allclose(sag, expected)
    
    def test_generate_wavefront_error(self, sci_comp):
        """Test wavefront error generation."""
        # Create small test coordinates
        X = np.array([[0.0, 1.0], [2.0, 3.0]])
        Y = np.array([[0.0, 0.5], [1.0, 1.5]])
        
        wavefront = sci_comp.generate_wavefront_error(X, Y)
        
        # Expected: combination of different aberrations
        defocus = 0.5 * (X**2 + Y**2)
        astigmatism = 0.3 * (X**2 - Y**2)
        coma = 0.2 * (X**2 + Y**2) * X
        spherical = 0.1 * (X**2 + Y**2)**2
        expected = defocus + astigmatism + coma + spherical
        
        np.testing.assert_allclose(wavefront, expected)
    
    def test_compute_zernike_coefficients(self, sci_comp):
        """Test Zernike coefficient computation."""
        # Create a simple test wavefront
        size = 10
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create a simple wavefront (just defocus)
        wavefront = X**2 + Y**2
        
        coeffs = sci_comp.compute_zernike_coefficients(wavefront, X, Y)
        
        # Check that coefficients is a dictionary
        assert isinstance(coeffs, dict)
        
        # Check that expected keys are present
        expected_keys = ['defocus', 'astigmatism_0', 'astigmatism_45']
        for key in expected_keys:
            assert key in coeffs
            assert isinstance(coeffs[key], (int, float))
    
    def test_reconstruct_from_zernike(self, sci_comp):
        """Test Zernike reconstruction."""
        # Create test coordinates
        size = 10
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create test coefficients
        coeffs = {
            'defocus': 1.0,
            'astigmatism_0': 0.5,
            'astigmatism_45': 0.3
        }
        
        reconstruction = sci_comp.reconstruct_from_zernike(coeffs, X, Y)
        
        # Check reconstruction shape
        assert reconstruction.shape == X.shape
        
        # Expected reconstruction
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        r_norm = r / np.max(r)
        
        z4 = 2 * r_norm**2 - 1
        z5 = r_norm**2 * np.cos(2*theta)
        z6 = r_norm**2 * np.sin(2*theta)
        
        expected = coeffs['defocus'] * z4 + coeffs['astigmatism_0'] * z5 + coeffs['astigmatism_45'] * z6
        
        np.testing.assert_allclose(reconstruction, expected)
    
    def test_optical_computation_metrics(self, sci_comp):
        """Test optical computation metrics."""
        # Create a simple wavefront
        size = 20
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        wavefront = sci_comp.generate_wavefront_error(X, Y)
        
        # Compute metrics
        rms_error = np.sqrt(np.mean(wavefront**2))
        peak_to_valley = np.max(wavefront) - np.min(wavefront)
        strehl_ratio = np.exp(-(2*np.pi*rms_error)**2)
        
        # Check that metrics are reasonable
        assert rms_error >= 0
        assert peak_to_valley >= 0
        assert 0 <= strehl_ratio <= 1
    
    @pytest.mark.visualization
    def test_visualization_methods(self, sci_comp, mock_matplotlib_show):
        """Test that visualization methods can be called without errors."""
        # Mock the show function to prevent actual plots from appearing
        with patch('matplotlib.pyplot.show'):
            try:
                sci_comp.create_3d_lens_surface()
                sci_comp.demonstrate_optical_computations()
                # If we get here, the methods ran successfully
                assert True
            except Exception as e:
                pytest.fail(f"Visualization method failed: {e}")


class TestMathematicalProperties:
    """Test mathematical properties and relationships."""
    
    def test_broadcasting_properties(self):
        """Test broadcasting mathematical properties."""
        # Test that broadcasting produces the same results as explicit loops
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        vector = np.array([10, 20, 30])
        
        # Broadcasting result
        result_broadcast = matrix + vector
        
        # Explicit loop result
        result_explicit = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result_explicit[i, j] = matrix[i, j] + vector[j]
        
        np.testing.assert_array_equal(result_broadcast, result_explicit)
    
    def test_vectorization_consistency(self):
        """Test that vectorization maintains mathematical consistency."""
        # Test that vectorized operations maintain mathematical properties
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 5)
        
        # Element-wise operations
        elementwise_add = A + B
        elementwise_mult = A * B
        
        # Check properties
        assert elementwise_add.shape == A.shape
        assert elementwise_mult.shape == A.shape
        
        # Check that element-wise multiplication is commutative
        np.testing.assert_array_equal(A * B, B * A)
        
        # Check that element-wise addition is commutative
        np.testing.assert_array_equal(A + B, B + A)


if __name__ == "__main__":
    pytest.main([__file__])