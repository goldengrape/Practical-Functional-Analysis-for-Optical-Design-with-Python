"""
Tests for l2_space_light_field_energy.py - Chapter 01 Functional Foundations
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch

# Add the chapter directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'chapter01_functional_foundations'))

from l2_space_light_field_energy import L2SpaceOptics


class TestL2SpaceOptics:
    """Test class for L2SpaceOptics class"""
    
    @pytest.fixture
    def l2_optics(self):
        """Create an instance of L2SpaceOptics for testing."""
        return L2SpaceOptics()
    
    def test_initialization(self, l2_optics):
        """Test proper initialization of L2SpaceOptics."""
        assert hasattr(l2_optics, 'x_range')
        assert hasattr(l2_optics, 'y_range')
        assert hasattr(l2_optics, 'X')
        assert hasattr(l2_optics, 'Y')
        assert hasattr(l2_optics, 'fx_range')
        assert hasattr(l2_optics, 'fy_range')
        assert hasattr(l2_optics, 'Fx')
        assert hasattr(l2_optics, 'Fy')
        
        # Check array shapes
        assert len(l2_optics.x_range) == 1000
        assert len(l2_optics.y_range) == 1000
        assert l2_optics.X.shape == (1000, 1000)
        assert l2_optics.Y.shape == (1000, 1000)
        assert l2_optics.Fx.shape == (1000, 1000)
        assert l2_optics.Fy.shape == (1000, 1000)
    
    def test_l2_norm_function(self, l2_optics):
        """Test L2 norm computation."""
        # Test with a simple function
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        
        l2_norm = l2_optics.l2_norm_function(f)
        
        # L2 norm should be positive
        assert l2_norm > 0
        assert np.isscalar(l2_norm)
        
        # Test with zero function
        zero_f = np.zeros_like(l2_optics.X)
        zero_norm = l2_optics.l2_norm_function(zero_f)
        assert zero_norm == 0.0
        
        # Test homogeneity property: ||αf|| = |α| ||f||
        alpha = 2.5
        scaled_f = alpha * f
        scaled_norm = l2_optics.l2_norm_function(scaled_f)
        expected_scaled_norm = abs(alpha) * l2_norm
        assert abs(scaled_norm - expected_scaled_norm) < 1e-10
    
    def test_l2_inner_product(self, l2_optics):
        """Test L2 inner product computation."""
        # Test with simple functions
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        g = np.exp(-2 * (l2_optics.X**2 + l2_optics.Y**2))
        
        inner_product = l2_optics.l2_inner_product(f, g)
        
        # Inner product should be a scalar
        assert np.isscalar(inner_product)
        
        # Test symmetry: <f, g> = <g, f>* (for complex functions)
        inner_product_fg = l2_optics.l2_inner_product(f, g)
        inner_product_gf = l2_optics.l2_inner_product(g, f)
        assert abs(inner_product_fg - np.conj(inner_product_gf)) < 1e-10
        
        # Test linearity in first argument: <αf + βg, h> = α<f, h> + β<g, h>
        h = np.exp(-3 * (l2_optics.X**2 + l2_optics.Y**2))
        alpha, beta = 2.0, 3.0
        
        left_side = l2_optics.l2_inner_product(alpha * f + beta * g, h)
        right_side = alpha * l2_optics.l2_inner_product(f, h) + beta * l2_optics.l2_inner_product(g, h)
        assert abs(left_side - right_side) < 1e-10
    
    def test_cauchy_schwarz_inequality(self, l2_optics):
        """Test Cauchy-Schwarz inequality: |<f, g>| ≤ ||f|| ||g||"""
        # Test with various functions
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        g = np.exp(-2 * (l2_optics.X**2 + l2_optics.Y**2))
        
        inner_product = abs(l2_optics.l2_inner_product(f, g))
        norm_f = l2_optics.l2_norm_function(f)
        norm_g = l2_optics.l2_norm_function(g)
        
        # Cauchy-Schwarz inequality
        assert inner_product <= norm_f * norm_g + 1e-10
    
    def test_parseval_theorem(self, l2_optics):
        """Test Parseval's theorem: ||f||² = ||F{f}||² / N (approximately)"""
        # Create a test function
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        
        # Compute L2 norm in spatial domain
        spatial_norm = l2_optics.l2_norm_function(f)
        
        # Compute Fourier transform
        f_fft = np.fft.fftshift(np.fft.fft2(f))
        
        # Compute L2 norm in frequency domain
        freq_norm = l2_optics.l2_norm_function(f_fft)
        
        # Parseval's theorem: ||f||² = ||F{f}||² / N
        # where N is the total number of samples
        N = len(l2_optics.x_range) * len(l2_optics.y_range)
        
        # The norms should be related by Parseval's theorem
        expected_freq_norm = spatial_norm * np.sqrt(N)
        assert abs(freq_norm - expected_freq_norm) < 1e-6
    
    def test_light_field_representations(self, l2_optics):
        """Test light field representations."""
        spatial_field, freq_field = l2_optics.light_field_representations()
        
        # Check that spatial field has correct shape
        assert spatial_field.shape == l2_optics.X.shape
        
        # Check that frequency field has correct shape
        assert freq_field.shape == l2_optics.X.shape
        
        # Check that spatial field is real and positive (Gaussian)
        assert np.all(np.isreal(spatial_field))
        assert np.all(spatial_field >= 0)
        
        # Check that frequency field is complex (Fourier transform)
        assert np.any(np.iscomplex(freq_field))
    
    def test_optical_energy_computation(self, l2_optics):
        """Test optical energy computation."""
        energies = l2_optics.optical_energy_computation()
        
        # Check that energies is a dictionary
        assert isinstance(energies, dict)
        
        # Check that expected patterns are present
        expected_patterns = ['Gaussian Beam', 'Plane Wave', 'Spherical Wave', 'Aberrated Wave']
        for pattern in expected_patterns:
            assert pattern in energies
            assert isinstance(energies[pattern], (int, float))
            assert energies[pattern] >= 0  # Energy should be non-negative
        
        # Check energy relationships
        # Gaussian beam should have finite energy
        assert energies['Gaussian Beam'] > 0
        assert energies['Gaussian Beam'] < np.inf
        
        # Plane wave should have infinite energy (but our discrete approximation is finite)
        assert energies['Plane Wave'] > 0
    
    def test_orthogonality_in_optics(self, l2_optics):
        """Test orthogonality concepts in optical functions."""
        inner_product_matrix = l2_optics.orthogonality_in_optics()
        
        # Check that matrix is square
        assert inner_product_matrix.shape[0] == inner_product_matrix.shape[1]
        assert inner_product_matrix.shape == (3, 3)
        
        # Check that matrix is symmetric (approximately)
        np.testing.assert_allclose(inner_product_matrix, inner_product_matrix.T, rtol=1e-10)
        
        # Check that diagonal elements are positive (norms)
        for i in range(3):
            assert inner_product_matrix[i, i] > 0
        
        # Check that off-diagonal elements are small (orthogonality)
        for i in range(3):
            for j in range(i+1, 3):
                # For orthogonal functions, inner product should be small
                inner_prod = abs(inner_product_matrix[i, j])
                norm_i = np.sqrt(inner_product_matrix[i, i])
                norm_j = np.sqrt(inner_product_matrix[j, j])
                relative_inner_prod = inner_prod / (norm_i * norm_j)
                assert relative_inner_prod < 0.1  # Should be nearly orthogonal
    
    def test_function_approximation(self, l2_optics):
        """Test function approximation in L2 space."""
        target, approximation, error, coefficients = l2_optics.function_approximation_demo()
        
        # Check shapes
        assert target.shape == l2_optics.X.shape
        assert approximation.shape == l2_optics.X.shape
        assert error.shape == l2_optics.X.shape
        
        # Check that coefficients is a list of numbers
        assert isinstance(coefficients, list)
        for coeff in coefficients:
            assert isinstance(coeff, (int, float, complex))
        
        # Check approximation quality
        target_norm = l2_optics.l2_norm_function(target)
        error_norm = l2_optics.l2_norm_function(error)
        relative_error = error_norm / target_norm
        
        # Relative error should be reasonable (less than 50%)
        assert relative_error < 0.5
        
        # Error should be smaller than target
        assert error_norm < target_norm
    
    def test_convergence_analysis(self, l2_optics):
        """Test convergence analysis of function approximations."""
        errors, num_basis_functions = l2_optics.convergence_analysis()
        
        # Check that errors is a list of numbers
        assert isinstance(errors, list)
        assert isinstance(num_basis_functions, list)
        assert len(errors) == len(num_basis_functions)
        
        # Check that errors are decreasing (convergence)
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i-1]  # Errors should be non-increasing
        
        # Check that number of basis functions is increasing
        for i in range(1, len(num_basis_functions)):
            assert num_basis_functions[i] > num_basis_functions[i-1]
        
        # Final error should be smaller than initial error
        assert errors[-1] < errors[0]
    
    @pytest.mark.visualization
    def test_visualization_methods(self, l2_optics, mock_matplotlib_show):
        """Test that visualization methods can be called without errors."""
        try:
            # Create some test data for visualization
            target = np.exp(-(l2_optics.X**2 + l2_optics.Y**2) / 9)
            approximation = target * 0.9  # Simple approximation
            error = target - approximation
            
            l2_optics.visualize_approximation(target, approximation, error)
            # If we get here, the method ran successfully
            assert True
        except Exception as e:
            pytest.fail(f"Visualization method failed: {e}")


class TestMathematicalProperties:
    """Test mathematical properties of L2 space operations."""
    
    def test_l2_norm_properties(self, l2_optics):
        """Test mathematical properties of L2 norm."""
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        g = np.exp(-2 * (l2_optics.X**2 + l2_optics.Y**2))
        
        # Test positive definiteness: ||f|| ≥ 0 and ||f|| = 0 ⟺ f = 0
        norm_f = l2_optics.l2_norm_function(f)
        assert norm_f >= 0
        
        zero_function = np.zeros_like(f)
        zero_norm = l2_optics.l2_norm_function(zero_function)
        assert zero_norm == 0
        
        # Test triangle inequality: ||f + g|| ≤ ||f|| + ||g||
        norm_f_plus_g = l2_optics.l2_norm_function(f + g)
        norm_f = l2_optics.l2_norm_function(f)
        norm_g = l2_optics.l2_norm_function(g)
        assert norm_f_plus_g <= norm_f + norm_g + 1e-10
    
    def test_inner_product_linearity(self, l2_optics):
        """Test linearity properties of inner product."""
        f = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))
        g = np.exp(-2 * (l2_optics.X**2 + l2_optics.Y**2))
        h = np.exp(-3 * (l2_optics.X**2 + l2_optics.Y**2))
        
        alpha, beta = 2.0, 3.0
        
        # Test linearity in first argument: <αf + βg, h> = α<f, h> + β<g, h>
        left_side = l2_optics.l2_inner_product(alpha * f + beta * g, h)
        right_side = alpha * l2_optics.l2_inner_product(f, h) + beta * l2_optics.l2_inner_product(g, h)
        assert abs(left_side - right_side) < 1e-10
        
        # Test conjugate symmetry: <f, g> = <g, f>*
        inner_fg = l2_optics.l2_inner_product(f, g)
        inner_gf = l2_optics.l2_inner_product(g, f)
        assert abs(inner_fg - np.conj(inner_gf)) < 1e-10
    
    def test_orthogonality_condition(self, l2_optics):
        """Test orthogonality condition: <f, g> = 0 for orthogonal functions."""
        # Create orthogonal functions (odd and even functions)
        f = l2_optics.X * np.exp(-(l2_optics.X**2 + l2_optics.Y**2))  # Odd in x
        g = np.exp(-(l2_optics.X**2 + l2_optics.Y**2))  # Even in x
        
        # These should be approximately orthogonal
        inner_product = l2_optics.l2_inner_product(f, g)
        assert abs(inner_product) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])