"""
Test suite for SimpleZernike implementation in chapter04_function_spaces.
Tests simplified Zernike polynomial analysis for optical wavefronts.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter04_function_spaces.zernike_simple import SimpleZernike


class TestSimpleZernike:
    """Test suite for SimpleZernike class."""
    
    @pytest.fixture
    def zernike(self):
        """Create a SimpleZernike instance for testing."""
        return SimpleZernike(n_points=50)
    
    def test_initialization(self, zernike):
        """Test proper initialization of SimpleZernike."""
        assert zernike.n_points == 50
        assert hasattr(zernike, 'X')
        assert hasattr(zernike, 'Y')
        assert hasattr(zernike, 'r')
        assert hasattr(zernike, 'theta')
        assert hasattr(zernike, 'mask')
        
        # Check coordinate system properties
        assert zernike.X.shape == (50, 50)
        assert zernike.Y.shape == (50, 50)
        assert zernike.r.shape == (50, 50)
        assert zernike.theta.shape == (50, 50)
        assert zernike.mask.shape == (50, 50)
        
        # Check that mask properly filters unit disk
        assert np.sum(zernike.mask) < 50 * 50  # Not all points in unit disk
        assert zernike.mask[25, 25] == True  # Center should be included
    
    def test_zernike_basic_polynomials(self, zernike):
        """Test computation of basic Zernike polynomials."""
        # Test n=0, m=0 (piston)
        z_00 = zernike.zernike(0, 0)
        assert z_00.shape == (50, 50)
        assert not np.allclose(z_00, 0.0)  # Should not be all zeros
        
        # Test n=1, m=1 (tilt x)
        z_11 = zernike.zernike(1, 1)
        assert z_11.shape == (50, 50)
        assert not np.allclose(z_11, 0.0)
        
        # Test n=2, m=0 (defocus)
        z_20 = zernike.zernike(2, 0)
        assert z_20.shape == (50, 50)
        assert not np.allclose(z_20, 0.0)
    
    def test_zernike_invalid_cases(self, zernike):
        """Test Zernike polynomial computation for invalid cases."""
        # Test |m| > n (should return zeros)
        z_invalid = zernike.zernike(2, 3)
        assert np.allclose(z_invalid, 0.0)
        
        # Test n-m odd (should return zeros)
        z_invalid2 = zernike.zernike(3, 0)
        assert np.allclose(z_invalid2, 0.0)
        
        # Test negative m with |m| > n (should return zeros)
        z_invalid3 = zernike.zernike(1, -2)
        assert np.allclose(z_invalid3, 0.0)
    
    def test_zernike_negative_m(self, zernike):
        """Test Zernike polynomials with negative m values."""
        # Test n=1, m=-1 (tilt y)
        z_1_neg1 = zernike.zernike(1, -1)
        assert z_1_neg1.shape == (50, 50)
        assert not np.allclose(z_1_neg1, 0.0)
        
        # Test n=2, m=-2 (astigmatism 45°)
        z_2_neg2 = zernike.zernike(2, -2)
        assert z_2_neg2.shape == (50, 50)
        assert not np.allclose(z_2_neg2, 0.0)
    
    def test_analyze_wavefront_basic(self, zernike):
        """Test basic wavefront analysis functionality."""
        # Create a simple defocus wavefront
        defocus_wavefront = 2 * zernike.r**2 - 1
        
        coefficients = zernike.analyze_wavefront(defocus_wavefront)
        
        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0
        
        # Check that coefficients are finite numbers
        for name, coeff in coefficients.items():
            assert isinstance(coeff, (int, float, np.number))
            assert np.isfinite(coeff)
            assert not np.isnan(coeff)
    
    def test_analyze_wavefront_piston(self, zernike):
        """Test analysis of piston wavefront."""
        # Create constant (piston) wavefront
        piston_wavefront = np.ones_like(zernike.r)
        
        coefficients = zernike.analyze_wavefront(piston_wavefront)
        
        # Should find significant piston coefficient
        piston_coeff = coefficients.get("Z_0,0", 0.0)
        assert abs(piston_coeff) > 0.1  # Should be significant
    
    def test_analyze_wavefront_tilt(self, zernike):
        """Test analysis of tilt wavefront."""
        # Create tilt wavefront (linear in x)
        tilt_wavefront = zernike.X  # Linear in x-coordinate
        
        coefficients = zernike.analyze_wavefront(tilt_wavefront)
        
        # Should find significant tilt coefficients
        tilt_x_coeff = coefficients.get("Z_1,1", 0.0)
        assert abs(tilt_x_coeff) > 0.05  # Should be significant
    
    def test_analyze_wavefront_defocus(self, zernike):
        """Test analysis of defocus wavefront."""
        # Create defocus wavefront (quadratic in r)
        defocus_wavefront = 2 * zernike.r**2 - 1
        
        coefficients = zernike.analyze_wavefront(defocus_wavefront)
        
        # Should find significant defocus coefficient
        defocus_coeff = coefficients.get("Z_2,0", 0.0)
        assert abs(defocus_coeff) > 0.1  # Should be significant
    
    def test_analyze_wavefront_astigmatism(self, zernike):
        """Test analysis of astigmatism wavefront."""
        # Create astigmatism wavefront (cos(2θ) dependence)
        astigmatism_wavefront = zernike.r**2 * np.cos(2 * zernike.theta)
        
        coefficients = zernike.analyze_wavefront(astigmatism_wavefront)
        
        # Should find significant astigmatism coefficient
        astigmatism_coeff = coefficients.get("Z_2,2", 0.0)
        assert abs(astigmatism_coeff) > 0.05  # Should be significant
    
    def test_analyze_wavefront_zero(self, zernike):
        """Test analysis of zero wavefront."""
        # Create zero wavefront
        zero_wavefront = np.zeros_like(zernike.r)
        
        coefficients = zernike.analyze_wavefront(zero_wavefront)
        
        # All coefficients should be very small
        for name, coeff in coefficients.items():
            assert abs(coeff) < 1e-10  # Should be very close to zero
    
    def test_different_grid_sizes(self):
        """Test SimpleZernike with different grid sizes."""
        for n_points in [20, 50, 100]:
            zernike = SimpleZernike(n_points=n_points)
            
            assert zernike.n_points == n_points
            assert zernike.X.shape == (n_points, n_points)
            assert zernike.mask.shape == (n_points, n_points)
            
            # Test basic functionality
            z_00 = zernike.zernike(0, 0)
            assert z_00.shape == (n_points, n_points)
            
            # Test wavefront analysis
            defocus = 2 * zernike.r**2 - 1
            coefficients = zernike.analyze_wavefront(defocus)
            assert len(coefficients) > 0
    
    def test_coefficient_consistency(self, zernike):
        """Test consistency of coefficient computation."""
        # Create a known wavefront
        test_wavefront = 0.5 * (2 * zernike.r**2 - 1) + 0.3 * zernike.r**2 * np.cos(2 * zernike.theta)
        
        # Analyze multiple times
        coeffs1 = zernike.analyze_wavefront(test_wavefront)
        coeffs2 = zernike.analyze_wavefront(test_wavefront)
        
        # Should get identical results
        for key in coeffs1:
            assert key in coeffs2
            assert abs(coeffs1[key] - coeffs2[key]) < 1e-12
    
    def test_polynomial_orthogonality_approximate(self, zernike):
        """Test approximate orthogonality of Zernike polynomials."""
        # Test a few polynomial pairs
        z_00 = zernike.zernike(0, 0)
        z_11 = zernike.zernike(1, 1)
        z_20 = zernike.zernike(2, 0)
        
        # Compute inner products (should be approximately orthogonal)
        ip_00_11 = np.sum(z_00 * z_11) / np.sum(zernike.mask)
        ip_00_20 = np.sum(z_00 * z_20) / np.sum(zernike.mask)
        ip_11_20 = np.sum(z_11 * z_20) / np.sum(zernike.mask)
        
        # Should be close to zero (orthogonal)
        assert abs(ip_00_11) < 1e-6
        assert abs(ip_00_20) < 1e-6
        assert abs(ip_11_20) < 1e-6
    
    def test_boundary_conditions(self, zernike):
        """Test behavior at boundary of unit disk."""
        # Test that polynomials are zero outside unit disk
        z_00 = zernike.zernike(0, 0)
        
        # Points outside unit disk should be masked (zero)
        outside_mask = ~zernike.mask
        if np.any(outside_mask):
            assert np.allclose(z_00[outside_mask], 0.0)
    
    def test_radial_polynomials(self, zernike):
        """Test specific radial polynomial properties."""
        # Test that radial polynomials have correct symmetry
        z_20 = zernike.zernike(2, 0)  # Defocus
        z_40 = zernike.zernike(4, 0)  # Spherical
        
        # Should be radially symmetric (independent of theta)
        # Check a few radial positions
        center_idx = zernike.n_points // 2
        
        # At center, should have specific values
        center_val_20 = z_20[center_idx, center_idx]
        center_val_40 = z_40[center_idx, center_idx]
        
        # These should be finite values
        assert np.isfinite(center_val_20)
        assert np.isfinite(center_val_40)
        
        # At edge (r=1), should be consistent with theory
        # Find point at edge of unit disk
        edge_positions = np.where(np.abs(zernike.r - 1.0) < 0.02)
        if len(edge_positions[0]) > 0:
            edge_idx = (edge_positions[0][0], edge_positions[1][0])
            edge_val_20 = z_20[edge_idx]
            edge_val_40 = z_40[edge_idx]
            
            assert np.isfinite(edge_val_20)
            assert np.isfinite(edge_val_40)
    
    def test_angular_polynomials(self, zernike):
        """Test specific angular polynomial properties."""
        # Test angular dependence
        z_11 = zernike.zernike(1, 1)  # Tilt x
        z_1_neg1 = zernike.zernike(1, -1)  # Tilt y
        
        # Should have correct angular symmetry
        # At theta=0 (positive x-axis)
        center_idx = zernike.n_points // 2
        right_idx = (center_idx, center_idx + 10)  # Move right along x-axis
        
        if right_idx[1] < zernike.n_points and zernike.mask[right_idx]:
            val_11_right = z_11[right_idx]
            val_1_neg1_right = z_1_neg1[right_idx]
            
            # Should be finite values
            assert np.isfinite(val_11_right)
            assert np.isfinite(val_1_neg1_right)
        
        # At theta=π/2 (positive y-axis)
        top_idx = (center_idx - 10, center_idx)  # Move up along y-axis
        
        if top_idx[0] >= 0 and zernike.mask[top_idx]:
            val_11_top = z_11[top_idx]
            val_1_neg1_top = z_1_neg1[top_idx]
            
            # Should be finite values
            assert np.isfinite(val_11_top)
            assert np.isfinite(val_1_neg1_top)