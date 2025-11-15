"""
Test suite for Zernike Polynomials implementation in chapter04_function_spaces.
Tests orthogonal function bases in circular domains for optical wavefront analysis.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter04_function_spaces.zernike_polynomials import ZernikePolynomials


class TestZernikePolynomials:
    """Test suite for ZernikePolynomials class."""
    
    @pytest.fixture
    def zernike_analyzer(self):
        """Create a ZernikePolynomials instance for testing."""
        return ZernikePolynomials(max_n=4, grid_size=50)
    
    def test_initialization(self, zernike_analyzer):
        """Test proper initialization of ZernikePolynomials."""
        assert zernike_analyzer.max_n == 4
        assert zernike_analyzer.grid_size == 50
        assert hasattr(zernike_analyzer, 'x')
        assert hasattr(zernike_analyzer, 'y')
        assert hasattr(zernike_analyzer, 'r')
        assert hasattr(zernike_analyzer, 'theta')
        assert hasattr(zernike_analyzer, 'mask')
        
        # Check coordinate system properties
        assert zernike_analyzer.x.shape == (50, 50)
        assert zernike_analyzer.y.shape == (50, 50)
        assert zernike_analyzer.r.shape == (50, 50)
        assert zernike_analyzer.theta.shape == (50, 50)
        assert zernike_analyzer.mask.shape == (50, 50)
        
        # Check that mask properly filters unit disk
        assert np.sum(zernike_analyzer.mask) < 50 * 50  # Not all points in unit disk
        assert zernike_analyzer.mask[25, 25] == True  # Center should be included
    
    def test_zernike_radial_basic(self, zernike_analyzer):
        """Test radial component computation for basic cases."""
        # Test n=0, m=0 (piston)
        radial_00 = zernike_analyzer.zernike_radial(0, 0, zernike_analyzer.r)
        assert radial_00.shape == zernike_analyzer.r.shape
        assert np.allclose(radial_00, 1.0)  # Z_0^0 = 1
        
        # Test n=2, m=0 (defocus)
        radial_20 = zernike_analyzer.zernike_radial(2, 0, zernike_analyzer.r)
        expected_20 = 2 * zernike_analyzer.r**2 - 1
        assert np.allclose(radial_20, expected_20)
        
        # Test invalid case: n-m odd
        radial_invalid = zernike_analyzer.zernike_radial(3, 0, zernike_analyzer.r)
        assert np.allclose(radial_invalid, 0.0)
    
    def test_zernike_polynomial_basic(self, zernike_analyzer):
        """Test full Zernike polynomial computation."""
        # Test n=0, m=0 (piston)
        z_00 = zernike_analyzer.zernike_polynomial(0, 0)
        assert z_00.shape == zernike_analyzer.r.shape
        assert np.allclose(z_00[zernike_analyzer.mask], 1.0)
        
        # Test n=1, m=1 (tilt x)
        z_11 = zernike_analyzer.zernike_polynomial(1, 1)
        expected_11 = zernike_analyzer.r * np.cos(zernike_analyzer.theta) * zernike_analyzer.mask
        assert np.allclose(z_11, expected_11)
        
        # Test n=1, m=-1 (tilt y)
        z_1_neg1 = zernike_analyzer.zernike_polynomial(1, -1)
        expected_1_neg1 = zernike_analyzer.r * np.sin(zernike_analyzer.theta) * zernike_analyzer.mask
        assert np.allclose(z_1_neg1, expected_1_neg1)
    
    def test_zernike_basis_generation(self, zernike_analyzer):
        """Test generation of complete Zernike basis."""
        basis = zernike_analyzer.zernike_basis()
        
        assert isinstance(basis, dict)
        assert len(basis) > 0
        
        # Check that all basis functions are properly shaped
        for name, polynomial in basis.items():
            assert polynomial.shape == (50, 50)
            assert name.startswith('Z_')
            assert '(n=' in name and 'm=' in name
        
        # Check that basis includes expected terms
        basis_names = list(basis.keys())
        assert any('n=0, m=0' in name for name in basis_names)
        assert any('n=1, m=1' in name for name in basis_names)
        assert any('n=2, m=0' in name for name in basis_names)
    
    def test_wavefront_decomposition_basic(self, zernike_analyzer):
        """Test wavefront decomposition into Zernike coefficients."""
        # Create a simple test wavefront (defocus)
        test_wavefront = 2 * zernike_analyzer.r**2 - 1
        
        coefficients = zernike_analyzer.wavefront_decomposition(test_wavefront)
        
        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0
        
        # Check that coefficients are reasonable values
        for name, coeff in coefficients.items():
            assert isinstance(coeff, (int, float, np.number))
            assert not np.isnan(coeff)
            assert not np.isinf(coeff)
    
    def test_reconstruct_wavefront_basic(self, zernike_analyzer):
        """Test wavefront reconstruction from coefficients."""
        # Create original wavefront
        original = zernike_analyzer.generate_test_wavefront('defocus', amplitude=1.0)
        
        # Decompose into coefficients
        coefficients = zernike_analyzer.wavefront_decomposition(original)
        
        # Reconstruct wavefront
        reconstructed = zernike_analyzer.reconstruct_wavefront(coefficients)
        
        assert reconstructed.shape == original.shape
        assert np.allclose(reconstructed, reconstructed * zernike_analyzer.mask)
        
        # Check reconstruction error
        error = original - reconstructed
        rms_error = np.sqrt(np.mean(error**2))
        assert rms_error < 1e-10  # Should be very small for perfect reconstruction
    
    def test_analyze_optical_aberrations(self, zernike_analyzer):
        """Test optical aberration analysis."""
        # Generate test wavefront with known aberrations
        test_wavefront = zernike_analyzer.generate_test_wavefront('mixed', amplitude=1.0)
        
        aberrations = zernike_analyzer.analyze_optical_aberrations(test_wavefront)
        
        assert isinstance(aberrations, dict)
        expected_keys = ['piston', 'tilt_x', 'tilt_y', 'defocus', 'astigmatism_0', 
                        'astigmatism_45', 'coma_x', 'coma_y', 'spherical', 'trefoil_x', 'trefoil_y']
        
        for key in expected_keys:
            assert key in aberrations
            assert isinstance(aberrations[key], (int, float, np.number))
            assert not np.isnan(aberrations[key])
            assert not np.isinf(aberrations[key])
    
    def test_generate_test_wavefront_varieties(self, zernike_analyzer):
        """Test generation of different types of test wavefronts."""
        test_types = ['defocus', 'astigmatism', 'coma', 'spherical', 'mixed']
        
        for test_type in test_types:
            wavefront = zernike_analyzer.generate_test_wavefront(test_type, amplitude=1.0)
            
            assert wavefront.shape == zernike_analyzer.r.shape
            assert np.allclose(wavefront, wavefront * zernike_analyzer.mask)
            
            # Check that wavefront is not all zeros (for valid types)
            if test_type != 'invalid':
                assert not np.allclose(wavefront, 0.0)
        
        # Test invalid type
        invalid_wavefront = zernike_analyzer.generate_test_wavefront('invalid', amplitude=1.0)
        assert np.allclose(invalid_wavefront, 0.0)
    
    def test_orthogonality_check(self, zernike_analyzer):
        """Test orthogonality verification of Zernike basis."""
        orthogonality = zernike_analyzer.orthogonality_check()
        
        assert isinstance(orthogonality, dict)
        assert 'inner_product_matrix' in orthogonality
        assert 'max_off_diagonal' in orthogonality
        assert 'is_orthogonal' in orthogonality
        
        # Check matrix properties
        matrix = orthogonality['inner_product_matrix']
        assert matrix.shape[0] == matrix.shape[1]
        assert np.allclose(matrix, matrix.T)  # Should be symmetric
        
        # Check orthogonality measure
        max_off_diag = orthogonality['max_off_diagonal']
        assert isinstance(max_off_diag, (int, float, np.number))
        assert max_off_diag >= 0
        
        # For numerical precision, should be approximately orthogonal
        assert max_off_diag < 1e-6  # Reasonable tolerance for discrete grid
    
    def test_edge_cases_and_boundary_conditions(self, zernike_analyzer):
        """Test edge cases and boundary conditions."""
        # Test with zero amplitude wavefront
        zero_wavefront = np.zeros_like(zernike_analyzer.r)
        zero_coeffs = zernike_analyzer.wavefront_decomposition(zero_wavefront)
        
        for coeff in zero_coeffs.values():
            assert abs(coeff) < 1e-10  # Should be very close to zero
        
        # Test with very small grid
        small_zernike = ZernikePolynomials(max_n=2, grid_size=10)
        assert small_zernike.grid_size == 10
        
        # Test reconstruction with limited terms
        test_wavefront = zernike_analyzer.generate_test_wavefront('defocus')
        coefficients = zernike_analyzer.wavefront_decomposition(test_wavefront)
        
        # Reconstruct with only first few terms
        limited_reconstruction = zernike_analyzer.reconstruct_wavefront(coefficients, max_terms=3)
        assert limited_reconstruction.shape == test_wavefront.shape
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_zernike_analysis(self, mock_figure, mock_show, zernike_analyzer):
        """Test the demonstration function with mocked visualization."""
        # Mock the figure and its methods
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock imshow to avoid actual plotting
        mock_ax.imshow.return_value = MagicMock()
        mock_ax.set_title.return_value = None
        mock_ax.set_xlabel.return_value = None
        mock_ax.set_ylabel.return_value = None
        
        # Import and run the demonstration
        from chapter04_function_spaces.zernike_polynomials import demonstrate_zernike_analysis
        
        # This should run without errors
        demonstrate_zernike_analysis()
        
        # Verify that visualization methods were called
        mock_show.assert_called()
    
    def test_mathematical_properties(self, zernike_analyzer):
        """Test mathematical properties of Zernike polynomials."""
        # Test that Z_0^0 = 1 everywhere in unit disk
        z_00 = zernike_analyzer.zernike_polynomial(0, 0)
        assert np.allclose(z_00[zernike_analyzer.mask], 1.0)
        
        # Test normalization: integral of (Z_n^m)^2 over unit disk should be π/(n+1)
        # (This is approximate due to discrete grid)
        z_20 = zernike_analyzer.zernike_polynomial(2, 0)
        integral = np.sum(z_20**2) / np.sum(zernike_analyzer.mask)
        expected_integral = np.pi / 3  # For n=2
        
        # Allow for numerical error due to discrete grid
        assert abs(integral - expected_integral) / expected_integral < 0.1
    
    def test_function_space_properties(self, zernike_analyzer):
        """Test that Zernike polynomials form a proper function space basis."""
        basis = zernike_analyzer.zernike_basis()
        
        # Test linear independence (approximate)
        # Create random linear combination
        coeffs = np.random.randn(len(basis))
        linear_combination = np.zeros_like(zernike_analyzer.r)
        
        for i, (name, polynomial) in enumerate(basis.items()):
            linear_combination += coeffs[i] * polynomial
        
        # If basis is linearly independent, non-zero coefficients should give non-zero result
        assert not np.allclose(linear_combination, 0.0)
        
        # Test completeness by decomposing and reconstructing a complex function
        complex_wavefront = (zernike_analyzer.r**3 * np.cos(3 * zernike_analyzer.theta) + 
                           0.5 * zernike_analyzer.r**2 * np.sin(2 * zernike_analyzer.theta))
        
        coefficients = zernike_analyzer.wavefront_decomposition(complex_wavefront)
        reconstructed = zernike_analyzer.reconstruct_wavefront(coefficients)
        
        # Should be able to approximate the function
        error = complex_wavefront - reconstructed
        relative_error = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(complex_wavefront**2))
        assert relative_error < 0.5  # Reasonable approximation
    
    def test_different_grid_sizes(self):
        """Test Zernike polynomials with different grid sizes."""
        for grid_size in [20, 50, 100]:
            zernike = ZernikePolynomials(max_n=3, grid_size=grid_size)
            
            # Test basic functionality
            z_00 = zernike.zernike_polynomial(0, 0)
            assert z_00.shape == (grid_size, grid_size)
            
            # Test wavefront analysis
            test_wavefront = zernike.generate_test_wavefront('defocus')
            coefficients = zernike.wavefront_decomposition(test_wavefront)
            assert len(coefficients) > 0
    
    def test_different_max_n_values(self):
        """Test Zernike polynomials with different maximum n values."""
        for max_n in [2, 4, 6]:
            zernike = ZernikePolynomials(max_n=max_n, grid_size=30)
            
            basis = zernike.zernike_basis()
            
            # Check that basis includes polynomials up to max_n
            basis_names = list(basis.keys())
            max_n_found = max([int(name.split('n=')[1].split(',')[0]) for name in basis_names])
            assert max_n_found <= max_n
            
            # Test orthogonality
            orthogonality = zernike.orthogonality_check()
            assert orthogonality['max_off_diagonal'] < 1e-5
    
    def test_aberration_amplitude_scaling(self, zernike_analyzer):
        """Test that aberration amplitudes scale linearly."""
        amplitudes = [0.5, 1.0, 2.0]
        defocus_coeffs = []
        
        for amp in amplitudes:
            wavefront = zernike_analyzer.generate_test_wavefront('defocus', amplitude=amp)
            aberrations = zernike_analyzer.analyze_optical_aberrations(wavefront)
            defocus_coeffs.append(aberrations['defocus'])
        
        # Check linear scaling (approximately)
        ratios = [defocus_coeffs[i] / defocus_coeffs[0] for i in range(1, len(defocus_coeffs))]
        expected_ratios = [amplitudes[i] / amplitudes[0] for i in range(1, len(amplitudes))]
        
        for ratio, expected in zip(ratios, expected_ratios):
            assert abs(ratio - expected) / expected < 0.1  # 10% tolerance
    
    def test_mixed_aberrations_consistency(self, zernike_analyzer):
        """Test that mixed aberrations are consistent with individual aberrations."""
        # Create individual aberrations
        defocus = zernike_analyzer.generate_test_wavefront('defocus', amplitude=0.5)
        astigmatism = zernike_analyzer.generate_test_wavefront('astigmatism', amplitude=0.3)
        coma = zernike_analyzer.generate_test_wavefront('coma', amplitude=0.2)
        
        # Create mixed aberration
        mixed = zernike_analyzer.generate_test_wavefront('mixed', amplitude=1.0)
        
        # Analyze all
        defocus_aberr = zernike_analyzer.analyze_optical_aberrations(defocus)
        astigmatism_aberr = zernike_analyzer.analyze_optical_aberrations(astigmatism)
        coma_aberr = zernike_analyzer.analyze_optical_aberrations(coma)
        mixed_aberr = zernike_analyzer.analyze_optical_aberrations(mixed)
        
        # Check that mixed aberration contains contributions from components
        assert abs(mixed_aberr['defocus']) > 0.1  # Should have significant defocus
        assert abs(mixed_aberr['astigmatism_0']) > 0.05  # Should have astigmatism
        assert abs(mixed_aberr['coma_x']) > 0.05  # Should have coma