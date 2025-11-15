import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from chapter04_function_spaces.zernike_polynomials import ZernikePolynomials

class TestZernikePolynomials:
    @pytest.fixture
    def zernike_analyzer(self):
        return ZernikePolynomials(max_n=4, grid_size=50)
    
    def test_initialization(self, zernike_analyzer):
        assert zernike_analyzer.max_n == 4
        assert zernike_analyzer.grid_size == 50
        assert hasattr(zernike_analyzer, 'x')
        assert hasattr(zernike_analyzer, 'y')
        assert hasattr(zernike_analyzer, 'r')
        assert hasattr(zernike_analyzer, 'theta')
        assert hasattr(zernike_analyzer, 'mask')
        
        assert zernike_analyzer.x.shape == (50, 50)
        assert zernike_analyzer.mask[25, 25] == True
    
    def test_zernike_radial_basic(self, zernike_analyzer):
        radial_00 = zernike_analyzer.zernike_radial(0, 0, zernike_analyzer.r)
        assert radial_00.shape == zernike_analyzer.r.shape
        assert np.allclose(radial_00, 1.0)
        
        radial_20 = zernike_analyzer.zernike_radial(2, 0, zernike_analyzer.r)
        expected_20 = 2 * zernike_analyzer.r**2 - 1
        assert np.allclose(radial_20, expected_20)
    
    def test_zernike_polynomial_basic(self, zernike_analyzer):
        z_00 = zernike_analyzer.zernike_polynomial(0, 0)
        assert z_00.shape == zernike_analyzer.r.shape
        assert np.allclose(z_00[zernike_analyzer.mask], 1.0)
    
    def test_zernike_basis_generation(self, zernike_analyzer):
        basis = zernike_analyzer.zernike_basis()
        assert isinstance(basis, dict)
        assert len(basis) > 0
        
        for name, polynomial in basis.items():
            assert polynomial.shape == (50, 50)
            assert name.startswith('Z_')
    
    def test_wavefront_decomposition_basic(self, zernike_analyzer):
        test_wavefront = 2 * zernike_analyzer.r**2 - 1
        coefficients = zernike_analyzer.wavefront_decomposition(test_wavefront)
        
        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0
        
        for name, coeff in coefficients.items():
            assert isinstance(coeff, (int, float, np.number))
            assert not np.isnan(coeff)
            assert not np.isinf(coeff)
    
    def test_reconstruct_wavefront_basic(self, zernike_analyzer):
        original = zernike_analyzer.generate_test_wavefront('defocus', amplitude=1.0)
        coefficients = zernike_analyzer.wavefront_decomposition(original)
        reconstructed = zernike_analyzer.reconstruct_wavefront(coefficients)
        
        assert reconstructed.shape == original.shape
        assert np.allclose(reconstructed, reconstructed * zernike_analyzer.mask)
        
        error = original - reconstructed
        rms_error = np.sqrt(np.mean(error**2))
        assert rms_error < 1e-10
    
    def test_analyze_optical_aberrations(self, zernike_analyzer):
        test_wavefront = zernike_analyzer.generate_test_wavefront('mixed', amplitude=1.0)
        aberrations = zernike_analyzer.analyze_optical_aberrations(test_wavefront)
        
        assert isinstance(aberrations, dict)
        expected_keys = ['piston', 'tilt_x', 'tilt_y', 'defocus', 'astigmatism_0', 
                        'astigmatism_45', 'coma_x', 'coma_y', 'spherical', 'trefoil_x', 'trefoil_y']
        
        for key in expected_keys:
            assert key in aberrations
            assert isinstance(aberrations[key], (int, float, np.number))
    
    def test_generate_test_wavefront_varieties(self, zernike_analyzer):
        test_types = ['defocus', 'astigmatism', 'coma', 'spherical', 'mixed']
        
        for test_type in test_types:
            wavefront = zernike_analyzer.generate_test_wavefront(test_type, amplitude=1.0)
            
            assert wavefront.shape == zernike_analyzer.r.shape
            assert np.allclose(wavefront, wavefront * zernike_analyzer.mask)
            
            if test_type != 'invalid':
                assert not np.allclose(wavefront, 0.0)
        
        invalid_wavefront = zernike_analyzer.generate_test_wavefront('invalid', amplitude=1.0)
        assert np.allclose(invalid_wavefront, 0.0)
    
    def test_orthogonality_check(self, zernike_analyzer):
        orthogonality = zernike_analyzer.orthogonality_check()
        
        assert isinstance(orthogonality, dict)
        assert 'inner_product_matrix' in orthogonality
        assert 'max_off_diagonal' in orthogonality
        assert 'is_orthogonal' in orthogonality
        
        matrix = orthogonality['inner_product_matrix']
        assert matrix.shape[0] == matrix.shape[1]
        assert np.allclose(matrix, matrix.T)
        
        max_off_diag = orthogonality['max_off_diagonal']
        assert isinstance(max_off_diag, (int, float, np.number))
        assert max_off_diag >= 0
        assert max_off_diag < 1e-6
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_zernike_analysis(self, mock_figure, mock_show):
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_ax.imshow.return_value = MagicMock()
        
        from chapter04_function_spaces.zernike_polynomials import demonstrate_zernike_analysis
        demonstrate_zernike_analysis()
        mock_show.assert_called()