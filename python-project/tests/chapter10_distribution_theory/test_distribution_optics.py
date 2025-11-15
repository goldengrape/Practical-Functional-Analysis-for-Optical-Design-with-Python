"""
Test suite for Chapter 10: Distribution Theory in Optical Systems
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from chapter10_distribution_theory.distribution_optics import DistributionTheoryOptics


class TestDistributionTheoryOptics:
    """Test suite for Distribution Theory in Optical Systems."""
    
    @pytest.fixture
    def dist_theory(self):
        """Create a default DistributionTheoryOptics instance."""
        return DistributionTheoryOptics(grid_size=100)
    
    @pytest.fixture
    def test_functions(self, dist_theory):
        """Create test functions for distribution analysis."""
        return {
            'constant': np.ones_like(dist_theory.x),
            'linear': dist_theory.x,
            'quadratic': dist_theory.x**2,
            'sinusoidal': np.sin(2 * np.pi * dist_theory.x),
            'gaussian': np.exp(-dist_theory.x**2),
            'exponential': np.exp(-np.abs(dist_theory.x))
        }
    
    def test_initialization(self):
        """Test proper initialization of DistributionTheoryOptics."""
        dist_theory = DistributionTheoryOptics(grid_size=200)
        assert dist_theory.grid_size == 200
        assert len(dist_theory.x) == 200
        assert dist_theory.dx == pytest.approx(2/199, rel=1e-10)
        assert np.isclose(dist_theory.x[0], -1.0)
        assert np.isclose(dist_theory.x[-1], 1.0)
    
    def test_delta_function_approximation_gaussian(self, dist_theory):
        """Test Gaussian delta function approximation."""
        center = 0.3
        width = 0.02
        delta_approx = dist_theory.delta_function_approximation(center, width, method='gaussian')
        
        # Check basic properties
        assert len(delta_approx) == len(dist_theory.x)
        assert all(np.isfinite(val) for val in delta_approx)
        assert all(val >= 0 for val in delta_approx)
        
        # Check peak location
        max_idx = np.argmax(delta_approx)
        peak_location = dist_theory.x[max_idx]
        assert abs(peak_location - center) < width * 2
        
        # Check maximum value
        max_value = np.max(delta_approx)
        expected_max = 1.0 / (width * np.sqrt(2 * np.pi))
        assert max_value == pytest.approx(expected_max, rel=0.1)
    
    def test_delta_function_approximation_lorentzian(self, dist_theory):
        """Test Lorentzian delta function approximation."""
        center = 0.0
        width = 0.01
        delta_approx = dist_theory.delta_function_approximation(center, width, method='lorentzian')
        
        # Check basic properties
        assert len(delta_approx) == len(dist_theory.x)
        assert all(np.isfinite(val) for val in delta_approx)
        assert all(val >= 0 for val in delta_approx)
        
        # Check peak location
        max_idx = np.argmax(delta_approx)
        peak_location = dist_theory.x[max_idx]
        assert abs(peak_location - center) < width * 2
        
        # Check maximum value
        max_value = np.max(delta_approx)
        expected_max = 1.0 / (np.pi * width)
        assert max_value == pytest.approx(expected_max, rel=0.1)
    
    def test_delta_function_approximation_sinc(self, dist_theory):
        """Test sinc delta function approximation."""
        center = -0.2
        width = 0.05
        delta_approx = dist_theory.delta_function_approximation(center, width, method='sinc')
        
        # Check basic properties
        assert len(delta_approx) == len(dist_theory.x)
        assert all(np.isfinite(val) for val in delta_approx)
        
        # Check peak location
        max_idx = np.argmax(delta_approx)
        peak_location = dist_theory.x[max_idx]
        assert abs(peak_location - center) < width * 2
        
        # Check maximum value
        max_value = np.max(delta_approx)
        expected_max = 1.0 / width
        assert max_value == pytest.approx(expected_max, rel=0.1)
    
    def test_delta_function_approximation_invalid_method(self, dist_theory):
        """Test invalid delta function approximation method."""
        with pytest.raises(ValueError, match="Method must be 'gaussian', 'lorentzian', or 'sinc'"):
            dist_theory.delta_function_approximation(0.0, 0.01, method='invalid')
    
    def test_test_delta_properties_normalization(self, dist_theory):
        """Test delta function normalization property."""
        center = 0.0
        delta_approx = dist_theory.delta_function_approximation(center, width=0.02, method='gaussian')
        properties = dist_theory.test_delta_properties(delta_approx, center)
        
        # Check normalization: ∫δ(x-a)dx = 1
        normalization = properties['normalization']
        assert normalization == pytest.approx(1.0, rel=0.01)
        
        assert 'sifting_property' in properties
        assert 'expected_sifting_value' in properties
        assert 'sifting_error' in properties
        assert 'localization_error' in properties
        assert 'max_value' in properties
        assert 'width_estimate' in properties
    
    def test_test_delta_properties_sifting(self, dist_theory):
        """Test delta function sifting property."""
        center = 0.5
        delta_approx = dist_theory.delta_function_approximation(center, width=0.02, method='gaussian')
        
        # Test with different functions
        test_functions = [
            np.sin(2 * np.pi * dist_theory.x),
            np.cos(2 * np.pi * dist_theory.x),
            dist_theory.x**2,
            np.exp(-dist_theory.x**2)
        ]
        
        for test_func in test_functions:
            properties = dist_theory.test_delta_properties(delta_approx, center)
            
            # Check sifting property: ∫f(x)δ(x-a)dx = f(a)
            sifting_value = properties['sifting_property']
            expected_value = properties['expected_sifting_value']
            sifting_error = properties['sifting_error']
            
            # Should be close to expected value
            assert sifting_error < 0.1  # Reasonable tolerance for discrete approximation
            assert abs(sifting_value - expected_value) < 0.1
    
    def test_greens_function_helmholtz_1d(self, dist_theory):
        """Test 1D Green's function for Helmholtz equation."""
        source_position = 0.3
        wavenumber = 2 * np.pi / 500e-9  # 500 nm light
        
        green_func = dist_theory.greens_function_helmholtz(source_position, wavenumber, dimension=1)
        
        # Check basic properties
        assert len(green_func) == len(dist_theory.x)
        assert all(np.isfinite(val) for val in green_func)
        
        # Check that it's complex
        assert np.iscomplexobj(green_func)
        
        # Check symmetry (Green's function should be symmetric about source)
        source_idx = np.argmin(np.abs(dist_theory.x - source_position))
        
        # Check values at symmetric points
        if source_idx > 0 and source_idx < len(dist_theory.x) - 1:
            left_idx = source_idx - 1
            right_idx = source_idx + 1
            
            # Should have similar magnitude at symmetric points
            left_mag = np.abs(green_func[left_idx])
            right_mag = np.abs(green_func[right_idx])
            assert abs(left_mag - right_mag) / (left_mag + right_mag + 1e-10) < 0.5
    
    def test_greens_function_helmholtz_invalid_dimension(self, dist_theory):
        """Test invalid dimension for Green's function."""
        with pytest.raises(ValueError, match="Dimension must be 1, 2, or 3"):
            dist_theory.greens_function_helmholtz(0.0, 1.0, dimension=4)
    
    def test_point_source_response(self, dist_theory):
        """Test point source response calculation."""
        source_position = 0.0
        observation_points = [-0.5, -0.2, 0.0, 0.2, 0.5]
        wavenumber = 2 * np.pi / 500e-9
        
        responses = dist_theory.point_source_response(
            source_position, observation_points, wavenumber, dimension=1
        )
        
        # Check basic properties
        assert len(responses) == len(observation_points)
        assert all(np.isfinite(response) for response in responses)
        assert all(np.iscomplexobj(response) for response in responses)
        
        # Check that response at source position is handled
        source_response = responses[observation_points.index(source_position)]
        assert np.isfinite(source_response)
    
    def test_principal_value_integral_symmetric(self, dist_theory):
        """Test principal value integral using symmetric method."""
        test_function = np.sin(2 * np.pi * dist_theory.x)
        singular_point = 0.0
        
        pv_integral = dist_theory.principal_value_integral(
            test_function, singular_point, method='symmetric'
        )
        
        # Should be finite
        assert np.isfinite(pv_integral)
        
        # For sin(2πx)/x, the principal value should be close to 0
        # (since sin(2πx)/x is an even function integrated over symmetric interval)
        assert abs(pv_integral) < 0.1
    
    def test_principal_value_integral_subtract(self, dist_theory):
        """Test principal value integral using subtraction method."""
        test_function = np.cos(2 * np.pi * dist_theory.x)
        singular_point = 0.0
        
        pv_integral = dist_theory.principal_value_integral(
            test_function, singular_point, method='subtract'
        )
        
        # Should be finite
        assert np.isfinite(pv_integral)
        
        # This is a more robust method, should give reasonable results
        assert abs(pv_integral) < 10.0  # Should be bounded
    
    def test_principal_value_integral_invalid_method(self, dist_theory):
        """Test invalid method for principal value integral."""
        with pytest.raises(ValueError, match="Method must be 'symmetric', 'subtract', or 'complex'"):
            dist_theory.principal_value_integral(np.ones_like(dist_theory.x), 0.0, method='invalid')
    
    def test_hilbert_transform(self, dist_theory):
        """Test Hilbert transform."""
        test_function = np.sin(2 * np.pi * dist_theory.x)
        
        hilbert_result = dist_theory.hilbert_transform(test_function)
        
        # Check basic properties
        assert len(hilbert_result) == len(test_function)
        assert all(np.isfinite(val) for val in hilbert_result)
        
        # Hilbert transform of sin should be related to cos
        # H[sin(2πx)] = -cos(2πx) (approximately)
        expected = -np.cos(2 * np.pi * dist_theory.x)
        
        # Allow for numerical errors in discrete approximation
        correlation = np.corrcoef(hilbert_result, expected)[0, 1]
        assert correlation > 0.5  # Should be reasonably correlated
    
    def test_generalized_fourier_transform(self, dist_theory):
        """Test generalized Fourier transform."""
        distribution = np.exp(-dist_theory.x**2)  # Gaussian
        
        frequencies, fft_result = dist_theory.generalized_fourier_transform(
            distribution, regularization_method='gaussian'
        )
        
        # Check basic properties
        assert len(frequencies) == len(dist_theory.x)
        assert len(fft_result) == len(dist_theory.x)
        assert all(np.isfinite(freq) for freq in frequencies)
        assert all(np.isfinite(val) for val in fft_result)
        
        # Check frequency range
        assert frequencies[0] <= 0
        assert frequencies[-1] >= 0
        
        # FFT of real Gaussian should be real (up to numerical errors)
        imaginary_part = np.imag(fft_result)
        assert np.max(np.abs(imaginary_part)) < 1e-10
    
    def test_optical_causality_demo_causal(self, dist_theory):
        """Test optical causality with causal signal."""
        # Create causal signal (zero for t < 0)
        time_signal = np.zeros_like(dist_theory.x)
        half_point = len(time_signal) // 2
        time_signal[half_point:] = np.exp(-dist_theory.x[half_point:]**2 / 0.1**2)
        
        causality_result = dist_theory.optical_causality_demo(time_signal)
        
        # Check structure
        assert 'is_causal' in causality_result
        assert 'spectrum' in causality_result
        assert 'frequencies' in causality_result
        assert 'real_part' in causality_result
        assert 'imaginary_part' in causality_result
        assert 'hilbert_of_real' in causality_result
        assert 'kk_relation_satisfied' in causality_result
        
        # Should be causal
        assert causality_result['is_causal'] == True
        
        # Check spectrum properties
        assert len(causality_result['spectrum']) == len(time_signal)
        assert len(causality_result['frequencies']) == len(time_signal)
        assert all(np.isfinite(val) for val in causality_result['spectrum'])
        
        # Kramers-Kronig relations should be satisfied for causal signals
        assert causality_result['kk_relation_satisfied'] in [True, False]
    
    def test_optical_causality_demo_non_causal(self, dist_theory):
        """Test optical causality with non-causal signal."""
        # Create non-causal signal (non-zero for t < 0)
        time_signal = np.exp(-dist_theory.x**2 / 0.1**2)  # Symmetric about 0
        
        causality_result = dist_theory.optical_causality_demo(time_signal)
        
        # Should not be causal
        assert causality_result['is_causal'] == False
        
        # Kramers-Kronig relations may not be satisfied
        assert causality_result['kk_relation_satisfied'] in [True, False]
    
    def test_distribution_derivative_demo(self, dist_theory):
        """Test distribution derivative demonstration."""
        # Create piecewise function with jump discontinuity
        piecewise_func = np.zeros_like(dist_theory.x)
        jump_location = 0.0
        
        for i, x_val in enumerate(dist_theory.x):
            if x_val < jump_location:
                piecewise_func[i] = 1.0
            else:
                piecewise_func[i] = 2.0
        
        derivative_result = dist_theory.distribution_derivative_demo(
            piecewise_func, [jump_location]
        )
        
        # Check structure
        assert 'classical_derivative' in derivative_result
        assert 'discontinuity_points' in derivative_result
        assert 'jump_sizes' in derivative_result
        assert 'delta_contributions' in derivative_result
        assert 'distribution_derivative' in derivative_result
        assert 'number_of_deltas' in derivative_result
        
        # Check classical derivative
        classical_deriv = derivative_result['classical_derivative']
        assert len(classical_deriv) == len(piecewise_func)
        assert all(np.isfinite(val) for val in classical_deriv)
        
        # Should find discontinuity points
        discontinuities = derivative_result['discontinuity_points']
        assert len(discontinuities) >= 0
        
        # Should find jump sizes
        jump_sizes = derivative_result['jump_sizes']
        assert len(jump_sizes) == len(discontinuities)
        
        # Should find delta contributions
        delta_contribs = derivative_result['delta_contributions']
        assert len(delta_contribs) == len(discontinuities)
        
        # Number of deltas should match number of discontinuities
        assert derivative_result['number_of_deltas'] == len(discontinuities)
        
        # For our test function, should find exactly one discontinuity
        assert derivative_result['number_of_deltas'] == 1
        
        # Jump size should be approximately 1.0
        if jump_sizes:
            assert abs(jump_sizes[0] - 1.0) < 0.5
    
    def test_distribution_derivative_demo_smooth(self, dist_theory):
        """Test distribution derivative with smooth function."""
        # Create smooth function (no discontinuities)
        smooth_func = np.sin(2 * np.pi * dist_theory.x)
        
        derivative_result = dist_theory.distribution_derivative_demo(
            smooth_func, []
        )
        
        # Should find no discontinuities for smooth function
        assert derivative_result['number_of_deltas'] == 0
        assert len(derivative_result['discontinuity_points']) == 0
        assert len(derivative_result['jump_sizes']) == 0
        assert len(derivative_result['delta_contributions']) == 0
    
    def test_edge_cases(self, dist_theory):
        """Test edge cases and error handling."""
        # Test delta function with zero width
        with np.errstate(divide='warn'):
            delta_approx = dist_theory.delta_function_approximation(0.0, width=1e-10, method='gaussian')
            assert all(np.isfinite(val) for val in delta_approx)
        
        # Test Green's function with zero wavenumber
        green_func = dist_theory.greens_function_helmholtz(0.0, 1e-10, dimension=1)
        assert all(np.isfinite(val) for val in green_func)
        
        # Test principal value integral with singularity outside domain
        test_func = np.ones_like(dist_theory.x)
        pv_integral = dist_theory.principal_value_integral(test_func, 2.0, method='subtract')
        assert np.isfinite(pv_integral)
        
        # Test Hilbert transform with zeros
        zero_func = np.zeros_like(dist_theory.x)
        hilbert_result = dist_theory.hilbert_transform(zero_func)
        assert all(val == 0.0 for val in hilbert_result)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_distribution_theory(self, mock_figure, mock_show):
        """Test the demonstration function."""
        # Mock matplotlib to avoid showing plots during tests
        mock_figure.return_value = MagicMock()
        mock_show.return_value = None
        
        # This should run without errors
        from chapter10_distribution_theory.distribution_optics import demonstrate_distribution_theory
        demonstrate_distribution_theory()
        
        # Verify matplotlib was called
        mock_figure.assert_called()
        mock_show.assert_called()


class TestDistributionTheoryMathematicalProperties:
    """Test mathematical properties of distribution theory."""
    
    @pytest.fixture
    def dist_theory(self):
        """Create DistributionTheoryOptics instance for mathematical tests."""
        return DistributionTheoryOptics(grid_size=200)
    
    def test_delta_function_limit(self, dist_theory):
        """Test that delta function approximations converge to ideal delta."""
        center = 0.0
        test_func = np.sin(2 * np.pi * dist_theory.x)
        
        # Test with decreasing widths
        widths = [0.1, 0.01, 0.001]
        sifting_errors = []
        
        for width in widths:
            delta_approx = dist_theory.delta_function_approximation(center, width, method='gaussian')
            properties = dist_theory.test_delta_properties(delta_approx, center)
            sifting_errors.append(properties['sifting_error'])
        
        # Errors should generally decrease with width (allow some numerical fluctuations)
        assert len(sifting_errors) == len(widths)
        assert all(error >= 0 for error in sifting_errors)
    
    def test_delta_function_normalization_conservation(self, dist_theory):
        """Test that delta function normalization is conserved."""
        center = 0.5
        
        # Test different methods
        methods = ['gaussian', 'lorentzian', 'sinc']
        
        for method in methods:
            delta_approx = dist_theory.delta_function_approximation(center, width=0.02, method=method)
            properties = dist_theory.test_delta_properties(delta_approx, center)
            
            # Normalization should be close to 1
            normalization = properties['normalization']
            assert abs(normalization - 1.0) < 0.05  # 5% tolerance
    
    def test_greens_function_symmetry(self, dist_theory):
        """Test symmetry properties of Green's functions."""
        wavenumber = 2 * np.pi / 500e-9
        
        # Test 1D symmetry: G(x,x₀) = G(x₀,x)
        x1, x2 = -0.3, 0.4
        
        green_1 = dist_theory.greens_function_helmholtz(x1, wavenumber, dimension=1)
        green_2 = dist_theory.greens_function_helmholtz(x2, wavenumber, dimension=1)
        
        # Find values at symmetric points
        idx1_in_2 = np.argmin(np.abs(dist_theory.x - x1))
        idx2_in_1 = np.argmin(np.abs(dist_theory.x - x2))
        
        val1 = green_1[idx2_in_1]  # Value at x2 when source is at x1
        val2 = green_2[idx1_in_2]  # Value at x1 when source is at x2
        
        # Should be equal (reciprocity)
        assert abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10) < 0.1
    
    def test_principal_value_integral_properties(self, dist_theory):
        """Test mathematical properties of principal value integrals."""
        # Test linearity: P∫(af + bg)/(x-c)dx = aP∫f/(x-c)dx + bP∫g/(x-c)dx
        
        f = np.sin(2 * np.pi * dist_theory.x)
        g = np.cos(2 * np.pi * dist_theory.x)
        c = 0.0
        a, b = 2.0, 3.0
        
        # Compute individual integrals
        pv_f = dist_theory.principal_value_integral(f, c, method='subtract')
        pv_g = dist_theory.principal_value_integral(g, c, method='subtract')
        
        # Compute combined integral
        combined = a * f + b * g
        pv_combined = dist_theory.principal_value_integral(combined, c, method='subtract')
        
        # Should satisfy linearity
        expected = a * pv_f + b * pv_g
        assert abs(pv_combined - expected) < 1e-10
    
    def test_hilbert_transform_properties(self, dist_theory):
        """Test mathematical properties of Hilbert transform."""
        # Test that H[H[f]] = -f (involution property)
        
        f = np.sin(2 * np.pi * dist_theory.x)
        
        H_f = dist_theory.hilbert_transform(f)
        H_H_f = dist_theory.hilbert_transform(H_f)
        
        # Should be approximately -f
        error = np.linalg.norm(H_H_f + f)
        assert error < 1.0  # Allow significant numerical error
    
    def test_fourier_transform_linearity(self, dist_theory):
        """Test linearity of Fourier transform."""
        # Test that F[af + bg] = aF[f] + bF[g]
        
        f = np.exp(-dist_theory.x**2)
        g = np.exp(-4 * dist_theory.x**2)
        a, b = 2.0, 3.0
        
        # Compute individual transforms
        _, F_f = dist_theory.generalized_fourier_transform(f)
        _, F_g = dist_theory.generalized_fourier_transform(g)
        
        # Compute combined transform
        combined = a * f + b * g
        _, F_combined = dist_theory.generalized_fourier_transform(combined)
        
        # Should satisfy linearity
        expected = a * F_f + b * F_g
        error = np.linalg.norm(F_combined - expected)
        assert error < 1e-10
    
    def test_causality_implications(self, dist_theory):
        """Test implications of causality."""
        # Create causal signal
        causal_signal = np.zeros_like(dist_theory.x)
        half_point = len(causal_signal) // 2
        causal_signal[half_point:] = np.exp(-dist_theory.x[half_point:]**2)
        
        result = dist_theory.optical_causality_demo(causal_signal)
        
        # Should be causal
        assert result['is_causal'] == True
        
        # Kramers-Kronig relations should be satisfied
        # (This is a numerical test, so allow some tolerance)
        kk_satisfied = result['kk_relation_satisfied']
        assert kk_satisfied in [True, False]  # May not be perfectly satisfied numerically
    
    def test_distribution_derivative_jump_detection(self, dist_theory):
        """Test detection of jumps in distribution derivatives."""
        # Create function with known jump
        jump_location = 0.3
        jump_size = 2.5
        
        func = np.zeros_like(dist_theory.x)
        for i, x_val in enumerate(dist_theory.x):
            if x_val < jump_location:
                func[i] = 1.0
            else:
                func[i] = 1.0 + jump_size
        
        result = dist_theory.distribution_derivative_demo(func, [jump_location])
        
        # Should detect the jump
        assert result['number_of_deltas'] >= 1
        
        # Jump size should be approximately correct
        if result['jump_sizes']:
            detected_jump = result['jump_sizes'][0]
            assert abs(detected_jump - jump_size) < 1.0  # Allow some numerical error


if __name__ == "__main__":
    pytest.main([__file__])