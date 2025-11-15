"""
Test module for Chapter 12: Uncertainty Quantification in Optical Systems
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter12_uncertainty_quantification.uncertainty_analysis_optics import UncertaintyQuantificationOptics


class TestUncertaintyQuantificationOptics:
    """Test class for uncertainty quantification methods in optical systems."""
    
    @pytest.fixture
    def uq_optics(self):
        """Fixture for UncertaintyQuantificationOptics instance."""
        return UncertaintyQuantificationOptics()
    
    def test_initialization(self, uq_optics):
        """Test proper initialization of UncertaintyQuantificationOptics."""
        assert hasattr(uq_optics, 'rng')
        assert isinstance(uq_optics.rng, np.random.RandomState)
        assert uq_optics.rng.get_state()[1][0] == 42  # Check seed
    
    def test_monte_carlo_wavefront_analysis_basic(self, uq_optics):
        """Test basic Monte Carlo wavefront analysis functionality."""
        results = uq_optics.monte_carlo_wavefront_analysis(num_samples=100)
        
        # Check that all expected keys are present
        expected_keys = [
            'rms_mean', 'rms_std', 'rms_95_ci', 'pv_mean', 'pv_std', 
            'pv_95_ci', 'correlation_matrix'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check statistical properties
        assert isinstance(results['rms_mean'], (int, float))
        assert isinstance(results['rms_std'], (int, float))
        assert len(results['rms_95_ci']) == 2
        assert results['rms_95_ci'][0] <= results['rms_95_ci'][1]
        
        # Check that RMS and PV values are reasonable
        assert results['rms_mean'] >= 0  # RMS should be non-negative
        assert results['pv_mean'] >= 0  # Peak-to-valley should be non-negative
        
        # Check correlation matrix
        assert results['correlation_matrix'].shape == (3, 3)
        assert np.allclose(results['correlation_matrix'], results['correlation_matrix'].T)
        
        # Diagonal should be 1 (self-correlation)
        np.testing.assert_allclose(np.diag(results['correlation_matrix']), 1.0)
    
    def test_polynomial_chaos_zernike_basic(self, uq_optics):
        """Test basic polynomial chaos expansion for Zernike coefficients."""
        results = uq_optics.polynomial_chaos_zernike(order=2)
        
        # Check that all expected keys are present
        expected_keys = [
            'expansion_coefficients', 'coefficient_samples', 
            'wavefront_variance', 'convergence_analysis'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check expansion coefficients
        assert len(results['expansion_coefficients']) == 3  # 3 coefficients
        for coeffs in results['expansion_coefficients']:
            assert len(coeffs) == 3  # order + 1 terms
        
        # Check coefficient samples
        coeff_samples = results['coefficient_samples']
        assert 'defocus' in coeff_samples
        assert 'astigmatism' in coeff_samples
        assert 'coma' in coeff_samples
    
    def test_bayesian_parameter_estimation_basic(self, uq_optics):
        """Test basic Bayesian parameter estimation."""
        results = uq_optics.bayesian_parameter_estimation()
        
        # Check that all expected keys are present
        expected_keys = [
            'posterior_samples', 'parameter_estimates', 
            'correlation_matrix', 'convergence_diagnostics'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check parameter estimates
        param_estimates = results['parameter_estimates']
        assert 'focal_length' in param_estimates
        assert 'aberration' in param_estimates
        
        for param_name, stats in param_estimates.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'ci_95' in stats
    
    def test_uncertainty_propagation_lens_design(self, uq_optics):
        """Test uncertainty propagation in lens design."""
        results = uq_optics.uncertainty_propagation_lens_design()
        
        # Check that all expected keys are present
        expected_keys = [
            'focal_length', 'spherical_aberration', 'chromatic_aberration',
            'correlation_matrix', 'yield_analysis'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check focal length statistics
        focal_length_stats = results['focal_length']
        assert 'mean' in focal_length_stats
        assert 'std' in focal_length_stats
        assert 'percentiles' in focal_length_stats
        
        # Check yield analysis
        yield_analysis = results['yield_analysis']
        assert 'yield_rate' in yield_analysis
        assert 0 <= yield_analysis['yield_rate'] <= 1  # Should be a probability
    
    def test_private_methods_pce_convergence(self, uq_optics):
        """Test private method for PCE convergence analysis."""
        # Create test expansion coefficients
        expansion_coeffs = [
            [1.0, 0.5, 0.1, 0.01],  # defocus
            [0.5, 0.3, 0.05, 0.005],  # astigmatism  
            [0.3, 0.2, 0.02, 0.002]   # coma
        ]
        
        convergence_stats = uq_optics._analyze_pce_convergence(expansion_coeffs)
        
        assert len(convergence_stats) == 3  # 3 coefficients
        
        for param_name, stats in convergence_stats.items():
            assert 'total_variance' in stats
            assert 'first_order_sobol_indices' in stats
            assert 'convergence_rate' in stats
    
    def test_private_methods_mcmc_diagnostics(self, uq_optics):
        """Test private method for MCMC convergence diagnostics."""
        # Generate synthetic MCMC samples
        np.random.seed(42)
        n_samples = 1000
        samples = np.column_stack([
            100.0 + np.random.normal(0, 1.0, n_samples),  # focal length
            0.1 + np.random.normal(0, 0.01, n_samples)   # aberration
        ])
        
        diagnostics = uq_optics._mcmc_convergence_diagnostics(samples)
        
        assert len(diagnostics) == 2  # 2 parameters
        
        for param_name, param_diagnostics in diagnostics.items():
            assert 'gelman_rubin' in param_diagnostics
            assert 'effective_sample_size' in param_diagnostics
            assert 'autocorrelation_lag' in param_diagnostics
    
    def test_private_methods_yield_analysis(self, uq_optics):
        """Test private method for yield analysis."""
        # Generate synthetic data
        np.random.seed(42)
        focal_lengths = 100.0 + np.random.normal(0, 2.0, 1000)
        spherical_aberrations = np.random.normal(0, 0.05, 1000)
        
        yield_stats = uq_optics._calculate_yield_analysis(focal_lengths, spherical_aberrations)
        
        expected_keys = [
            'yield_rate', 'yield_confidence_interval',
            'focal_length_acceptance_rate', 'spherical_aberration_acceptance_rate'
        ]
        for key in expected_keys:
            assert key in yield_stats
        
        # Check yield rate
        yield_rate = yield_stats['yield_rate']
        assert 0 <= yield_rate <= 1  # Should be a valid probability
    
    def test_reproducibility(self, uq_optics):
        """Test that results are reproducible with the same random seed."""
        # First run
        results1 = uq_optics.monte_carlo_wavefront_analysis(num_samples=100)
        
        # Second run with same instance (same seed)
        results2 = uq_optics.monte_carlo_wavefront_analysis(num_samples=100)
        
        # Results should be identical due to fixed seed
        np.testing.assert_allclose(results1['rms_mean'], results2['rms_mean'])
        np.testing.assert_allclose(results1['rms_std'], results2['rms_std'])
        np.testing.assert_allclose(results1['correlation_matrix'], results2['correlation_matrix'])
    
    def test_edge_cases_small_samples(self, uq_optics):
        """Test edge cases with small sample sizes."""
        # Test with very small sample size
        results = uq_optics.monte_carlo_wavefront_analysis(num_samples=10)
        
        # Should still produce valid results
        assert 'rms_mean' in results
        assert 'rms_std' in results
        assert results['rms_mean'] >= 0
    
    def test_mathematical_consistency(self, uq_optics):
        """Test mathematical consistency of methods."""
        # Test that correlation matrices have correct properties
        mc_results = uq_optics.monte_carlo_wavefront_analysis(num_samples=100)
        correlation_matrix = mc_results['correlation_matrix']
        
        # Should be symmetric
        assert np.allclose(correlation_matrix, correlation_matrix.T)
        
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(correlation_matrix), 1.0)
        
        # All values should be in [-1, 1]
        assert np.all(correlation_matrix >= -1)
        assert np.all(correlation_matrix <= 1)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_all_methods(self, mock_figure, mock_show, uq_optics):
        """Test the demonstration method."""
        # This should run without errors
        uq_optics.demonstrate_all_methods()
        
        # Check that matplotlib functions were called
        assert mock_figure.called or mock_show.called