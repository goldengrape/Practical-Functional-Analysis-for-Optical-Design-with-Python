"""
Test suite for Chapter 8: Banach Spaces - Optical Norms
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from chapter08_banach_spaces.optical_norms import OpticalBanachSpaces


class TestOpticalBanachSpaces:
    """Test suite for Optical Banach Spaces functionality."""
    
    @pytest.fixture
    def banach(self):
        """Create a default OpticalBanachSpaces instance."""
        return OpticalBanachSpaces(grid_size=50)
    
    @pytest.fixture
    def test_functions(self, banach):
        """Create test functions for analysis."""
        return {
            'defocus': 0.5 * (2 * banach.x**2 - 1),
            'astigmatism': 0.3 * banach.x**2,
            'coma': 0.2 * (3 * banach.x**3 - 2 * banach.x),
            'spherical': 0.1 * (6 * banach.x**4 - 6 * banach.x**2 + 1),
            'constant': np.ones_like(banach.x),
            'linear': banach.x,
            'gaussian': np.exp(-banach.x**2)
        }
    
    def test_initialization(self):
        """Test proper initialization of OpticalBanachSpaces."""
        banach = OpticalBanachSpaces(grid_size=100)
        assert banach.grid_size == 100
        assert len(banach.x) == 100
        assert banach.dx == pytest.approx(2/99, rel=1e-10)
        assert np.isclose(banach.x[0], -1.0)
        assert np.isclose(banach.x[-1], 1.0)
    
    def test_lp_norm_l2(self, banach):
        """Test L² norm computation."""
        # Test with constant function
        const_func = np.ones_like(banach.x)
        l2_norm = banach.lp_norm(const_func, p=2)
        expected = np.sqrt(2.0)  # sqrt(∫1²dx) from -1 to 1 = sqrt(2)
        assert l2_norm == pytest.approx(expected, rel=1e-2)
        
        # Test with zero function
        zero_func = np.zeros_like(banach.x)
        l2_norm_zero = banach.lp_norm(zero_func, p=2)
        assert l2_norm_zero == pytest.approx(0.0, abs=1e-10)
    
    def test_lp_norm_l1(self, banach):
        """Test L¹ norm computation."""
        # Test with constant function
        const_func = np.ones_like(banach.x)
        l1_norm = banach.lp_norm(const_func, p=1)
        expected = 2.0  # ∫1dx from -1 to 1 = 2
        assert l1_norm == pytest.approx(expected, rel=1e-2)
    
    def test_lp_norm_linf(self, banach):
        """Test L∞ norm computation."""
        # Test with constant function
        const_func = np.ones_like(banach.x)
        linf_norm = banach.lp_norm(const_func, p=np.inf)
        assert linf_norm == pytest.approx(1.0, abs=1e-10)
        
        # Test with linear function
        linear_func = banach.x
        linf_norm_linear = banach.lp_norm(linear_func, p=np.inf)
        assert linf_norm_linear == pytest.approx(1.0, abs=1e-10)
    
    def test_sobolev_norm_basic(self, banach):
        """Test basic Sobolev norm computation."""
        # Test with constant function (derivative should be zero)
        const_func = np.ones_like(banach.x)
        h1_norm = banach.sobolev_norm(const_func, s=1, p=2)
        
        # For constant function: ‖f‖₂ + ‖f'‖₂ = √2 + 0 = √2
        expected = np.sqrt(2.0)
        assert h1_norm == pytest.approx(expected, rel=1e-2)
    
    def test_sobolev_norm_higher_order(self, banach):
        """Test higher-order Sobolev norm."""
        # Test with linear function
        linear_func = banach.x
        h2_norm = banach.sobolev_norm(linear_func, s=2, p=2)
        
        # Should be finite and positive
        assert h2_norm > 0
        assert np.isfinite(h2_norm)
    
    def test_optical_quality_metrics(self, banach, test_functions):
        """Test optical quality metrics computation."""
        for func_name, func in test_functions.items():
            metrics = banach.optical_quality_metrics(func)
            
            # Check all required metrics are present
            required_metrics = ['rms_error', 'peak_valley', 'gradient_energy', 
                              'total_variation', 'smoothness']
            for metric in required_metrics:
                assert metric in metrics
                assert np.isfinite(metrics[metric])
                assert metrics[metric] >= 0
            
            # RMS should be L² norm
            expected_rms = banach.lp_norm(func, p=2)
            assert metrics['rms_error'] == pytest.approx(expected_rms, rel=1e-10)
            
            # Peak-to-valley should be L∞ norm
            expected_pv = banach.lp_norm(func, p=np.inf)
            assert metrics['peak_valley'] == pytest.approx(expected_pv, rel=1e-10)
    
    def test_norm_equivalence_analysis(self, banach, test_functions):
        """Test norm equivalence analysis."""
        equivalence = banach.norm_equivalence_analysis(test_functions)
        
        assert len(equivalence) == len(test_functions)
        
        for func_name, norms in equivalence.items():
            # Check all required norms are present
            required_norms = ['L1', 'L2', 'L_inf', 'H1', 'L2/L1_ratio', 
                            'L_inf/L2_ratio', 'H1/L2_ratio']
            for norm in required_norms:
                assert norm in norms
                assert np.isfinite(norms[norm])
                assert norms[norm] >= 0
            
            # Test norm relationships
            # L² ≤ L∞ for finite domains (roughly)
            assert norms['L2'] <= norms['L_inf'] * 1.1  # Allow small numerical error
            
            # Ratios should be positive
            assert norms['L2/L1_ratio'] > 0
            assert norms['L_inf/L2_ratio'] > 0
            assert norms['H1/L2_ratio'] > 0
    
    def test_completeness_test_cauchy(self, banach):
        """Test completeness test with Cauchy sequence."""
        # Create a convergent sequence
        convergent_sequence = []
        target = np.exp(-banach.x**2)
        for n in range(1, 6):
            func = target + np.sin(n * banach.x) / (n**2)
            convergent_sequence.append(func)
        
        completeness = banach.completeness_test(convergent_sequence)
        
        assert 'is_cauchy' in completeness
        assert 'differences' in completeness
        assert 'final_difference' in completeness
        
        # This should be approximately Cauchy
        assert completeness['is_cauchy'] == True
        assert len(completeness['differences']) == len(convergent_sequence) - 1
        assert completeness['final_difference'] >= 0
    
    def test_completeness_test_non_cauchy(self, banach):
        """Test completeness test with non-Cauchy sequence."""
        # Create a non-convergent sequence
        non_convergent = []
        for n in range(1, 6):
            func = n * banach.x  # Grows without bound
            non_convergent.append(func)
        
        completeness = banach.completeness_test(non_convergent)
        
        # This should not be Cauchy
        assert completeness['is_cauchy'] == False
    
    def test_dual_space_analysis(self, banach, test_functions):
        """Test dual space analysis."""
        original_function = test_functions['gaussian']
        
        # Create test linear functionals
        test_functionals = {
            'identity': np.ones_like(banach.x),  # ∫f(x)dx
            'linear_weight': banach.x,  # ∫xf(x)dx
            'quadratic_weight': banach.x**2,  # ∫x²f(x)dx
            'gaussian_weight': np.exp(-banach.x**2)  # ∫e^(-x²)f(x)dx
        }
        
        dual_analysis = banach.dual_space_analysis(original_function, test_functionals)
        
        assert len(dual_analysis) == len(test_functionals)
        
        for func_name, result in dual_analysis.items():
            assert 'dual_value' in result
            assert 'functional_norm' in result
            assert 'normalized_value' in result
            
            assert np.isfinite(result['dual_value'])
            assert np.isfinite(result['functional_norm'])
            assert result['functional_norm'] >= 0
            
            # Normalized value should be finite
            if result['functional_norm'] > 0:
                assert np.isfinite(result['normalized_value'])
    
    def test_optimization_in_banach_space_l2(self, banach):
        """Test optimization in L² Banach space."""
        target = np.exp(-banach.x**2)
        initial = np.zeros_like(target)
        
        result, history = banach.optimization_in_banach_space(target, initial, norm_order=2)
        
        assert len(result) == len(target)
        assert len(history) > 0
        assert len(history) <= 100  # Max iterations
        
        # Final objective should be small
        assert history[-1] < 1.0  # Should converge to reasonable error
        
        # Result should be closer to target than initial
        initial_error = banach.lp_norm(initial - target, p=2)
        final_error = banach.lp_norm(result - target, p=2)
        assert final_error < initial_error
    
    def test_optimization_in_banach_space_l1(self, banach):
        """Test optimization in L¹ Banach space."""
        target = np.exp(-banach.x**2)
        initial = np.zeros_like(target)
        
        result, history = banach.optimization_in_banach_space(target, initial, norm_order=1)
        
        assert len(result) == len(target)
        assert len(history) > 0
        assert history[-1] < 1.0  # Should converge to reasonable error
    
    def test_optimization_convergence(self, banach):
        """Test that optimization converges properly."""
        target = np.sin(banach.x)
        initial = np.zeros_like(target)
        
        result, history = banach.optimization_in_banach_space(target, initial, norm_order=2)
        
        # History should generally decrease (allow some fluctuations)
        decreasing_count = 0
        for i in range(1, len(history)):
            if history[i] <= history[i-1]:
                decreasing_count += 1
        
        # At least 80% of steps should decrease or stay the same
        assert decreasing_count >= 0.8 * (len(history) - 1)
    
    def test_edge_cases(self, banach):
        """Test edge cases and error handling."""
        # Test with empty arrays (should not crash)
        empty_func = np.array([])
        with pytest.raises((IndexError, ValueError)):
            banach.lp_norm(empty_func, p=2)
        
        # Test with NaN values
        nan_func = np.full_like(banach.x, np.nan)
        result = banach.lp_norm(nan_func, p=2)
        assert np.isnan(result)
        
        # Test with infinite values
        inf_func = np.full_like(banach.x, np.inf)
        result = banach.lp_norm(inf_func, p=2)
        assert np.isinf(result)
    
    def test_norm_properties(self, banach, test_functions):
        """Test mathematical properties of norms."""
        func = test_functions['gaussian']
        
        # Triangle inequality: ‖f+g‖ ≤ ‖f‖ + ‖g‖
        func2 = test_functions['linear']
        sum_func = func + func2
        
        norm_sum = banach.lp_norm(sum_func, p=2)
        norm_f = banach.lp_norm(func, p=2)
        norm_g = banach.lp_norm(func2, p=2)
        
        assert norm_sum <= norm_f + norm_g + 1e-10  # Allow numerical error
        
        # Homogeneity: ‖αf‖ = |α|‖f‖
        alpha = 2.5
        scaled_func = alpha * func
        norm_scaled = banach.lp_norm(scaled_func, p=2)
        expected_scaled = abs(alpha) * banach.lp_norm(func, p=2)
        
        assert norm_scaled == pytest.approx(expected_scaled, rel=1e-10)
    
    def test_sobolev_properties(self, banach):
        """Test properties of Sobolev norms."""
        # Higher smoothness should give larger norm for oscillatory functions
        oscillatory = np.sin(5 * banach.x)
        
        h1_norm = banach.sobolev_norm(oscillatory, s=1, p=2)
        h2_norm = banach.sobolev_norm(oscillatory, s=2, p=2)
        
        # H² should be larger than H¹ for oscillatory functions
        assert h2_norm > h1_norm
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_banach_spaces(self, mock_figure, mock_show):
        """Test the demonstration function."""
        # Mock matplotlib to avoid showing plots during tests
        mock_figure.return_value = MagicMock()
        mock_show.return_value = None
        
        # This should run without errors
        from chapter08_banach_spaces.optical_norms import demonstrate_banach_spaces
        demonstrate_banach_spaces()
        
        # Verify matplotlib was called
        mock_figure.assert_called()
        mock_show.assert_called()


class TestBanachSpaceMathematicalProperties:
    """Test mathematical properties of Banach spaces."""
    
    @pytest.fixture
    def banach(self):
        """Create OpticalBanachSpaces instance for mathematical tests."""
        return OpticalBanachSpaces(grid_size=100)
    
    def test_finite_dimensional_equivalence(self, banach):
        """Test that all norms are equivalent in finite dimensions."""
        # Create a test function
        func = np.sin(banach.x) + 0.5 * np.cos(2 * banach.x)
        
        # Compute different norms
        l1 = banach.lp_norm(func, p=1)
        l2 = banach.lp_norm(func, p=2)
        l3 = banach.lp_norm(func, p=3)
        linf = banach.lp_norm(func, p=np.inf)
        
        # All norms should be positive and finite
        assert l1 > 0 and np.isfinite(l1)
        assert l2 > 0 and np.isfinite(l2)
        assert l3 > 0 and np.isfinite(l3)
        assert linf > 0 and np.isfinite(linf)
        
        # Norms should be ordered: L¹ ≥ L² ≥ L³ ≥ ... ≥ L∞ (roughly)
        # This is true for functions on finite domains with appropriate normalization
        assert l1 >= l2 * 0.9  # Allow some numerical tolerance
        assert l2 >= linf * 0.9
    
    def test_completeness_property(self, banach):
        """Test the completeness property of Banach spaces."""
        # Create a sequence that should converge
        target = np.exp(-banach.x**2)
        sequence = []
        
        for n in range(1, 8):
            # Approximation that gets better with n
            approx = target + np.exp(-n) * np.sin(n * banach.x)
            sequence.append(approx)
        
        completeness = banach.completeness_test(sequence)
        
        # This should be a Cauchy sequence
        assert completeness['is_cauchy'] == True
        
        # Differences should decrease
        differences = completeness['differences']
        assert len(differences) == len(sequence) - 1
        
        # Final difference should be small
        assert completeness['final_difference'] < 0.1
    
    def test_dual_space_relationships(self, banach):
        """Test relationships in dual spaces."""
        # For L² space, the dual is L² (Hilbert space is self-dual)
        func = np.exp(-banach.x**2)
        
        # Create a functional in L²
        functional = np.sin(banach.x)
        
        # Compute dual pairing
        dual_value = np.sum(func * functional) * banach.dx
        
        # Compute norms
        func_norm = banach.lp_norm(func, p=2)
        functional_norm = banach.lp_norm(functional, p=2)
        
        # Cauchy-Schwarz inequality: |⟨f,g⟩| ≤ ‖f‖‖g‖
        assert abs(dual_value) <= func_norm * functional_norm + 1e-10
    
    def test_optimization_convexity(self, banach):
        """Test that optimization problems are convex for L^p norms with p ≥ 1."""
        target = np.sin(banach.x)
        initial = np.zeros_like(target)
        
        # Test different norm orders
        for p in [1, 1.5, 2, 3]:
            result, history = banach.optimization_in_banach_space(target, initial, norm_order=p)
            
            # Should converge for convex problems
            assert len(history) > 0
            assert history[-1] < history[0]  # Should improve
            
            # Final result should be reasonable
            final_error = banach.lp_norm(result - target, p)
            initial_error = banach.lp_norm(initial - target, p)
            assert final_error < initial_error


if __name__ == "__main__":
    pytest.main([__file__])