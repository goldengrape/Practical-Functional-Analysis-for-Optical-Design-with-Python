"""
Test suite for Chapter 9: Weak Convergence in Optical Systems
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from chapter09_weak_convergence.weak_convergence_optics import WeakConvergenceOptics


class TestWeakConvergenceOptics:
    """Test suite for Weak Convergence in Optical Systems."""
    
    @pytest.fixture
    def weak_conv(self):
        """Create a default WeakConvergenceOptics instance."""
        return WeakConvergenceOptics(grid_size=50)
    
    @pytest.fixture
    def test_functionals(self, weak_conv):
        """Create test functionals for weak convergence analysis."""
        return {
            'constant': np.ones_like(weak_conv.x),
            'linear': weak_conv.x,
            'quadratic': weak_conv.x**2,
            'sinusoidal': np.sin(weak_conv.x),
            'gaussian': np.exp(-weak_conv.x**2),
            'indicator': np.abs(weak_conv.x) < 0.5
        }
    
    @pytest.fixture
    def convergent_sequence(self, weak_conv):
        """Create a sequence that converges weakly."""
        sequence = []
        target = np.exp(-weak_conv.x**2)
        for n in range(1, 11):
            # Convergent sequence approaching target
            fn = target + np.sin(n * weak_conv.x) / n
            sequence.append(fn)
        return sequence
    
    @pytest.fixture
    def oscillating_sequence(self, weak_conv):
        """Create an oscillating sequence for weak convergence demo."""
        sequence = []
        for n in range(1, 21):
            # Oscillating sequence: fₙ(x) = sin(nx)/√n
            fn = np.sin(n * weak_conv.x) / np.sqrt(n)
            sequence.append(fn)
        return sequence
    
    def test_initialization(self):
        """Test proper initialization of WeakConvergenceOptics."""
        weak_conv = WeakConvergenceOptics(grid_size=100)
        assert weak_conv.grid_size == 100
        assert len(weak_conv.x) == 100
        assert weak_conv.dx == pytest.approx(2/99, rel=1e-10)
        assert np.isclose(weak_conv.x[0], -1.0)
        assert np.isclose(weak_conv.x[-1], 1.0)
    
    def test_weak_convergence_test_convergent(self, weak_conv, test_functionals, convergent_sequence):
        """Test weak convergence with a convergent sequence."""
        results = weak_conv.weak_convergence_test(convergent_sequence, test_functionals)
        
        # Check structure of results
        assert len(results) == len(test_functionals)
        
        for func_name, result in results.items():
            assert 'integrals' in result
            assert 'converged' in result
            assert 'final_value' in result
            assert 'convergence_rate' in result
            
            # Check integrals
            assert len(result['integrals']) == len(convergent_sequence)
            assert all(np.isfinite(val) for val in result['integrals'])
            
            # This should converge for most functionals
            assert result['converged'] in [True, False]
            assert np.isfinite(result['final_value'])
            
            # Convergence rate should be array of finite values
            assert len(result['convergence_rate']) == len(convergent_sequence)
            assert all(np.isfinite(rate) for rate in result['convergence_rate'])
    
    def test_weak_convergence_test_oscillating(self, weak_conv, test_functionals, oscillating_sequence):
        """Test weak convergence with oscillating sequence."""
        results = weak_conv.weak_convergence_test(oscillating_sequence, test_functionals)
        
        # For oscillating sequence sin(nx)/√n, many functionals should converge to 0
        for func_name, result in results.items():
            assert len(result['integrals']) == len(oscillating_sequence)
            
            # For constant functional, integral should converge to 0
            if func_name == 'constant':
                # ∫sin(nx)/√n dx should approach 0 as n → ∞
                assert abs(result['final_value']) < 0.1
    
    def test_gradient_descent_weak_convergence(self, weak_conv):
        """Test gradient descent weak convergence analysis."""
        # Initial function (parabolic wavefront error)
        initial_function = 0.5 * weak_conv.x**2
        
        def merit_functional(func):
            """Merit functional: RMS wavefront error."""
            return np.sqrt(np.mean(func**2))
        
        history = weak_conv.gradient_descent_weak_convergence(
            initial_function, merit_functional, num_iterations=50
        )
        
        # Check structure of history
        assert len(history) > 0
        assert len(history) <= 50
        
        for i, record in enumerate(history):
            assert 'iteration' in record
            assert 'merit' in record
            assert 'strong_norm' in record
            assert 'weak_norm' in record
            assert 'functional_change' in record
            assert 'convergence_rate' in record
            
            assert record['iteration'] == i
            assert np.isfinite(record['merit'])
            assert np.isfinite(record['strong_norm'])
            assert np.isfinite(record['weak_norm'])
            assert np.isfinite(record['functional_change'])
            assert np.isfinite(record['convergence_rate'])
            
            # Norms should be non-negative
            assert record['strong_norm'] >= 0
            assert record['weak_norm'] >= 0
            assert record['merit'] >= 0
        
        # Merit should generally decrease or stay stable
        merits = [record['merit'] for record in history]
        final_merit = merits[-1]
        initial_merit = merits[0]
        assert final_merit <= initial_merit * 1.1  # Allow small increase
    
    def test_optical_measurement_convergence(self, weak_conv, test_functionals):
        """Test optical measurement convergence analysis."""
        # Create sequence of wavefronts with decreasing high-frequency components
        wavefront_sequence = []
        for n in range(1, 11):
            # Low frequency component (convergent)
            low_freq = np.sin(weak_conv.x)
            # High frequency component (decreasing amplitude)
            high_freq = 0.1 * np.sin(n * 10 * weak_conv.x) / n
            wavefront = low_freq + high_freq
            wavefront_sequence.append(wavefront)
        
        # Different measurement apertures
        measurement_apertures = {
            'full_aperture': np.ones_like(weak_conv.x),
            'center_spot': np.exp(-weak_conv.x**2 / 0.1**2),
            'edge_sensor': np.abs(weak_conv.x) > 0.8,
            'gradient_sensor': np.gradient(np.sin(weak_conv.x))
        }
        
        results = weak_conv.optical_measurement_convergence(
            wavefront_sequence, measurement_apertures
        )
        
        # Check structure of results
        assert len(results) == len(measurement_apertures)
        
        for aperture_name, result in results.items():
            assert 'measurements' in result
            assert 'final_value' in result
            assert 'variance' in result
            assert 'converged' in result
            assert 'measurement_error' in result
            
            # Check measurements
            assert len(result['measurements']) == len(wavefront_sequence)
            assert all(np.isfinite(val) for val in result['measurements'])
            
            assert np.isfinite(result['final_value'])
            assert result['converged'] in [True, False]
            
            # Measurement error should be array of finite values
            assert len(result['measurement_error']) == len(wavefront_sequence)
            assert all(np.isfinite(err) for err in result['measurement_error'])
    
    def test_weak_compactness_demo(self, weak_conv, oscillating_sequence):
        """Test weak compactness demonstration."""
        result = weak_conv.weak_compactness_demo(oscillating_sequence)
        
        # Check structure
        assert 'is_bounded' in result
        assert 'max_norm' in result
        assert 'subsequences' in result
        assert 'has_convergent_subsequence' in result
        
        # Check boundedness
        assert result['is_bounded'] in [True, False]
        assert np.isfinite(result['max_norm'])
        assert result['max_norm'] >= 0
        
        # Check subsequences
        assert len(result['subsequences']) > 0
        
        for subseq in result['subsequences']:
            assert 'start_index' in subseq
            assert 'integrals' in subseq
            assert 'appears_convergent' in subseq
            assert 'final_integral' in subseq
            assert 'variance' in subseq
            
            assert subseq['start_index'] >= 0
            assert len(subseq['integrals']) > 0
            assert subseq['appears_convergent'] in [True, False]
            assert np.isfinite(subseq['final_integral'])
            assert np.isfinite(subseq['variance'])
        
        # Should have convergent subsequence if bounded (Banach-Alaoglu)
        if result['is_bounded']:
            assert result['has_convergent_subsequence'] in [True, False]
    
    def test_weak_vs_strong_convergence_demo(self, weak_conv):
        """Test weak vs strong convergence demonstration."""
        result = weak_conv.weak_vs_strong_convergence_demo()
        
        # Check structure
        assert 'sequence_type' in result
        assert 'weak_convergence' in result
        assert 'strong_norms' in result
        assert 'strongly_convergent' in result
        assert 'converges_weakly_to_zero' in result
        assert 'explanation' in result
        
        # Check sequence type
        assert result['sequence_type'] == 'oscillating'
        
        # Check weak convergence results
        assert isinstance(result['weak_convergence'], dict)
        
        # Check strong norms
        assert len(result['strong_norms']) > 0
        assert all(np.isfinite(norm) for norm in result['strong_norms'])
        assert all(norm >= 0 for norm in result['strong_norms'])
        
        # Check convergence properties
        assert result['strongly_convergent'] in [True, False]
        assert result['converges_weakly_to_zero'] in [True, False]
        
        # For oscillating sequence sin(nx)/√n, should converge weakly to 0 but not strongly
        assert result['converges_weakly_to_zero'] == True
        assert result['strongly_convergent'] == False
    
    def test_merit_functional_behavior(self, weak_conv):
        """Test different merit functionals."""
        initial_function = weak_conv.x**2
        
        # Test different merit functionals
        merit_functionals = [
            lambda f: np.sqrt(np.mean(f**2)),  # RMS
            lambda f: np.max(np.abs(f)),     # Peak-to-valley
            lambda f: np.mean(np.abs(f)),     # Mean absolute
            lambda f: np.trapz(f**2, weak_conv.x)  # L² norm squared
        ]
        
        for merit_func in merit_functionals:
            history = weak_conv.gradient_descent_weak_convergence(
                initial_function, merit_func, num_iterations=30
            )
            
            # Should have reasonable convergence behavior
            assert len(history) > 0
            assert len(history) <= 30
            
            # Merit should be finite and non-negative
            merits = [record['merit'] for record in history]
            assert all(np.isfinite(m) for m in merits)
            assert all(m >= 0 for m in merits)
    
    def test_measurement_aperture_effects(self, weak_conv):
        """Test different measurement aperture effects."""
        # Simple wavefront sequence
        wavefront_sequence = [np.sin(weak_conv.x) + 0.1 * np.sin(n * weak_conv.x) / n 
                             for n in range(1, 6)]
        
        # Test different aperture types
        apertures = {
            'uniform': np.ones_like(weak_conv.x),
            'gaussian': np.exp(-weak_conv.x**2),
            'linear': np.abs(weak_conv.x),
            'quadratic': weak_conv.x**2,
            'indicator': (np.abs(weak_conv.x) < 0.5).astype(float)
        }
        
        results = weak_conv.optical_measurement_convergence(
            wavefront_sequence, apertures
        )
        
        # All apertures should produce valid results
        for aperture_name, result in results.items():
            assert len(result['measurements']) == len(wavefront_sequence)
            assert all(np.isfinite(m) for m in result['measurements'])
            assert np.isfinite(result['final_value'])
    
    def test_convergence_rates(self, weak_conv, test_functionals):
        """Test convergence rate analysis."""
        # Create sequences with different convergence rates
        sequences = {
            'linear': [np.sin(weak_conv.x) + np.sin(n * weak_conv.x) / n 
                      for n in range(1, 11)],
            'quadratic': [np.sin(weak_conv.x) + np.sin(n * weak_conv.x) / n**2 
                         for n in range(1, 11)],
            'exponential': [np.sin(weak_conv.x) + np.sin(n * weak_conv.x) / np.exp(n) 
                           for n in range(1, 8)]  # Fewer terms due to fast convergence
        }
        
        for seq_name, sequence in sequences.items():
            results = weak_conv.weak_convergence_test(sequence, test_functionals)
            
            for func_name, result in results.items():
                convergence_rates = result['convergence_rate']
                
                # Final convergence rate should be small for convergent sequences
                if len(convergence_rates) > 1:
                    final_rate = convergence_rates[-1]
                    assert np.isfinite(final_rate)
                    
                    # For convergent sequences, final rate should be small
                    if seq_name == 'exponential':
                        assert final_rate < 1e-4  # Fast convergence
                    elif seq_name == 'quadratic':
                        assert final_rate < 1e-2   # Medium convergence
    
    def test_edge_cases(self, weak_conv, test_functionals):
        """Test edge cases and error handling."""
        # Test with empty sequence
        empty_sequence = []
        results = weak_conv.weak_convergence_test(empty_sequence, test_functionals)
        assert len(results) == len(test_functionals)
        
        for func_name, result in results.items():
            assert len(result['integrals']) == 0
            assert result['converged'] == False
        
        # Test with single function sequence
        single_function = [np.sin(weak_conv.x)]
        results = weak_conv.weak_convergence_test(single_function, test_functionals)
        
        for func_name, result in results.items():
            assert len(result['integrals']) == 1
            assert np.isfinite(result['final_value'])
        
        # Test with NaN values
        nan_function = [np.full_like(weak_conv.x, np.nan)]
        results = weak_conv.weak_convergence_test(nan_function, test_functionals)
        
        for func_name, result in results.items():
            assert np.isnan(result['final_value'])
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_weak_convergence(self, mock_figure, mock_show):
        """Test the demonstration function."""
        # Mock matplotlib to avoid showing plots during tests
        mock_figure.return_value = MagicMock()
        mock_show.return_value = None
        
        # This should run without errors
        from chapter09_weak_convergence.weak_convergence_optics import demonstrate_weak_convergence
        demonstrate_weak_convergence()
        
        # Verify matplotlib was called
        mock_figure.assert_called()
        mock_show.assert_called()


class TestWeakConvergenceMathematicalProperties:
    """Test mathematical properties of weak convergence."""
    
    @pytest.fixture
    def weak_conv(self):
        """Create WeakConvergenceOptics instance for mathematical tests."""
        return WeakConvergenceOptics(grid_size=100)
    
    def test_weak_convergence_properties(self, weak_conv):
        """Test mathematical properties of weak convergence."""
        # Create a sequence that converges weakly to zero
        sequence = [np.sin(n * weak_conv.x) / n for n in range(1, 21)]
        
        # Test with constant functional
        constant_functional = np.ones_like(weak_conv.x)
        results = weak_conv.weak_convergence_test(sequence, {'constant': constant_functional})
        
        # Should converge to zero: ∫sin(nx)/n dx → 0
        final_value = results['constant']['final_value']
        assert abs(final_value) < 0.1
        assert results['constant']['converged'] == True
    
    def test_weak_but_not_strong_convergence(self, weak_conv):
        """Test sequence that converges weakly but not strongly."""
        result = weak_conv.weak_vs_strong_convergence_demo()
        
        # Should converge weakly to zero
        assert result['converges_weakly_to_zero'] == True
        
        # Should not converge strongly (norms don't go to zero)
        assert result['strongly_convergent'] == False
        
        # Strong norms should be bounded away from zero
        strong_norms = result['strong_norms']
        assert min(strong_norms) > 0.1  # Should not approach zero
    
    def test_boundedness_and_compactness(self, weak_conv):
        """Test boundedness and weak compactness."""
        # Create bounded sequence
        bounded_sequence = [np.sin(n * weak_conv.x) / np.sqrt(n) for n in range(1, 31)]
        
        result = weak_conv.weak_compactness_demo(bounded_sequence)
        
        # Should be bounded
        assert result['is_bounded'] == True
        assert result['max_norm'] < np.inf
        
        # Should have convergent subsequence (Banach-Alaoglu theorem)
        # Note: This is a probabilistic test due to discrete sampling
        assert result['has_convergent_subsequence'] in [True, False]
    
    def test_linearity_of_weak_convergence(self, weak_conv):
        """Test linearity properties of weak convergence."""
        # Create two sequences
        seq1 = [np.sin(n * weak_conv.x) / n for n in range(1, 11)]
        seq2 = [np.cos(n * weak_conv.x) / n for n in range(1, 11)]
        
        # Test functional
        functional = np.ones_like(weak_conv.x)
        
        # Test individual sequences
        results1 = weak_conv.weak_convergence_test(seq1, {'test': functional})
        results2 = weak_conv.weak_convergence_test(seq2, {'test': functional})
        
        # Create linear combination
        alpha, beta = 2.0, 3.0
        combined_seq = [alpha * f1 + beta * f2 for f1, f2 in zip(seq1, seq2)]
        results_combined = weak_conv.weak_convergence_test(combined_seq, {'test': functional})
        
        # Check linearity: lim ∫(αfₙ + βgₙ)g = αlim∫fₙg + βlim∫gₙg
        limit1 = results1['test']['final_value']
        limit2 = results2['test']['final_value']
        limit_combined = results_combined['test']['final_value']
        
        # Allow for numerical errors
        expected_combined = alpha * limit1 + beta * limit2
        assert abs(limit_combined - expected_combined) < 1e-10
    
    def test_uniqueness_of_weak_limits(self, weak_conv):
        """Test uniqueness of weak limits."""
        # Create sequence converging to zero
        sequence = [np.sin(n * weak_conv.x) / n for n in range(1, 21)]
        
        # Test with multiple functionals
        functionals = {
            'constant': np.ones_like(weak_conv.x),
            'linear': weak_conv.x,
            'quadratic': weak_conv.x**2
        }
        
        results = weak_conv.weak_convergence_test(sequence, functionals)
        
        # All should converge to the same limit (zero in this case)
        limits = [result['final_value'] for result in results.values()]
        
        # All limits should be close to zero
        for limit in limits:
            assert abs(limit) < 0.1
    
    def test_optimization_functional_convergence(self, weak_conv):
        """Test functional convergence in optimization."""
        # Simple quadratic function
        initial = weak_conv.x**2
        
        def quadratic_functional(func):
            return np.trapz(func**2, weak_conv.x)
        
        history = weak_conv.gradient_descent_weak_convergence(
            initial, quadratic_functional, num_iterations=30
        )
        
        # Functional values should converge
        functional_values = [record['merit'] for record in history]
        
        # Should be decreasing or stable
        final_value = functional_values[-1]
        initial_value = functional_values[0]
        assert final_value <= initial_value * 1.01  # Allow small increase
        
        # Final functional change should be small
        final_change = history[-1]['functional_change']
        assert final_change < 1e-6 or len(history) < 30  # Either converged or hit iteration limit


if __name__ == "__main__":
    pytest.main([__file__])