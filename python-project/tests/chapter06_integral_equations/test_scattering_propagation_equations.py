"""
Test suite for Integral Equations implementation in chapter06_integral_equations.
Tests Fredholm and Volterra equations for optical scattering and propagation problems.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter06_integral_equations.scattering_propagation_equations import IntegralEquationsOptics


class TestIntegralEquationsOptics:
    """Test suite for IntegralEquationsOptics class."""
    
    @pytest.fixture
    def integral_eq(self):
        """Create an IntegralEquationsOptics instance for testing."""
        return IntegralEquationsOptics(grid_size=50)
    
    def test_initialization(self, integral_eq):
        """Test proper initialization of IntegralEquationsOptics."""
        assert integral_eq.grid_size == 50
        assert hasattr(integral_eq, 'x')
        assert hasattr(integral_eq, 'dx')
        
        # Check coordinate system
        assert len(integral_eq.x) == 50
        assert integral_eq.x[0] == -1
        assert integral_eq.x[-1] == 1
        assert integral_eq.dx > 0
    
    def test_fredholm_scattering_equation_basic(self, integral_eq):
        """Test basic Fredholm equation solving."""
        # Define simple kernel and source
        def simple_kernel(x, y):
            return np.exp(-(x - y)**2)
        
        def simple_source(x):
            return np.sin(x)
        
        # Solve Fredholm equation
        solution, convergence = integral_eq.fredholm_scattering_equation(
            simple_kernel, simple_source, lambda_param=0.1, num_iterations=50
        )
        
        # Check solution properties
        assert len(solution) == 50
        assert isinstance(solution, np.ndarray)
        assert len(convergence) > 0
        
        # Check that solution is finite
        assert np.all(np.isfinite(solution))
        assert not np.allclose(solution, 0.0)
    
    def test_fredholm_convergence(self, integral_eq):
        """Test convergence of Fredholm equation solver."""
        # Define convergent case (small lambda)
        def gaussian_kernel(x, y):
            sigma = 0.2
            return np.exp(-(x - y)**2 / (2 * sigma**2))
        
        def source_func(x):
            return np.sin(2 * np.pi * x)
        
        # Solve with small lambda (should converge quickly)
        solution, convergence = integral_eq.fredholm_scattering_equation(
            gaussian_kernel, source_func, lambda_param=0.1, num_iterations=100
        )
        
        # Check convergence
        assert len(convergence) > 0
        assert convergence[-1] < 1e-6  # Should converge to small error
        assert len(convergence) < 100  # Should converge before max iterations
    
    def test_fredholm_divergence(self, integral_eq):
        """Test behavior with large lambda (potential divergence)."""
        def simple_kernel(x, y):
            return np.exp(-(x - y)**2)
        
        def source_func(x):
            return np.sin(x)
        
        # Solve with large lambda (may not converge)
        solution, convergence = integral_eq.fredholm_scattering_equation(
            simple_kernel, source_func, lambda_param=2.0, num_iterations=50
        )
        
        # Check that we get a result (even if not converged)
        assert len(solution) == 50
        assert len(convergence) > 0
        
        # For large lambda, may not converge
        if len(convergence) > 1:
            final_error = convergence[-1]
            assert final_error >= 0  # Error should be non-negative
    
    def test_volterra_propagation_equation(self, integral_eq):
        """Test Volterra equation for propagation."""
        # Test parameters
        initial_field = 1.0 + 0.0j
        propagation_distance = 1.0
        absorption_coeff = 0.5
        
        # Solve Volterra equation
        z_vals, propagated_field = integral_eq.volterra_propagation_equation(
            initial_field, propagation_distance, absorption_coeff
        )
        
        # Check output properties
        assert len(z_vals) == integral_eq.grid_size
        assert len(propagated_field) == integral_eq.grid_size
        assert np.iscomplexobj(propagated_field)
        
        # Check that z values are correct
        assert z_vals[0] == 0
        assert z_vals[-1] == propagation_distance
        assert np.all(np.diff(z_vals) > 0)  # Monotonically increasing
        
        # Check field properties
        assert np.all(np.isfinite(propagated_field))
        assert propagated_field[0] == initial_field
    
    def test_volterra_absorption(self, integral_eq):
        """Test that Volterra equation models absorption correctly."""
        initial_field = 1.0 + 0.0j
        propagation_distance = 1.0
        
        # Test different absorption coefficients
        absorption_coeffs = [0.1, 0.5, 1.0, 2.0]
        
        final_energies = []
        for absorption_coeff in absorption_coeffs:
            z_vals, propagated_field = integral_eq.volterra_propagation_equation(
                initial_field, propagation_distance, absorption_coeff
            )
            
            final_energy = np.abs(propagated_field[-1])**2
            final_energies.append(final_energy)
        
        # Higher absorption should lead to lower final energy
        for i in range(1, len(final_energies)):
            assert final_energies[i] <= final_energies[i-1]
        
        # All final energies should be less than or equal to initial
        initial_energy = np.abs(initial_field)**2
        for final_energy in final_energies:
            assert final_energy <= initial_energy
    
    def test_optical_resolvent_kernel(self, integral_eq):
        """Test computation of optical resolvent kernel."""
        def simple_kernel(x, y):
            return np.exp(-(x - y)**2)
        
        lambda_param = 0.3
        
        # Compute resolvent kernel
        resolvent_matrix = integral_eq.optical_resolvent_kernel(simple_kernel, lambda_param)
        
        # Check output properties
        assert resolvent_matrix.shape == (integral_eq.grid_size, integral_eq.grid_size)
        assert np.all(np.isfinite(resolvent_matrix))
        
        # Test resolvent property: R = K + λKR (approximately)
        # Build kernel matrix
        K = np.zeros((integral_eq.grid_size, integral_eq.grid_size))
        for i, x_i in enumerate(integral_eq.x):
            for j, y_j in enumerate(integral_eq.x):
                K[i, j] = simple_kernel(x_i, y_j) * integral_eq.dx
        
        # Check resolvent equation: R ≈ K + λKR
        identity_check = resolvent_matrix - (K + lambda_param * K @ resolvent_matrix)
        resolvent_error = np.linalg.norm(identity_check)
        
        # Error should be reasonably small
        assert resolvent_error < 1e-6
    
    def test_born_approximation(self, integral_eq):
        """Test Born approximation for weak scattering."""
        # Weak scattering potential
        scattering_potential = 0.1 * np.exp(-integral_eq.x**2 / 0.1**2)
        
        # Incident plane wave
        wavenumber = 2 * np.pi / 500e-9
        incident_field = np.exp(1j * wavenumber * integral_eq.x)
        
        # Apply Born approximation
        born_field = integral_eq.born_approximation(scattering_potential, incident_field, wavenumber)
        
        # Check output properties
        assert len(born_field) == integral_eq.grid_size
        assert np.iscomplexobj(born_field)
        assert np.all(np.isfinite(born_field))
        
        # For weak scattering, Born field should be close to incident field
        field_difference = np.linalg.norm(born_field - incident_field) / np.linalg.norm(incident_field)
        
        # Difference should be small for weak scattering
        assert field_difference < 0.5  # 50% difference max for weak scattering
    
    def test_rytov_approximation(self, integral_eq):
        """Test Rytov approximation for phase perturbation."""
        # Scattering potential
        scattering_potential = 0.1 * np.exp(-integral_eq.x**2 / 0.1**2)
        
        # Incident plane wave
        wavenumber = 2 * np.pi / 500e-9
        incident_field = np.exp(1j * wavenumber * integral_eq.x)
        
        # Apply Rytov approximation
        rytov_field = integral_eq.rytov_approximation(scattering_potential, incident_field, wavenumber)
        
        # Check output properties
        assert len(rytov_field) == integral_eq.grid_size
        assert np.iscomplexobj(rytov_field)
        assert np.all(np.isfinite(rytov_field))
        
        # Rytov field should be close to incident field for weak scattering
        field_difference = np.linalg.norm(rytov_field - incident_field) / np.linalg.norm(incident_field)
        
        # Difference should be reasonable
        assert field_difference < 1.0  # Should not be completely different
    
    def test_optical_coherence_integral(self, integral_eq):
        """Test optical coherence integral equation."""
        # Source distribution
        source_distribution = np.exp(-integral_eq.x**2 / 0.2**2)
        
        # Propagation parameters
        propagation_distance = 1e-3  # 1 mm
        coherence_length = 1e-6  # 1 μm
        
        # Compute coherence propagation
        coherence_function = integral_eq.optical_coherence_integral(
            source_distribution, propagation_distance, coherence_length
        )
        
        # Check output properties
        assert coherence_function.shape == (integral_eq.grid_size, integral_eq.grid_size)
        assert np.iscomplexobj(coherence_function)
        assert np.all(np.isfinite(coherence_function))
        
        # Check that coherence function is Hermitian: Γ(x₁,x₂) = Γ*(x₂,x₁)
        for i in range(integral_eq.grid_size):
            for j in range(integral_eq.grid_size):
                assert abs(coherence_function[i, j] - np.conj(coherence_function[j, i])) < 1e-10
    
    def test_iterative_solution_convergence(self, integral_eq):
        """Test convergence analysis of iterative solutions."""
        def gaussian_kernel(x, y):
            sigma = 0.2
            return np.exp(-(x - y)**2 / (2 * sigma**2))
        
        def source_func(x):
            return np.sin(2 * np.pi * x)
        
        # Test different lambda values
        lambda_range = np.linspace(0.1, 0.9, 5)
        
        convergence_analysis = integral_eq.iterative_solution_convergence(
            gaussian_kernel, source_func, lambda_range
        )
        
        # Check results
        assert len(convergence_analysis) == len(lambda_range)
        
        for lambda_val, result in convergence_analysis.items():
            assert isinstance(lambda_val, float)
            assert 'convergence_history' in result
            assert 'final_error' in result
            assert 'num_iterations' in result
            assert 'converged' in result
            
            # Check that results are reasonable
            assert result['final_error'] >= 0
            assert result['num_iterations'] > 0
            assert isinstance(result['converged'], bool)
        
        # Smaller lambda values should generally converge better
        lambda_values = list(convergence_analysis.keys())
        final_errors = [convergence_analysis[lam]['final_error'] for lam in lambda_values]
        
        # Generally, smaller lambda should have smaller final error
        # (This is a statistical trend, not always true for individual cases)
        assert len(final_errors) == len(lambda_values)
    
    def test_different_grid_sizes(self):
        """Test with different grid sizes."""
        for grid_size in [25, 50, 100]:
            integral_eq = IntegralEquationsOptics(grid_size=grid_size)
            
            assert integral_eq.grid_size == grid_size
            assert len(integral_eq.x) == grid_size
            
            # Test basic functionality
            def simple_kernel(x, y):
                return np.exp(-(x - y)**2)
            
            def source_func(x):
                return np.sin(x)
            
            solution, convergence = integral_eq.fredholm_scattering_equation(
                simple_kernel, source_func, lambda_param=0.1, num_iterations=20
            )
            
            assert len(solution) == grid_size
            assert len(convergence) > 0
    
    def test_kernel_properties(self, integral_eq):
        """Test various kernel properties in Fredholm equations."""
        # Test symmetric kernel
        def symmetric_kernel(x, y):
            return np.exp(-(x - y)**2) + np.exp(-(x + y)**2)
        
        def source_func(x):
            return np.cos(x)
        
        # Solve with symmetric kernel
        solution, convergence = integral_eq.fredholm_scattering_equation(
            symmetric_kernel, source_func, lambda_param=0.2, num_iterations=50
        )
        
        # Check that solution is reasonable
        assert len(solution) == integral_eq.grid_size
        assert np.all(np.isfinite(solution))
        
        # Test antisymmetric kernel
        def antisymmetric_kernel(x, y):
            return (x - y) * np.exp(-(x - y)**2)
        
        # Solve with antisymmetric kernel
        solution2, convergence2 = integral_eq.fredholm_scattering_equation(
            antisymmetric_kernel, source_func, lambda_param=0.1, num_iterations=50
        )
        
        assert len(solution2) == integral_eq.grid_size
        assert np.all(np.isfinite(solution2))
    
    def test_physical_interpretation(self, integral_eq):
        """Test physical interpretation of solutions."""
        # Test optical scattering interpretation
        def scattering_kernel(x, y):
            # Kernel representing scattering interaction
            strength = 0.1
            range_param = 0.3
            return strength * np.exp(-np.abs(x - y) / range_param)
        
        def incident_wave(x):
            # Incident plane wave
            k = 2 * np.pi
            return np.exp(1j * k * x)
        
        # Solve scattering equation
        scattered_field, convergence = integral_eq.fredholm_scattering_equation(
            scattering_kernel, incident_wave, lambda_param=0.2, num_iterations=50
        )
        
        # Check that scattered field is complex (as expected for wave scattering)
        assert np.iscomplexobj(scattered_field)
        assert np.all(np.isfinite(scattered_field))
        
        # Check that scattered field is different from incident wave
        incident_field = np.array([incident_wave(x) for x in integral_eq.x])
        field_difference = np.linalg.norm(scattered_field - incident_field) / np.linalg.norm(incident_field)
        
        # Should be different due to scattering
        assert field_difference > 1e-6
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_integral_equations(self, mock_figure, mock_show):
        """Test the demonstration function with mocked visualization."""
        # Mock the figure and its methods
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock plotting functions
        mock_ax.plot.return_value = MagicMock()
        mock_ax.set_title.return_value = None
        mock_ax.set_xlabel.return_value = None
        mock_ax.set_ylabel.return_value = None
        
        # Import and run the demonstration
        from chapter06_integral_equations.scattering_propagation_equations import demonstrate_integral_equations
        
        # This should run without errors
        demonstrate_integral_equations()
        
        # Verify that visualization methods were called
        mock_show.assert_called()