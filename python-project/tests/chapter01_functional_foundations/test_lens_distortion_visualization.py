"""
Tests for lens_distortion_visualization.py - Chapter 01 Functional Foundations
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch

# Add the chapter directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'chapter01_functional_foundations'))

from lens_distortion_visualization import LensDistortionAnalyzer


class TestLensDistortionAnalyzer:
    """Test class for LensDistortionAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create an instance of LensDistortionAnalyzer for testing."""
        return LensDistortionAnalyzer()
    
    def test_initialization(self, analyzer):
        """Test proper initialization of LensDistortionAnalyzer."""
        assert hasattr(analyzer, 'lens_radius')
        assert hasattr(analyzer, 'lens_thickness')
        assert hasattr(analyzer, 'refractive_index')
        assert hasattr(analyzer, 'wavelength')
        
        # Check default values
        assert analyzer.lens_radius == 25.0
        assert analyzer.lens_thickness == 2.0
        assert analyzer.refractive_index == 1.5
        assert analyzer.wavelength == 0.55
    
    def test_spherical_surface(self, analyzer):
        """Test spherical surface generation."""
        # Test with simple coordinates
        x, y = 1.0, 2.0
        R = 10.0
        
        sag = analyzer.spherical_surface(x, y, R)
        
        # Expected: sag = (x² + y²) / (2R) = (1 + 4) / 20 = 5/20 = 0.25
        expected = (x**2 + y**2) / (2 * R)
        assert abs(sag - expected) < 1e-10
        
        # Test with arrays
        x_array = np.array([1.0, 2.0])
        y_array = np.array([0.5, 1.5])
        sag_array = analyzer.spherical_surface(x_array, y_array, R)
        
        expected_array = (x_array**2 + y_array**2) / (2 * R)
        np.testing.assert_allclose(sag_array, expected_array)
    
    def test_compute_surface_normal(self, analyzer):
        """Test surface normal computation."""
        # Test with simple spherical surface
        def test_surface(x, y):
            return (x**2 + y**2) / 20  # Spherical surface with R=10
        
        x, y = 1.0, 2.0
        normal = analyzer.compute_surface_normal(x, y, test_surface)
        
        # Check that normal is a unit vector
        assert len(normal) == 3
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6
        
        # For a spherical surface z = (x² + y²)/(2R), 
        # the normal should be proportional to (-x, -y, R)
        # At (1, 2), normal should be approximately (-1, -2, 10) normalized
        expected_direction = np.array([-x, -y, 10])
        expected_normal = expected_direction / np.linalg.norm(expected_direction)
        
        # Check that computed normal is in the right direction
        dot_product = np.dot(normal, expected_normal)
        assert dot_product > 0.9  # Should be nearly parallel
    
    def test_apply_snells_law(self, analyzer):
        """Test Snell's law application."""
        # Test with normal incidence
        incident_angle = 0.0
        normal = np.array([0, 0, 1])
        transmitted_angle = analyzer.apply_snells_law(incident_angle, normal)
        
        # For normal incidence, transmitted angle should be 0
        assert abs(transmitted_angle) < 1e-10
        
        # Test with oblique incidence
        incident_angle = np.pi / 6  # 30 degrees
        transmitted_angle = analyzer.apply_snells_law(incident_angle, normal)
        
        # Check that transmitted angle is different from incident angle
        assert abs(transmitted_angle - incident_angle) > 1e-6
        
        # Test with different refractive indices
        n1, n2 = 1.0, 1.5
        transmitted_angle_custom = analyzer.apply_snells_law(incident_angle, normal, n1, n2)
        
        # Manual calculation of expected angle
        sin_theta2 = (n1 / n2) * np.sin(incident_angle)
        expected_angle = np.arctan2(sin_theta2, np.sqrt(1 - sin_theta2**2))
        assert abs(transmitted_angle_custom - expected_angle) < 1e-10
    
    def test_compute_optical_path_difference(self, analyzer):
        """Test optical path difference computation."""
        # Create test ray positions
        ray_positions = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
        
        def test_surface(x, y):
            return (x**2 + y**2) / 20
        
        opd_values = analyzer.compute_optical_path_difference(ray_positions, test_surface)
        
        # Check that OPD values are computed for each ray
        assert len(opd_values) == len(ray_positions)
        
        # Check that OPD values are reasonable
        for opd in opd_values:
            assert isinstance(opd, (int, float, np.number))
            assert opd >= 0  # OPD should be non-negative for this test case
        
        # Check that OPD increases with radial distance
        # (since surface sag increases with distance from center)
        for i in range(1, len(opd_values)):
            radial_dist_i = np.sqrt(ray_positions[i][0]**2 + ray_positions[i][1]**2)
            radial_dist_j = np.sqrt(ray_positions[i-1][0]**2 + ray_positions[i-1][1]**2)
            if radial_dist_i > radial_dist_j:
                assert opd_values[i] >= opd_values[i-1] - 1e-6
    
    def test_ray_trace_through_lens(self, analyzer):
        """Test ray tracing through lens."""
        # Simple test case
        ray_position = np.array([5.0, 0.0])
        ray_angle = np.pi / 12  # 15 degrees
        
        def test_surface(x, y):
            return (x**2 + y**2) / 20
        
        z_intersect, transmitted_angle = analyzer.ray_trace_through_lens(ray_position, ray_angle, test_surface)
        
        # Check that intersection height is computed
        assert isinstance(z_intersect, (int, float, np.number))
        
        # Check that transmitted angle is computed
        assert isinstance(transmitted_angle, (int, float, np.number))
        
        # For a convex surface, transmitted angle should be different from incident angle
        assert abs(transmitted_angle - ray_angle) > 1e-6
    
    def test_optimized_functional_surface(self, analyzer):
        """Test optimized functional surface generation."""
        # Create test coordinates
        X = np.array([[0, 1, 2], [3, 4, 5]])
        Y = np.array([[0, 0.5, 1], [1.5, 2, 2.5]])
        
        surface = analyzer.optimized_functional_surface(X, Y)
        
        # Check surface shape
        assert surface.shape == X.shape
        
        # Check that surface values are reasonable
        assert np.all(np.isfinite(surface))
        
        # Check that surface is smooth (no extreme variations)
        surface_max = np.max(np.abs(surface))
        assert surface_max < 10.0  # Should be within reasonable bounds
    
    def test_discrete_optimized_surface(self, analyzer):
        """Test discrete optimized surface generation."""
        # Create test coordinates
        X = np.array([[0, 1, 2], [3, 4, 5]])
        Y = np.array([[0, 0.5, 1], [1.5, 2, 2.5]])
        
        surface = analyzer.discrete_optimized_surface(X, Y)
        
        # Check surface shape
        assert surface.shape == X.shape
        
        # Check that surface values are reasonable
        assert np.all(np.isfinite(surface))
        
        # Should be a spherical surface
        # Check that it follows spherical surface equation
        R = 30.0  # Expected radius from the implementation
        expected_surface = (X**2 + Y**2) / (2 * R)
        np.testing.assert_allclose(surface, expected_surface)
    
    def test_continuous_thinking_demonstration(self, analyzer):
        """Test continuous thinking demonstration."""
        # Test the mathematical models used in continuous thinking
        field_angles = np.linspace(0, 20, 21)
        
        # Test discrete approach model
        discrete_performance = []
        for angle in field_angles:
            def angle_merit(params):
                return (params[0] - angle)**2 + 0.1 * angle**2
            
            from scipy.optimize import minimize
            result = minimize(angle_merit, [angle])
            discrete_performance.append(result.fun)
        
        # Check that discrete performance values are reasonable
        assert len(discrete_performance) == len(field_angles)
        for perf in discrete_performance:
            assert perf >= 0
            assert isinstance(perf, (int, float, np.number))
        
        # Test functional approach model
        def functional_field_merit(params):
            a, b, c = params
            total_merit = 0
            
            for angle in field_angles:
                predicted_performance = a + b*angle + c*angle**2
                total_merit += (predicted_performance - 0)**2 + 0.01 * angle**2
            
            return total_merit
        
        # Test optimization
        from scipy.optimize import minimize
        result_functional = minimize(functional_field_merit, [0, 0, 0])
        a_opt, b_opt, c_opt = result_functional.x
        
        # Check that optimization was successful
        assert result_functional.success or result_functional.status == 0
        
        # Check that optimized parameters are reasonable
        assert np.all(np.isfinite([a_opt, b_opt, c_opt]))
        
        # Test functional performance
        functional_performance = []
        for angle in field_angles:
            functional_performance.append(a_opt + b_opt*angle + c_opt*angle**2)
        
        assert len(functional_performance) == len(field_angles)
        for perf in functional_performance:
            assert isinstance(perf, (int, float, np.number))
    
    @pytest.mark.visualization
    def test_visualization_methods(self, analyzer, mock_matplotlib_show):
        """Test that visualization methods can be called without errors."""
        try:
            # Test lens distortion visualization
            surfaces = analyzer.visualize_lens_distortion()
            
            # Check that surfaces is returned
            assert isinstance(surfaces, dict)
            assert len(surfaces) > 0
            
            # Test continuous thinking demonstration (mock the plotting)
            with patch('matplotlib.pyplot.figure'), \
                 patch('matplotlib.pyplot.subplot'), \
                 patch('matplotlib.pyplot.plot'), \
                 patch('matplotlib.pyplot.xlabel'), \
                 patch('matplotlib.pyplot.ylabel'), \
                 patch('matplotlib.pyplot.title'), \
                 patch('matplotlib.pyplot.legend'), \
                 patch('matplotlib.pyplot.grid'), \
                 patch('matplotlib.pyplot.tight_layout'), \
                 patch('matplotlib.pyplot.show'):
                
                analyzer.demonstrate_continuous_thinking()
                
            assert True
        except Exception as e:
            pytest.fail(f"Visualization method failed: {e}")


class TestMathematicalProperties:
    """Test mathematical properties and relationships."""
    
    def test_spherical_surface_properties(self, analyzer):
        """Test mathematical properties of spherical surface."""
        # Test that spherical surface has correct mathematical form
        x_vals = np.array([0, 1, 2, 3, 4])
        y_vals = np.array([0, 0, 0, 0, 0])
        R = 20.0
        
        sag = analyzer.spherical_surface(x_vals, y_vals, R)
        
        # Expected: sag = x²/(2R) when y=0
        expected = x_vals**2 / (2 * R)
        np.testing.assert_allclose(sag, expected)
        
        # Test rotational symmetry
        x_sym = np.array([3, 0, -3, 0])
        y_sym = np.array([0, 3, 0, -3])
        sag_sym = analyzer.spherical_surface(x_sym, y_sym, R)
        
        # All values should be the same (3² + 0² = 0² + 3² = 9)
        expected_sym = 9.0 / (2 * R)
        np.testing.assert_allclose(sag_sym, expected_sym)
    
    def test_surface_normal_properties(self, analyzer):
        """Test mathematical properties of surface normal."""
        # Test that surface normal is a unit vector
        def test_surface(x, y):
            return (x**2 + y**2) / 20
        
        test_points = [(0, 0), (1, 0), (0, 1), (2, 2)]
        
        for x, y in test_points:
            normal = analyzer.compute_surface_normal(x, y, test_surface)
            
            # Should be a unit vector
            norm = np.linalg.norm(normal)
            assert abs(norm - 1.0) < 1e-10
            
            # Should point in the correct direction
            # For a paraboloid z = (x² + y²)/(2R), normal ∝ (-x, -y, R)
            expected_direction = np.array([-x, -y, 10])  # R = 10
            expected_normal = expected_direction / np.linalg.norm(expected_direction)
            
            # Should be parallel (dot product ≈ ±1)
            dot_product = np.dot(normal, expected_normal)
            assert abs(abs(dot_product) - 1.0) < 0.1
    
    def test_snells_law_properties(self, analyzer):
        """Test mathematical properties of Snell's law."""
        # Test conservation of Snell's law
        n1, n2 = 1.0, 1.5
        normal = np.array([0, 0, 1])
        
        # Test various incident angles
        incident_angles = np.array([0, np.pi/12, np.pi/6, np.pi/4])
        
        for theta1 in incident_angles:
            theta2 = analyzer.apply_snells_law(theta1, normal, n1, n2)
            
            # Check Snell's law: n1*sin(θ1) = n2*sin(θ2)
            snell_check = n1 * np.sin(theta1) - n2 * np.sin(theta2)
            assert abs(snell_check) < 1e-10
        
        # Test total internal reflection condition
        # When n1 > n2 and θ1 > critical angle
        n1, n2 = 1.5, 1.0  # Light going from dense to sparse medium
        critical_angle = np.arcsin(n2 / n1)
        
        # Test just below critical angle
        theta1_below = critical_angle * 0.9
        theta2_below = analyzer.apply_snells_law(theta1_below, normal, n1, n2)
        assert np.isfinite(theta2_below)
        
        # Test at critical angle
        theta1_critical = critical_angle
        theta2_critical = analyzer.apply_snells_law(theta1_critical, normal, n1, n2)
        assert abs(theta2_critical - np.pi/2) < 1e-6  # Should be 90 degrees


if __name__ == "__main__":
    pytest.main([__file__])