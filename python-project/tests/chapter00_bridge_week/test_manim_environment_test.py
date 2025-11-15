"""
Tests for manim_environment_test.py - Chapter 00 Bridge Week
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch, MagicMock

# Add the chapter directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'chapter00_bridge_week'))

# Mock Manim if not available
try:
    from manim import *
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    # Create mock classes for testing
    class Scene:
        def construct(self):
            pass
    
    class Line:
        def __init__(self, start, end, color=None, stroke_width=None):
            self.start = start
            self.end = end
            self.color = color
            self.stroke_width = stroke_width
    
    class Text:
        def __init__(self, text, font_size=None):
            self.text = text
            self.font_size = font_size
        
        def next_to(self, obj, direction):
            return self
    
    class Axes:
        def __init__(self, x_range=None, y_range=None, x_length=None, y_length=None, axis_config=None):
            self.x_range = x_range
            self.y_range = y_range
            self.x_length = x_length
            self.y_length = y_length
            self.axis_config = axis_config
        
        def plot(self, func, color=None, stroke_width=None):
            return MagicMock()
        
        def get_x_axis_label(self, label):
            return Text(label)
        
        def get_y_axis_label(self, label):
            return Text(label)
        
        def c2p(self, x, y):
            return (x, y, 0)
    
    class VMobject:
        def set_points_as_corners(self, points):
            self.points = points
        
        def set_color(self, color):
            self.color = color
        
        def set_stroke(self, width):
            self.stroke_width = width
    
    class VGroup:
        def __init__(self):
            self.objects = []
        
        def add(self, obj):
            self.objects.append(obj)
    
    class Dot:
        def __init__(self, point=None, color=None, radius=None):
            self.point = point
            self.color = color
            self.radius = radius
    
    class NumberPlane:
        def __init__(self, x_range=None, y_range=None):
            self.x_range = x_range
            self.y_range = y_range

from manim_environment_test import OpticalManimDemo


class TestOpticalManimDemo:
    """Test class for OpticalManimDemo class"""
    
    @pytest.fixture
    def demo(self):
        """Create an instance of OpticalManimDemo for testing."""
        # Mock the show methods to prevent actual plots from appearing
        with patch('matplotlib.pyplot.show'):
            return OpticalManimDemo()
    
    def test_initialization(self, demo):
        """Test proper initialization of OpticalManimDemo."""
        assert hasattr(demo, 'create_static_demo')
        assert hasattr(demo, 'show_animation_examples')
        assert callable(demo.create_static_demo)
        assert callable(demo.show_animation_examples)
    
    def test_static_demo_creation(self):
        """Test static demo creation."""
        demo = OpticalManimDemo()
        
        # Mock matplotlib to prevent actual plots
        with patch('matplotlib.pyplot.figure') as mock_fig, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'):
            
            # Mock the subplot axes
            mock_axes = [MagicMock() for _ in range(4)]
            mock_fig_instance = MagicMock()
            mock_fig.return_value = mock_fig_instance
            mock_subplots.return_value = (mock_fig_instance, mock_axes)
            
            # This should not raise an exception
            try:
                demo.create_static_demo()
                assert True
            except Exception as e:
                pytest.fail(f"Static demo creation failed: {e}")
    
    def test_light_ray_data_generation(self):
        """Test light ray data generation for static demo."""
        # Test the data that would be used for plotting
        x = np.linspace(0, 10, 100)
        y1 = np.zeros_like(x)
        y2 = np.sin(0.5 * x)
        
        # Check that the data is reasonable
        assert len(x) == 100
        assert len(y1) == 100
        assert len(y2) == 100
        assert np.allclose(y1, 0)  # First ray should be straight
        assert np.max(np.abs(y2)) <= 1  # Sine wave should be bounded
    
    def test_lens_surface_data_generation(self):
        """Test lens surface data generation."""
        # Test the data that would be used for plotting
        x_lens = np.linspace(-2, 2, 100)
        y_lens = x_lens**2 / 4  # Parabolic lens
        
        # Check that the data is reasonable
        assert len(x_lens) == 100
        assert len(y_lens) == 100
        assert y_lens[0] == 1.0  # y = x²/4 at x = -2
        assert y_lens[-1] == 1.0  # y = x²/4 at x = 2
        assert y_lens[len(x_lens)//2] == 0  # Minimum at x = 0
    
    def test_wavefront_error_data_generation(self):
        """Test wavefront error data generation."""
        # Test the data that would be used for plotting
        x_wave = np.linspace(-3, 3, 50)
        y_wave = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_wave, y_wave)
        wavefront = 0.5 * (X**2 + Y**2) + 0.2 * np.sin(X) * np.cos(Y)
        
        # Check that the data is reasonable
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert wavefront.shape == (50, 50)
        
        # Check some properties
        assert np.min(wavefront) < np.max(wavefront)  # Should have variation
        assert wavefront[len(x_wave)//2, len(y_wave)//2] == 0  # Should be zero at center
    
    def test_optimization_data_generation(self):
        """Test optimization convergence data generation."""
        # Test the data that would be used for plotting
        iterations = np.arange(1, 21)
        error = 10 * np.exp(-iterations/5) + 0.1 * np.random.randn(20)
        
        # Check that the data is reasonable
        assert len(iterations) == 20
        assert len(error) == 20
        assert iterations[0] == 1
        assert iterations[-1] == 20
        assert np.all(error > 0)  # Error should be positive
        assert error[0] > error[-1]  # Error should generally decrease
    
    @pytest.mark.manim
    def test_manim_scene_classes(self):
        """Test Manim scene classes if available."""
        if MANIM_AVAILABLE:
            # Test that scene classes can be instantiated
            try:
                from manim_environment_test import LightRayAnimation, LensSurfaceCreation
                
                # These should be available classes
                assert hasattr(LightRayAnimation, 'construct')
                assert hasattr(LensSurfaceCreation, 'construct')
                
                # Test instantiation (construct method will be called)
                light_scene = LightRayAnimation()
                lens_scene = LensSurfaceCreation()
                
                # If we get here, the classes work
                assert True
            except Exception as e:
                pytest.fail(f"Manim scene class test failed: {e}")
        else:
            pytest.skip("Manim not available")
    
    def test_animation_examples_output(self, demo):
        """Test that animation examples produce output."""
        # Capture print output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            demo.show_animation_examples()
        
        output = f.getvalue()
        
        # Check that expected content is in the output
        assert "Light Ray Propagation Animation:" in output
        assert "Lens Surface Creation(Scene):" in output
        assert "Wavefront Error Visualization(Scene):" in output
        assert "Optimization Convergence Animation:" in output
    
    def test_manim_availability_detection(self):
        """Test Manim availability detection."""
        # Test that the module correctly detects Manim availability
        import manim_environment_test
        
        # The MANIM_AVAILABLE constant should be set correctly
        if MANIM_AVAILABLE:
            assert manim_environment_test.MANIM_AVAILABLE is True
        else:
            assert manim_environment_test.MANIM_AVAILABLE is False


class TestMathematicalConcepts:
    """Test mathematical concepts used in the visualizations."""
    
    def test_parabolic_lens_equation(self):
        """Test parabolic lens equation."""
        # Test the mathematical relationship used for parabolic lens
        x_vals = np.array([-2, -1, 0, 1, 2])
        expected_sag = x_vals**2 / 4
        
        # This should create a parabola with vertex at origin
        assert expected_sag[2] == 0  # Minimum at x = 0
        assert expected_sag[0] == expected_sag[-1]  # Symmetric
        assert expected_sag[0] == 1.0  # y = 1 at x = ±2
    
    def test_wavefront_error_composition(self):
        """Test wavefront error composition."""
        # Test the mathematical composition of wavefront error
        X = np.array([[0, 1], [2, 3]])
        Y = np.array([[0, 0.5], [1, 1.5]])
        
        # Reconstruct the wavefront error equation
        defocus = 0.5 * (X**2 + Y**2)
        astigmatism = 0.3 * (X**2 - Y**2)
        coma = 0.2 * (X**2 + Y**2) * X
        spherical = 0.1 * (X**2 + Y**2)**2
        
        total_wavefront = defocus + astigmatism + coma + spherical
        
        # Check that the composition is correct
        assert total_wavefront.shape == X.shape
        assert np.all(total_wavefront >= 0)  # Should be positive for these test values
    
    def test_optimization_decay_model(self):
        """Test optimization decay model."""
        # Test the mathematical model used for optimization convergence
        iterations = np.arange(0, 21)
        base_error = 10 * np.exp(-iterations/5)
        
        # Check properties of the exponential decay
        assert base_error[0] == 10.0  # Initial value
        assert base_error[-1] < 1.0  # Should decay significantly
        assert np.all(np.diff(base_error) <= 0)  # Should be monotonically decreasing


if __name__ == "__main__":
    pytest.main([__file__])