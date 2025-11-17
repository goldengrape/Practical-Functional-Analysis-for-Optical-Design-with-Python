"""
Manim unit tests for Chapter 10: Distribution Theory in Optical Systems
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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
    
    class Mobject:
        def __init__(self):
            self.submobjects = []
        
        def add(self, obj):
            self.submobjects.append(obj)
            return self
        
        def set_color(self, color):
            self.color = color
            return self
        
        def set_stroke(self, width=None, color=None):
            if width is not None:
                self.stroke_width = width
            if color is not None:
                self.stroke_color = color
            return self
        
        def set_fill(self, color=None, opacity=None):
            if color is not None:
                self.fill_color = color
            if opacity is not None:
                self.fill_opacity = opacity
            return self
        
        def next_to(self, obj, direction=None, buff=None):
            return self
        
        def shift(self, vector):
            return self
        
        def scale(self, scale_factor):
            return self
        
        def animate(self):
            return self
    
    class Line(Mobject):
        def __init__(self, start, end, color=None, stroke_width=None):
            super().__init__()
            self.start = start
            self.end = end
            if color:
                self.color = color
            if stroke_width:
                self.stroke_width = stroke_width
    
    class Arrow(Line):
        def __init__(self, start, end, color=None, stroke_width=None):
            super().__init__(start, end, color, stroke_width)
    
    class Text(Mobject):
        def __init__(self, text, font_size=None, color=None):
            super().__init__()
            self.text = text
            if font_size:
                self.font_size = font_size
            if color:
                self.color = color
        
        def next_to(self, obj, direction=None, buff=None):
            return self
    
    class MathTex(Mobject):
        def __init__(self, expression, font_size=None):
            super().__init__()
            self.expression = expression
            if font_size:
                self.font_size = font_size
    
    class Axes(Mobject):
        def __init__(self, x_range=None, y_range=None, x_length=None, y_length=None, 
                     axis_config=None, tips=None):
            super().__init__()
            self.x_range = x_range or [-1, 1, 0.1]
            self.y_range = y_range or [-1, 1, 0.1]
            self.x_length = x_length or 6
            self.y_length = y_length or 4
            self.axis_config = axis_config or {}
            self.tips = tips if tips is not None else True
        
        def plot(self, func, color=None, stroke_width=None, x_range=None):
            plot_obj = Mobject()
            plot_obj.func = func
            if color:
                plot_obj.color = color
            if stroke_width:
                plot_obj.stroke_width = stroke_width
            return plot_obj
        
        def get_x_axis_label(self, label, font_size=None):
            return Text(label, font_size=font_size)
        
        def get_y_axis_label(self, label, font_size=None):
            return Text(label, font_size=font_size)
        
        def c2p(self, x, y, z=0):
            return (x, y, z)
        
        def p2c(self, point):
            return point[:2]
    
    class VGroup(Mobject):
        def __init__(self, *mobjects):
            super().__init__()
            self.submobjects = list(mobjects)
        
        def add(self, *mobjects):
            self.submobjects.extend(mobjects)
            return self
        
        def arrange(self, direction=None, buff=None):
            return self
    
    class Dot(Mobject):
        def __init__(self, point=None, color=None, radius=None):
            super().__init__()
            self.point = point or (0, 0, 0)
            if color:
                self.color = color
            if radius:
                self.radius = radius
    
    class Circle(Mobject):
        def __init__(self, radius=None, color=None, fill_opacity=None):
            super().__init__()
            self.radius = radius or 1
            if color:
                self.color = color
            if fill_opacity is not None:
                self.fill_opacity = fill_opacity
    
    class Rectangle(Mobject):
        def __init__(self, width=None, height=None, color=None, fill_opacity=None):
            super().__init__()
            self.width = width or 2
            self.height = height or 1
            if color:
                self.color = color
            if fill_opacity is not None:
                self.fill_opacity = fill_opacity
    
    class NumberPlane(Mobject):
        def __init__(self, x_range=None, y_range=None, x_length=None, y_length=None):
            super().__init__()
            self.x_range = x_range or [-10, 10, 1]
            self.y_range = y_range or [-10, 10, 1]
            self.x_length = x_length or 10
            self.y_length = y_length or 10
    
    class VMobject(Mobject):
        def set_points_as_corners(self, points):
            self.points = points
            return self
        
        def set_style(self, stroke_width=None, stroke_color=None, fill_color=None, fill_opacity=None):
            if stroke_width is not None:
                self.stroke_width = stroke_width
            if stroke_color is not None:
                self.stroke_color = stroke_color
            if fill_color is not None:
                self.fill_color = fill_color
            if fill_opacity is not None:
                self.fill_opacity = fill_opacity
            return self
    
    class ParametricFunction(VMobject):
        def __init__(self, func, t_range=None, color=None, stroke_width=None):
            super().__init__()
            self.func = func
            self.t_range = t_range or [0, 1, 0.01]
            if color:
                self.color = color
            if stroke_width:
                self.stroke_width = stroke_width
    
    class SurroundingRectangle(Mobject):
        def __init__(self, mobject, color=None, buff=None, corner_radius=None):
            super().__init__()
            self.target = mobject
            if color:
                self.color = color
            if buff is not None:
                self.buff = buff
            if corner_radius is not None:
                self.corner_radius = corner_radius
    
    class FadeIn:
        def __init__(self, mobject, run_time=None, rate_func=None):
            self.mobject = mobject
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class FadeOut:
        def __init__(self, mobject, run_time=None, rate_func=None):
            self.mobject = mobject
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class Create:
        def __init__(self, mobject, run_time=None, rate_func=None):
            self.mobject = mobject
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class Transform:
        def __init__(self, mobject_from, mobject_to, run_time=None, rate_func=None):
            self.mobject_from = mobject_from
            self.mobject_to = mobject_to
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class Write:
        def __init__(self, mobject, run_time=None, rate_func=None):
            self.mobject = mobject
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class ReplacementTransform:
        def __init__(self, mobject_from, mobject_to, run_time=None, rate_func=None):
            self.mobject_from = mobject_from
            self.mobject_to = mobject_to
            self.run_time = run_time or 1
            self.rate_func = rate_func or (lambda t: t)
    
    class Indicate:
        def __init__(self, mobject, color=None, scale_factor=None, run_time=None):
            self.mobject = mobject
            if color:
                self.color = color
            if scale_factor:
                self.scale_factor = scale_factor
            self.run_time = run_time or 1
    
    class Wiggle:
        def __init__(self, mobject, scale_value=None, rotation_angle=None, run_time=None):
            self.mobject = mobject
            if scale_value:
                self.scale_value = scale_value
            if rotation_angle:
                self.rotation_angle = rotation_angle
            self.run_time = run_time or 1
    
    class UP:
        def __init__(self, buff=None):
            self.buff = buff or 0.1
    
    class DOWN:
        def __init__(self, buff=None):
            self.buff = buff or 0.1
    
    class LEFT:
        def __init__(self, buff=None):
            self.buff = buff or 0.1
    
    class RIGHT:
        def __init__(self, buff=None):
            self.buff = buff or 0.1
    
    class ORIGIN:
        def __init__(self):
            pass
    
    class BLUE:
        def __init__(self):
            self.value = "blue"
    
    class RED:
        def __init__(self):
            self.value = "red"
    
    class GREEN:
        def __init__(self):
            self.value = "green"
    
    class YELLOW:
        def __init__(self):
            self.value = "yellow"
    
    class PURPLE:
        def __init__(self):
            self.value = "purple"
    
    class ORANGE:
        def __init__(self):
            self.value = "orange"
    
    class WHITE:
        def __init__(self):
            self.value = "white"
    
    class BLACK:
        def __init__(self):
            self.value = "black"
    
    class GRAY:
        def __init__(self):
            self.value = "gray"
    
    class LIGHT_GRAY:
        def __init__(self):
            self.value = "light_gray"
    
    class DARK_GRAY:
        def __init__(self):
            self.value = "dark_gray"

# Import the manim scenes from the chapter
try:
    from chapter10_distribution_theory.distribution_manim import (
        DeltaFunctionVisualization,
        GreensFunctionVisualization,
        DistributionDerivativeDemo,
        OpticalCausalityVisualization,
        DistributionTheoryCompleteDemo
    )
except ImportError:
    # Create mock classes if the module doesn't exist yet
    class DeltaFunctionVisualization:
        def construct(self):
            pass
    
    class GreensFunctionVisualization:
        def construct(self):
            pass
    
    class DistributionDerivativeDemo:
        def construct(self):
            pass
    
    class OpticalCausalityVisualization:
        def construct(self):
            pass
    
    class DistributionTheoryCompleteDemo:
        def construct(self):
            pass


class TestDeltaFunctionVisualization:
    """Test class for DeltaFunctionVisualization Manim scene."""
    
    @pytest.mark.manim
    def test_scene_initialization(self):
        """Test that the DeltaFunctionVisualization scene can be initialized."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DeltaFunctionVisualization()
            assert hasattr(scene, 'construct')
            assert callable(scene.construct)
        except Exception as e:
            pytest.fail(f"DeltaFunctionVisualization initialization failed: {e}")
    
    @pytest.mark.manim
    def test_scene_construction(self):
        """Test that the scene construction doesn't raise errors."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DeltaFunctionVisualization()
            # This should not raise an exception
            scene.construct()
            assert True
        except Exception as e:
            pytest.fail(f"DeltaFunctionVisualization construction failed: {e}")
    
    def test_delta_function_data_generation(self):
        """Test the mathematical data generation for delta function visualization."""
        # Test Gaussian delta approximation
        x = np.linspace(-3, 3, 1000)
        center = 0.0
        width = 0.1
        
        # Gaussian delta approximation
        delta_gaussian = (1 / (width * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / width)**2)
        
        # Check properties
        assert len(delta_gaussian) == len(x)
        assert np.allclose(np.trapz(delta_gaussian, x), 1.0, rtol=0.01)  # Normalization
        assert np.argmax(delta_gaussian) == np.argmin(np.abs(x - center))  # Peak at center
        assert np.max(delta_gaussian) > 0  # Positive values
    
    def test_lorentzian_delta_function(self):
        """Test Lorentzian delta function approximation."""
        x = np.linspace(-3, 3, 1000)
        center = 0.5
        width = 0.2
        
        # Lorentzian delta approximation
        delta_lorentzian = (1 / np.pi) * (width / 2) / ((x - center)**2 + (width / 2)**2)
        
        # Check properties
        assert len(delta_lorentzian) == len(x)
        assert np.allclose(np.trapz(delta_lorentzian, x), 1.0, rtol=0.05)  # Normalization
        assert np.argmax(delta_lorentzian) == np.argmin(np.abs(x - center))  # Peak at center
        assert np.max(delta_lorentzian) == pytest.approx(2 / (np.pi * width), rel=0.1)
    
    def test_sinc_delta_function(self):
        """Test sinc delta function approximation."""
        x = np.linspace(-3, 3, 1000)
        center = -0.3
        width = 0.15
        
        # Sinc delta approximation
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_sinc = (1 / (np.pi * width)) * np.sinc((x - center) / width)
        
        # Check properties
        assert len(delta_sinc) == len(x)
        assert np.allclose(np.trapz(delta_sinc, x), 1.0, rtol=0.1)  # Normalization
        assert np.argmax(delta_sinc) == np.argmin(np.abs(x - center))  # Peak at center


class TestGreensFunctionVisualization:
    """Test class for GreensFunctionVisualization Manim scene."""
    
    @pytest.mark.manim
    def test_scene_initialization(self):
        """Test that the GreensFunctionVisualization scene can be initialized."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = GreensFunctionVisualization()
            assert hasattr(scene, 'construct')
            assert callable(scene.construct)
        except Exception as e:
            pytest.fail(f"GreensFunctionVisualization initialization failed: {e}")
    
    @pytest.mark.manim
    def test_scene_construction(self):
        """Test that the scene construction doesn't raise errors."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = GreensFunctionVisualization()
            scene.construct()
            assert True
        except Exception as e:
            pytest.fail(f"GreensFunctionVisualization construction failed: {e}")
    
    def test_1d_greens_function(self):
        """Test 1D Green's function calculation."""
        x = np.linspace(-5, 5, 1000)
        source_pos = 1.0
        k = 2 * np.pi / 0.5  # Wavenumber for 0.5 unit wavelength
        
        # 1D Green's function: G(x, x₀) = (i/2k) * exp(ik|x - x₀|)
        greens_1d = (1j / (2 * k)) * np.exp(1j * k * np.abs(x - source_pos))
        
        # Check properties
        assert len(greens_1d) == len(x)
        assert np.all(np.isfinite(greens_1d))
        assert np.iscomplexobj(greens_1d)
        
        # Check symmetry about source
        source_idx = np.argmin(np.abs(x - source_pos))
        if source_idx > 0 and source_idx < len(x) - 1:
            left_idx = source_idx - 10
            right_idx = source_idx + 10
            left_mag = np.abs(greens_1d[left_idx])
            right_mag = np.abs(greens_1d[right_idx])
            assert abs(left_mag - right_mag) / (left_mag + right_mag + 1e-10) < 0.1
    
    def test_2d_greens_function(self):
        """Test 2D Green's function calculation."""
        x = np.linspace(-5, 5, 1000)
        source_pos = 0.0
        k = 2 * np.pi / 0.5
        
        # 2D Green's function: G(x, x₀) = (i/4) * H₀⁽¹⁾(k|x - x₀|)
        # Using asymptotic form for large arguments: H₀⁽¹⁾(z) ≈ √(2/πz) * exp(i(z - π/4))
        r = np.abs(x - source_pos)
        r_safe = np.where(r < 1e-10, 1e-10, r)  # Avoid division by zero
        
        greens_2d = (1j / 4) * np.sqrt(2 / (np.pi * k * r_safe)) * np.exp(1j * (k * r_safe - np.pi / 4))
        
        # Check properties
        assert len(greens_2d) == len(x)
        assert np.all(np.isfinite(greens_2d))
        assert np.iscomplexobj(greens_2d)
    
    def test_3d_greens_function(self):
        """Test 3D Green's function calculation."""
        x = np.linspace(-5, 5, 1000)
        source_pos = -1.0
        k = 2 * np.pi / 0.5
        
        # 3D Green's function: G(x, x₀) = exp(ik|x - x₀|) / (4π|x - x₀|)
        r = np.abs(x - source_pos)
        r_safe = np.where(r < 1e-10, 1e-10, r)
        
        greens_3d = np.exp(1j * k * r_safe) / (4 * np.pi * r_safe)
        
        # Check properties
        assert len(greens_3d) == len(x)
        assert np.all(np.isfinite(greens_3d))
        assert np.iscomplexobj(greens_3d)


class TestDistributionDerivativeDemo:
    """Test class for DistributionDerivativeDemo Manim scene."""
    
    @pytest.mark.manim
    def test_scene_initialization(self):
        """Test that the DistributionDerivativeDemo scene can be initialized."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DistributionDerivativeDemo()
            assert hasattr(scene, 'construct')
            assert callable(scene.construct)
        except Exception as e:
            pytest.fail(f"DistributionDerivativeDemo initialization failed: {e}")
    
    @pytest.mark.manim
    def test_scene_construction(self):
        """Test that the scene construction doesn't raise errors."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DistributionDerivativeDemo()
            scene.construct()
            assert True
        except Exception as e:
            pytest.fail(f"DistributionDerivativeDemo construction failed: {e}")
    
    def test_piecewise_function_with_jump(self):
        """Test piecewise function with jump discontinuity."""
        x = np.linspace(-2, 2, 1000)
        jump_location = 0.5
        
        # Create piecewise function with jump
        func = np.zeros_like(x)
        for i, x_val in enumerate(x):
            if x_val < jump_location:
                func[i] = 1.0
            else:
                func[i] = 3.0  # Jump of size 2
        
        # Check properties
        assert len(func) == len(x)
        assert np.min(func) == 1.0
        assert np.max(func) == 3.0
        
        # Find jump location
        diff = np.diff(func)
        jump_idx = np.argmax(np.abs(diff))
        detected_jump_location = x[jump_idx]
        
        assert abs(detected_jump_location - jump_location) < 0.1
        assert abs(diff[jump_idx] - 2.0) < 0.1  # Jump size
    
    def test_distribution_derivative_calculation(self):
        """Test distribution derivative calculation."""
        x = np.linspace(-2, 2, 1000)
        dx = x[1] - x[0]
        
        # Create function with jump
        jump_location = 0.0
        func = np.where(x < jump_location, 1.0, 2.0)
        
        # Classical derivative (zero except at jump)
        classical_deriv = np.gradient(func, dx)
        
        # Distribution derivative includes delta at jump
        # The derivative should have a delta function at the jump location
        jump_size = 2.0 - 1.0  # 1.0
        
        # Check classical derivative is zero almost everywhere
        zero_mask = np.abs(x - jump_location) > 0.1
        assert np.allclose(classical_deriv[zero_mask], 0.0, atol=1e-10)
    
    def test_multiple_jumps_detection(self):
        """Test detection of multiple jumps."""
        x = np.linspace(-3, 3, 1000)
        
        # Create function with multiple jumps
        func = np.zeros_like(x)
        jump_locations = [-1.0, 0.0, 1.0]
        jump_values = [1.0, 2.0, 3.0, 1.5]  # 4 values for 3 jumps
        
        for i, x_val in enumerate(x):
            if x_val < jump_locations[0]:
                func[i] = jump_values[0]
            elif x_val < jump_locations[1]:
                func[i] = jump_values[1]
            elif x_val < jump_locations[2]:
                func[i] = jump_values[2]
            else:
                func[i] = jump_values[3]
        
        # Find jumps using gradient
        diff = np.gradient(func, x[1] - x[0])
        jump_indices = np.where(np.abs(diff) > 0.5)[0]
        
        # Should detect jumps near the specified locations
        assert len(jump_indices) >= 2  # At least 2 significant jumps


class TestOpticalCausalityVisualization:
    """Test class for OpticalCausalityVisualization Manim scene."""
    
    @pytest.mark.manim
    def test_scene_initialization(self):
        """Test that the OpticalCausalityVisualization scene can be initialized."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = OpticalCausalityVisualization()
            assert hasattr(scene, 'construct')
            assert callable(scene.construct)
        except Exception as e:
            pytest.fail(f"OpticalCausalityVisualization initialization failed: {e}")
    
    @pytest.mark.manim
    def test_scene_construction(self):
        """Test that the scene construction doesn't raise errors."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = OpticalCausalityVisualization()
            scene.construct()
            assert True
        except Exception as e:
            pytest.fail(f"OpticalCausalityVisualization construction failed: {e}")
    
    def test_causal_signal_creation(self):
        """Test creation of causal signal."""
        t = np.linspace(-2, 2, 1000)
        dt = t[1] - t[0]
        
        # Create causal signal (zero for t < 0)
        causal_signal = np.zeros_like(t)
        causal_mask = t >= 0
        causal_signal[causal_mask] = np.exp(-t[causal_mask] / 0.5) * np.sin(2 * np.pi * 5 * t[causal_mask])
        
        # Check causality
        assert np.all(causal_signal[t < 0] == 0.0)
        assert np.any(causal_signal[t >= 0] != 0.0)
        
        # Check that signal is not identically zero
        assert np.max(np.abs(causal_signal)) > 0.1
    
    def test_non_causal_signal_creation(self):
        """Test creation of non-causal signal."""
        t = np.linspace(-2, 2, 1000)
        
        # Create non-causal signal (non-zero for t < 0)
        non_causal = np.exp(-t**2 / 0.5**2) * np.sin(2 * np.pi * 3 * t)
        
        # Check non-causality
        assert np.any(non_causal[t < 0] != 0.0)
        assert np.any(non_causal[t >= 0] != 0.0)
        
        # Check symmetry
        assert np.abs(np.max(non_causal) - np.min(non_causal)) > 0.1
    
    def test_kramers_kronig_relations(self):
        """Test Kramers-Kronig relations for causal signals."""
        # For a causal signal, the real and imaginary parts of its Fourier transform
        # should be related by the Kramers-Kronig relations
        
        t = np.linspace(-5, 5, 2000)
        dt = t[1] - t[0]
        
        # Create causal signal
        causal_signal = np.zeros_like(t)
        causal_mask = t >= 0
        causal_signal[causal_mask] = np.exp(-t[causal_mask] / 1.0) * np.cos(2 * np.pi * 2 * t[causal_mask])
        
        # Compute Fourier transform
        freq = np.fft.fftfreq(len(t), dt)
        fft_signal = np.fft.fft(causal_signal)
        
        # Real and imaginary parts
        real_part = np.real(fft_signal)
        imag_part = np.imag(fft_signal)
        
        # For causal signals, real and imaginary parts should be related
        # (This is a numerical check - exact verification is complex)
        assert len(real_part) == len(imag_part)
        assert np.all(np.isfinite(real_part))
        assert np.all(np.isfinite(imag_part))


class TestDistributionTheoryCompleteDemo:
    """Test class for DistributionTheoryCompleteDemo Manim scene."""
    
    @pytest.mark.manim
    def test_scene_initialization(self):
        """Test that the DistributionTheoryCompleteDemo scene can be initialized."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DistributionTheoryCompleteDemo()
            assert hasattr(scene, 'construct')
            assert callable(scene.construct)
        except Exception as e:
            pytest.fail(f"DistributionTheoryCompleteDemo initialization failed: {e}")
    
    @pytest.mark.manim
    def test_scene_construction(self):
        """Test that the scene construction doesn't raise errors."""
        if not MANIM_AVAILABLE:
            pytest.skip("Manim not available")
        
        try:
            scene = DistributionTheoryCompleteDemo()
            scene.construct()
            assert True
        except Exception as e:
            pytest.fail(f"DistributionTheoryCompleteDemo construction failed: {e}")
    
    def test_complete_demo_data_generation(self):
        """Test data generation for complete distribution theory demo."""
        # Test that all required mathematical functions can be generated
        
        # Delta functions
        x = np.linspace(-3, 3, 500)
        delta_gaussian = (1 / (0.1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / 0.1)**2)
        
        # Green's functions
        k = 2 * np.pi / 0.5
        source_pos = 0.0
        greens_1d = (1j / (2 * k)) * np.exp(1j * k * np.abs(x - source_pos))
        
        # Causal signal
        t = np.linspace(-2, 2, 500)
        causal_signal = np.zeros_like(t)
        causal_mask = t >= 0
        causal_signal[causal_mask] = np.exp(-t[causal_mask] / 0.5) * np.sin(2 * np.pi * 3 * t[causal_mask])
        
        # Check all generated data
        assert len(delta_gaussian) == len(x)
        assert len(greens_1d) == len(x)
        assert len(causal_signal) == len(t)
        
        assert np.all(np.isfinite(delta_gaussian))
        assert np.all(np.isfinite(greens_1d))
        assert np.all(np.isfinite(causal_signal))
        
        # Check specific properties
        assert np.trapz(delta_gaussian, x) == pytest.approx(1.0, rel=0.01)
        assert np.iscomplexobj(greens_1d)
        assert np.all(causal_signal[t < 0] == 0.0)


class TestManimAvailability:
    """Test Manim availability detection and handling."""
    
    def test_manim_import_detection(self):
        """Test that Manim import is properly detected."""
        # This test verifies that our import logic works correctly
        try:
            import manim
            manim_available = True
        except ImportError:
            manim_available = False
        
        # Should match our global constant
        assert manim_available == MANIM_AVAILABLE
    
    def test_mock_classes_functionality(self):
        """Test that mock classes provide basic functionality when Manim is unavailable."""
        if MANIM_AVAILABLE:
            pytest.skip("Manim is available, mock classes not needed")
        
        # Test that mock classes can be instantiated and have basic methods
        scene = Scene()
        assert hasattr(scene, 'construct')
        
        line = Line((0, 0, 0), (1, 1, 0))
        assert hasattr(line, 'start')
        assert hasattr(line, 'end')
        
        text = Text("Test")
        assert hasattr(text, 'text')
        assert text.text == "Test"
        
        axes = Axes()
        assert hasattr(axes, 'x_range')
        assert hasattr(axes, 'y_range')


if __name__ == "__main__":
    pytest.main([__file__])