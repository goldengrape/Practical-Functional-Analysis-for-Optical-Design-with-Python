"""
Shared test fixtures and configuration for the test suite.

This file contains common fixtures, utilities, and configuration that can be
used across all test modules in the project.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional, Tuple
import sys
import os

# Add the python-project directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure matplotlib for testing (non-interactive backend)
matplotlib.use('Agg')


@pytest.fixture
def sample_optical_wavelength() -> float:
    """Provide a sample optical wavelength (in meters)."""
    return 550e-9  # 550 nm (green light)


@pytest.fixture
def sample_refractive_indices() -> dict:
    """Provide sample refractive indices for common optical materials."""
    return {
        'air': 1.0,
        'water': 1.33,
        'glass': 1.5,
        'crown_glass': 1.52,
        'flint_glass': 1.65,
        'silica': 1.46
    }


@pytest.fixture
def tolerance() -> float:
    """Provide a standard numerical tolerance for floating point comparisons."""
    return 1e-10


@pytest.fixture
def sample_grid_size() -> int:
    """Provide a standard grid size for numerical computations."""
    return 64


@pytest.fixture
def sample_domain() -> Tuple[float, float]:
    """Provide a standard domain for function evaluations."""
    return (-1.0, 1.0)


@pytest.fixture
def disable_plotting(monkeypatch):
    """Fixture to disable plotting during tests."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)


@pytest.fixture
def capture_plot() -> Callable:
    """Fixture to capture plot data for testing."""
    plots = []
    
    def capture():
        fig = plt.gcf()
        plots.append(fig)
        return fig
    
    return capture


@pytest.fixture
def sample_2d_function() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Provide a sample 2D function for testing."""
    def func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.sqrt(x**2 + y**2))
    return func


@pytest.fixture
def sample_1d_function() -> Callable[[np.ndarray], np.ndarray]:
    """Provide a sample 1D function for testing."""
    def func(x: np.ndarray) -> np.ndarray:
        return np.exp(-x**2) * np.sin(2*np.pi*x)
    return func


@pytest.fixture
def sample_polynomial_coefficients() -> np.ndarray:
    """Provide sample polynomial coefficients for testing."""
    return np.array([1.0, -0.5, 0.1, -0.01])  # cubic polynomial


@pytest.fixture
def sample_zernike_modes() -> dict:
    """Provide sample Zernike polynomial modes for testing."""
    return {
        'piston': (0, 0),
        'tilt_x': (1, 1),
        'tilt_y': (1, -1),
        'defocus': (2, 0),
        'astigmatism_45': (2, 2),
        'astigmatism_90': (2, -2),
        'coma_x': (3, 1),
        'coma_y': (3, -1),
        'spherical': (4, 0)
    }


@pytest.fixture
def sample_optical_system_params() -> dict:
    """Provide sample optical system parameters."""
    return {
        'focal_length': 0.1,  # 10 cm
        'aperture_diameter': 0.025,  # 25 mm
        'wavelength': 550e-9,  # 550 nm
        'numerical_aperture': 0.12
    }


@pytest.fixture
def mock_matplotlib_show(monkeypatch):
    """Mock matplotlib show function to prevent plots from appearing."""
    monkeypatch.setattr(plt, 'show', lambda: None)


@pytest.fixture
def sample_noisy_data() -> Tuple[np.ndarray, np.ndarray]:
    """Provide sample noisy data for testing signal processing functions."""
    x = np.linspace(0, 1, 100)
    clean_signal = np.sin(2 * np.pi * 5 * x)
    noise = 0.1 * np.random.randn(len(x))
    noisy_signal = clean_signal + noise
    return x, noisy_signal


@pytest.fixture
def sample_optical_path() -> np.ndarray:
    """Provide a sample optical path for testing ray tracing functions."""
    # Simple straight line path
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])


@pytest.fixture
def sample_lens_params() -> dict:
    """Provide sample lens parameters for testing."""
    return {
        'radius_of_curvature': 0.05,  # 50 mm
        'refractive_index': 1.5,
        'thickness': 0.002,  # 2 mm
        'diameter': 0.025  # 25 mm
    }


# Custom markers for different test categories
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests that create visualizations"
    )
    config.addinivalue_line(
        "markers", "optiland: marks tests that require optiland library"
    )
    config.addinivalue_line(
        "markers", "manim: marks tests that require manim (often skipped in CI due to rendering requirements)"
    )


# Helper functions for tests
def assert_array_equal_with_tolerance(actual: np.ndarray, expected: np.ndarray, 
                                    tolerance: float = 1e-10) -> None:
    """Assert that two arrays are equal within a given tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def assert_function_properties(func: Callable, domain: Tuple[float, float], 
                           expected_properties: dict) -> None:
    """Assert that a function has expected properties."""
    x = np.linspace(domain[0], domain[1], 100)
    y = func(x)
    
    if 'continuity' in expected_properties:
        # Simple continuity check - no large jumps
        dy = np.diff(y)
        max_jump = np.max(np.abs(dy))
        assert max_jump < expected_properties['continuity']['max_jump']
    
    if 'differentiability' in expected_properties:
        # Check if derivative exists and is continuous
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        max_second_deriv = np.max(np.abs(d2y))
        assert max_second_deriv < expected_properties['differentiability']['max_second_deriv']


def create_test_figure() -> plt.Figure:
    """Create a simple test figure."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x)
    ax.plot(x, y)
    return fig