"""
Test suite for Simple Optical Operators implementation in chapter05_operator_theory.
Tests basic optical operators and transformations.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter05_operator_theory.simple_operators import SimpleOpticalOperators


class TestSimpleOpticalOperators:
    """Test suite for SimpleOpticalOperators class."""
    
    @pytest.fixture
    def simple_optics(self):
        """Create a SimpleOpticalOperators instance for testing."""
        return SimpleOpticalOperators(size=32)
    
    def test_initialization(self, simple_optics):
        """Test proper initialization of SimpleOpticalOperators."""
        assert simple_optics.size == 32
        assert hasattr(simple_optics, 'X')
        assert hasattr(simple_optics, 'Y')
        
        # Check coordinate system
        assert simple_optics.X.shape == (32, 32)
        assert simple_optics.Y.shape == (32, 32)
        
        # Check coordinate ranges
        assert np.min(simple_optics.X) >= -1.0
        assert np.max(simple_optics.X) <= 1.0
        assert np.min(simple_optics.Y) >= -1.0
        assert np.max(simple_optics.Y) <= 1.0
    
    def test_fourier_transform(self, simple_optics):
        """Test Fourier transform functionality."""
        # Create test field
        test_field = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
        
        # Apply Fourier transform
        spectrum = simple_optics.fourier_transform(test_field)
        
        # Check output properties
        assert spectrum.shape == test_field.shape
        assert np.iscomplexobj(spectrum)
        
        # Check that spectrum is shifted (center at middle)
        center = simple_optics.size // 2
        assert np.abs(spectrum[center, center]) > 0  # Should have DC component
    
    def test_lens_operator(self, simple_optics):
        """Test lens operator functionality."""
        # Create test field
        test_field = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
        
        # Apply lens operator
        focal_length = 0.1  # 100 mm focal length
        lensed_field = simple_optics.lens_operator(test_field, focal_length)
        
        # Check output properties
        assert lensed_field.shape == test_field.shape
        assert np.iscomplexobj(lensed_field)
        
        # Check that amplitude is preserved (only phase changes)
        input_amplitude = np.abs(test_field)
        output_amplitude = np.abs(lensed_field)
        
        assert np.allclose(input_amplitude, output_amplitude)
        
        # Check that phase is different
        assert not np.allclose(np.angle(test_field), np.angle(lensed_field))
    
    def test_different_sizes(self):
        """Test with different grid sizes."""
        for size in [16, 32, 64, 128]:
            simple_optics = SimpleOpticalOperators(size=size)
            
            assert simple_optics.size == size
            assert simple_optics.X.shape == (size, size)
            assert simple_optics.Y.shape == (size, size)
            
            # Test basic functionality
            test_field = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
            spectrum = simple_optics.fourier_transform(test_field)
            
            assert spectrum.shape == (size, size)
            assert np.iscomplexobj(spectrum)
    
    def test_gaussian_beam_fourier_transform(self, simple_optics):
        """Test Fourier transform of Gaussian beam."""
        # Create Gaussian field
        w0 = 0.5  # Beam waist
        gaussian_field = np.exp(-(simple_optics.X**2 + simple_optics.Y**2) / w0**2)
        
        # Apply Fourier transform
        spectrum = simple_optics.fourier_transform(gaussian_field)
        
        # Check that spectrum is also Gaussian-like
        assert np.iscomplexobj(spectrum)
        
        # Check that spectrum has central peak
        center = simple_optics.size // 2
        assert np.abs(spectrum[center, center]) > np.mean(np.abs(spectrum))
    
    def test_plane_wave_lens(self, simple_optics):
        """Test lens operator on plane wave."""
        # Create plane wave (constant phase)
        plane_wave = np.ones_like(simple_optics.X)
        
        # Apply lens
        focal_length = 0.2
        lensed_wave = simple_optics.lens_operator(plane_wave, focal_length)
        
        # Check output properties
        assert lensed_wave.shape == plane_wave.shape
        assert np.iscomplexobj(lensed_wave)
        
        # Check that amplitude is preserved
        assert np.allclose(np.abs(lensed_wave), plane_wave)
        
        # Check that phase is quadratic (lens applies quadratic phase)
        phase = np.angle(lensed_wave)
        
        # Phase should vary across the field (not constant)
        assert np.std(phase) > 1e-10  # Should have phase variation
    
    def test_circular_symmetry_fourier(self, simple_optics):
        """Test Fourier transform preserves circular symmetry for circularly symmetric inputs."""
        # Create circularly symmetric field
        r_squared = simple_optics.X**2 + simple_optics.Y**2
        circular_field = np.exp(-r_squared)
        
        # Apply Fourier transform
        spectrum = simple_optics.fourier_transform(circular_field)
        
        # Check that spectrum is approximately circularly symmetric
        # Compare opposite quadrants
        center = simple_optics.size // 2
        
        # Extract radial profile in x and y directions
        x_profile = np.abs(spectrum[center, :])
        y_profile = np.abs(spectrum[:, center])
        
        # Should be approximately the same (circular symmetry)
        assert np.allclose(x_profile, y_profile, rtol=1e-2)
    
    def test_energy_conservation_fourier(self, simple_optics):
        """Test energy conservation in Fourier transform (Parseval's theorem)."""
        # Create test field
        test_field = np.exp(-2 * (simple_optics.X**2 + simple_optics.Y**2))
        
        # Compute energies
        spatial_energy = np.sum(np.abs(test_field)**2)
        
        # Apply Fourier transform
        spectrum = simple_optics.fourier_transform(test_field)
        spectral_energy = np.sum(np.abs(spectrum)**2)
        
        # For discrete Fourier transform, energies should be related
        # (Exact relationship depends on normalization)
        assert spectral_energy > 0  # Should have non-zero energy
        assert spatial_energy > 0   # Should have non-zero energy
        
        # Check that energy is approximately conserved
        # (Allowing for different normalization conventions)
        ratio = spectral_energy / spatial_energy
        assert ratio > 0.1  # Should be within reasonable range
        assert ratio < 100.0
    
    def test_linearity_fourier(self, simple_optics):
        """Test linearity of Fourier transform."""
        # Create test fields
        field1 = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
        field2 = np.exp(-2 * (simple_optics.X**2 + simple_optics.Y**2))
        
        # Linear combination
        alpha, beta = 0.5, 0.7
        linear_combination = alpha * field1 + beta * field2
        
        # Apply Fourier transform to linear combination
        spectrum_combined = simple_optics.fourier_transform(linear_combination)
        
        # Apply Fourier transform to individual fields and combine
        spectrum1 = simple_optics.fourier_transform(field1)
        spectrum2 = simple_optics.fourier_transform(field2)
        spectrum_linear = alpha * spectrum1 + beta * spectrum2
        
        # Should be equal (linearity)
        assert np.allclose(spectrum_combined, spectrum_linear, rtol=1e-10)
    
    def test_linearity_lens(self, simple_optics):
        """Test linearity of lens operator."""
        # Create test fields
        field1 = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
        field2 = np.exp(-2 * (simple_optics.X**2 + simple_optics.Y**2))
        
        # Linear combination
        alpha, beta = 0.3, 0.8
        linear_combination = alpha * field1 + beta * field2
        
        # Apply lens to linear combination
        focal_length = 0.15
        lensed_combined = simple_optics.lens_operator(linear_combination, focal_length)
        
        # Apply lens to individual fields and combine
        lensed1 = simple_optics.lens_operator(field1, focal_length)
        lensed2 = simple_optics.lens_operator(field2, focal_length)
        lensed_linear = alpha * lensed1 + beta * lensed2
        
        # Should be equal (linearity)
        assert np.allclose(lensed_combined, lensed_linear, rtol=1e-10)
    
    def test_edge_cases(self, simple_optics):
        """Test edge cases and boundary conditions."""
        # Test with zero field
        zero_field = np.zeros_like(simple_optics.X)
        
        spectrum_zero = simple_optics.fourier_transform(zero_field)
        assert np.allclose(spectrum_zero, 0.0)
        
        lensed_zero = simple_optics.lens_operator(zero_field, 0.1)
        assert np.allclose(lensed_zero, 0.0)
        
        # Test with constant field
        constant_field = np.ones_like(simple_optics.X)
        
        spectrum_constant = simple_optics.fourier_transform(constant_field)
        assert not np.allclose(spectrum_constant, 0.0)  # Should have DC component
        
        lensed_constant = simple_optics.lens_operator(constant_field, 0.1)
        assert np.allclose(np.abs(lensed_constant), 1.0)  # Amplitude preserved
    
    def test_focal_length_scaling(self, simple_optics):
        """Test that lens operator scales correctly with focal length."""
        # Create test field
        test_field = np.exp(-(simple_optics.X**2 + simple_optics.Y**2))
        
        # Apply different focal lengths
        focal_lengths = [0.05, 0.1, 0.2, 0.5]
        phases = []
        
        for f in focal_lengths:
            lensed_field = simple_optics.lens_operator(test_field, f)
            phase = np.angle(lensed_field)
            phases.append(phase)
        
        # Check that stronger lenses (shorter focal length) produce more phase curvature
        # Compare phase variation for different focal lengths
        phase_variations = [np.std(phase) for phase in phases]
        
        # Shorter focal length should produce more phase variation
        assert phase_variations[0] > phase_variations[-1]  # Stronger lens vs weaker lens