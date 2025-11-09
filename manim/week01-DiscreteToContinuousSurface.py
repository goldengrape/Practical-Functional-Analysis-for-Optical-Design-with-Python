# DiscreteToContinuousSurface.py
#
# To run this animation, you need Manim, scipy, and icontract:
# pip install manim scipy icontract
#
# Then, run the command in your terminal:
# manim -pql render DiscreteToContinuousSurface.py DiscreteToContinuousSurface

from manim import *
import numpy as np
from scipy.interpolate import Rbf
import icontract
from typing import Callable, Tuple

# --- Axiomatic Design: Independent Functional Requirements ---

# FR1: Generate simulated measurement data.
# This function is pure and has no dependency on Manim.
def generate_scatter_data(num_points: int, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
    """
    Generates a 3D point cloud based on an underlying sinc function.
    This simulates discrete data from a measurement device like an interferometer.
    
    Returns:
        np.ndarray: An array of shape (num_points, 3) representing [x, y, z] coordinates.
    """
    # Generate random (x, y) points within the specified ranges
    xs = np.random.uniform(*x_range, num_points)
    ys = np.random.uniform(*y_range, num_points)
    
    # The "true" underlying continuous function we are sampling from
    rs = np.sqrt(xs**2 + ys**2)
    zs = 2 * np.sinc(rs) # sinc(x) = sin(pi*x)/(pi*x)
    
    # Add some measurement noise
    zs += np.random.normal(0, 0.1, num_points)
    
    return np.stack([xs, ys, zs], axis=-1)

# FR2: Build a continuous model from discrete data.
# This function is also pure and independent of Manim.
@icontract.require(lambda data: isinstance(data, np.ndarray))
@icontract.require(lambda data: data.ndim == 2 and data.shape[1] == 3)
@icontract.require(lambda data: data.shape[0] >= 4, "Need at least 4 points for Rbf interpolation")
def create_interpolating_function(data: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Creates a continuous, callable function that interpolates the given scatter data.
    
    Args:
        data: A numpy array of shape (N, 3) for N points.
        
    Returns:
        A callable function `f(x, y)` that returns the interpolated z value.
    """
    x, y, z = data.T
    # Use Radial Basis Function interpolation for smooth fitting of scattered data.
    # 'cubic' spline provides C^2 continuity, essential for optical surfaces.
    rbf_interpolator = Rbf(x, y, z, function='cubic', smooth=0.1)
    return rbf_interpolator


class DiscreteToContinuousSurface(ThreeDScene):
    """
    A Manim scene that visualizes the conceptual leap from a discrete set of
    data points to a continuous surface model via interpolation.
    """
    def construct(self):
        # --- 1. SETUP: Data-Oriented flow starts here ---
        # Generate the raw data and the derived continuous model first.
        x_range = (-4, 4)
        y_range = (-4, 4)
        
        scatter_data = generate_scatter_data(num_points=100, x_range=x_range, y_range=y_range)
        surface_func = create_interpolating_function(scatter_data)

        # --- 2. SCENE PREPARATION: Set up the 3D environment ---
        axes = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            z_range=(-2, 3),
            x_length=8,
            y_length=8,
            z_length=5,
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES, zoom=0.8)
        
        # --- 3. CREATE MOBJECTS: Derive visual objects from data and model ---
        
        # A VGroup of Dot3D objects derived from the raw data
        dots_group = VGroup(*[Dot3D(point=axes.c2p(d), radius=0.06, color=BLUE) for d in scatter_data])
        
        # A Surface object derived from the continuous model function
        surface = Surface(
            lambda u, v: axes.c2p(u, v, surface_func(u, v)),
            u_range=x_range,
            v_range=y_range,
            resolution=(42, 42),
            fill_opacity=0.7,
            checkerboard_colors=[TEAL, GREEN_C],
        )

        # Labels for explaining the concept
        label_discrete = Tex("Discrete Measurement Data", r"($W_i$ at $(x_i, y_i)$)").to_corner(UL)
        label_continuous = Tex("Continuous Surface Model", r"($W(x, y)$)").to_corner(UL)

        # --- 4. ANIMATION SEQUENCE: Tell the story ---
        
        # a. Introduce the context (the coordinate system)
        self.add_fixed_in_frame_mobjects(label_discrete)
        self.play(Create(axes))
        
        # b. Show the discrete data points appearing
        self.play(
            Write(label_discrete),
            LaggedStart(*[FadeIn(dot, scale=0.5) for dot in dots_group]),
            run_time=2
        )
        self.wait(1.5)
        
        # c. The core conceptual leap: Transform the discrete points into the continuous surface
        self.play(
            # Occam's Razor: Transform is the simplest, most direct way to show this.
            Transform(dots_group, surface),
            FadeTransform(label_discrete, label_continuous),
            run_time=2.5,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(1)

        # d. Show off the result
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        self.wait(2)