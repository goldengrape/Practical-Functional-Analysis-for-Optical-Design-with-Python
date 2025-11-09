# L2NormVisualization.py
#
# To run this animation, you need Manim, scipy, and icontract:
# pip install manim scipy icontract
#
# Then, execute the command in your terminal:
# manim -pqh render L2NormVisualization.py L2NormVisualization

from manim import *
import numpy as np
from scipy.integrate import dblquad
import icontract
from typing import Callable, Tuple

# --- Axiomatic Design: Independent Functional Requirements (Math Models) ---

def wavefront_defocus(u: float, v: float, amplitude: float) -> float:
    """A pure mathematical model for defocus aberration."""
    return amplitude * (u**2 + v**2)

@icontract.require(lambda amplitude: amplitude >= 0)
def calculate_volume_l2_norm(amplitude: float, u_range: Tuple[float, float]) -> float:
    """
    Calculates the L2 norm squared (volume) for the squared defocus function
    using numerical double integration.
    """
    # The function to integrate is the square of the wavefront function.
    integrand = lambda u, v: (wavefront_defocus(u, v, amplitude))**2
    
    # Integrate over a circular domain, which is more typical for pupils.
    # We integrate over u from -r to r, and v over the circle's bounds.
    r = u_range[1]
    volume, _ = dblquad(integrand, -r, r, lambda u: -np.sqrt(r**2 - u**2), lambda u: np.sqrt(r**2 - u**2))
    return volume


class L2NormVisualization(ThreeDScene):
    """
    A Manim scene that visualizes the L2 norm of a function as the volume
    under the surface of its square, connecting it to the RMS wavefront error concept.
    """
    def construct(self):
        # --- 1. SETUP: Scene and Data ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=-120 * DEGREES, zoom=0.9)
        
        # Data-Oriented: The entire animation is driven by this single ValueTracker.
        aberration_amplitude = ValueTracker(0.5)
        
        # Define the domain for our functions (the "pupil")
        u_min, u_max = -2, 2
        v_min, v_max = -2, 2

        axes = ThreeDAxes(
            x_range=(u_min, u_max, 1),
            y_range=(v_min, v_max, 1),
            z_range=(0, 8, 2),
            x_length=8,
            y_length=8,
            z_length=6
        )
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="W")

        # --- 2. MOBJECTS DEFINITION: Derived from the data state ---
        
        # Surface for the wavefront W(x, y)
        surface_w = always_redraw(lambda: Surface(
            lambda u, v: axes.c2p(u, v, wavefront_defocus(u, v, aberration_amplitude.get_value())),
            u_range=(u_min, u_max),
            v_range=(v_min, v_max),
            resolution=(24, 24)
        ).set_fill_by_value(axes=axes, colorscale=[(BLUE, -2), (GREEN, 0), (YELLOW, 2)]))

        # Surface for the squared wavefront W(x, y)^2
        surface_w_sq = always_redraw(lambda: Surface(
            lambda u, v: axes.c2p(u, v, wavefront_defocus(u, v, aberration_amplitude.get_value())**2),
            u_range=(u_min, u_max),
            v_range=(v_min, v_max),
            resolution=(24, 24)
        ).set_fill_by_value(axes=axes, colorscale=[(GREEN_C, 0), (ORANGE, 5), (RED, 10)]))

        # Volume visualization using Riemann prisms
        volume_prisms = always_redraw(lambda: axes.get_riemann_rectangles(
            graph=surface_w_sq,
            dx=0.4,
            dy=0.4,
            input_sample_type="center",
            stroke_width=0.5,
            fill_opacity=0.6
        ))

        # --- 3. LABELS: Also data-driven ---

        # Labels will be fixed on screen
        label_w = MathTex("W(x, y) = A(x^2 + y^2)", font_size=36).to_corner(UL)
        label_w_sq = MathTex("W(x, y)^2", font_size=36).to_corner(UL)
        label_l2_norm = MathTex(r"||W||_2^2 = \iint_{\text{pupil}} |W|^2 \,dx\,dy \approx \text{Volume}", font_size=36).to_corner(UL)
        volume_value_text = MathTex("= 0.00", font_size=42).next_to(label_l2_norm, DOWN, align=LEFT)
        
        self.add_fixed_in_frame_mobjects(label_w, label_w_sq, label_l2_norm, volume_value_text)
        # Hide them initially
        label_w_sq.set_opacity(0)
        label_l2_norm.set_opacity(0)
        volume_value_text.set_opacity(0)

        # Updater for the volume value text
        def volume_updater(mob):
            amp = aberration_amplitude.get_value()
            vol = calculate_volume_l2_norm(amp, (u_min, u_max))
            new_text = MathTex(f"= {vol:.2f}", font_size=42).move_to(mob)
            mob.become(new_text)
        volume_value_text.add_updater(volume_updater)

        # --- 4. ANIMATION SEQUENCE ---

        # a. Introduce the wavefront W
        self.play(Create(axes), Write(axis_labels))
        self.play(FadeIn(surface_w), Write(label_w), run_time=1.5)
        self.wait(1)

        # b. Transform W to W^2
        self.play(
            ReplacementTransform(surface_w, surface_w_sq),
            FadeTransform(label_w, label_w_sq),
            run_time=2
        )
        self.wait(1.5)
        
        # c. Build the volume and show the L2 norm formula
        self.play(
            FadeOut(label_w_sq),
            FadeIn(label_l2_norm, volume_value_text),
            Create(volume_prisms),
            run_time=2
        )
        self.wait(1.5)

        # d. Animate the aberration amplitude and see the volume change
        self.play(
            aberration_amplitude.animate.set_value(1.0),
            run_time=4,
            rate_func=rate_functions.there_and_back
        )
        self.wait(1)
        
        # e. Final showcase with camera rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(aberration_amplitude.animate.set_value(0.7), run_time=2)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait(2)