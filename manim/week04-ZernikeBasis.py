# ZernikeBasis.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render ZernikeBasis.py ZernikeBasisAnimation

from manim import *
import numpy as np
from scipy.special import factorial
import math

class ZernikeBasisAnimation(ThreeDScene):
    """
    Animation demonstrating Zernike polynomial basis functions used in optical
    wavefront analysis. Shows how these orthogonal polynomials form a complete
    basis for representing wavefront aberrations.
    """
    
    def construct(self):
        # Set up 3D scene
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES, zoom=0.8)
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("Zernike多项式基函数", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=(-1, 1, 0.5),
            y_range=(-1, 1, 0.5),
            z_range=(-1, 1, 0.5),
            x_length=6,
            y_length=6,
            z_length=4
        )
        
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="Z")
        
        self.play(Create(axes), Write(axis_labels))
        
        # Define Zernike polynomial functions
        def zernike_radial(n, m, r):
            """Radial part of Zernike polynomial"""
            if (n - m) % 2 != 0:
                return np.zeros_like(r)
            
            radial = np.zeros_like(r)
            for k in range((n - m) // 2 + 1):
                coeff = ((-1)**k * factorial(n - k)) / (
                    factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
                )
                radial += coeff * r**(n - 2*k)
            
            return radial
        
        def zernike_polynomial(n, m, x, y):
            """Complete Zernike polynomial"""
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Only compute within unit circle
            mask = r <= 1.0
            result = np.zeros_like(x)
            
            if m >= 0:
                result[mask] = zernike_radial(n, m, r[mask]) * np.cos(m * theta[mask])
            else:
                result[mask] = zernike_radial(n, -m, r[mask]) * np.sin(-m * theta[mask])
            
            return result
        
        # Define a sequence of Zernike polynomials to show
        zernike_modes = [
            (0, 0, "活塞 (Piston)"),
            (1, 1, "倾斜 X (Tip)"),
            (1, -1, "倾斜 Y (Tilt)"),
            (2, 0, "离焦 (Defocus)"),
            (2, 2, "像散 0° (Astigmatism)"),
            (2, -2, "像散 45° (Astigmatism)"),
            (3, 1, "彗差 X (Coma)"),
            (3, -1, "彗差 Y (Coma)"),
            (4, 0, "球差 (Spherical)")
        ]
        
        colors = [BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, YELLOW, PINK, MAROON]
        
        # Show each Zernike mode
        for i, (n, m, name) in enumerate(zernike_modes):
            # Create surface for this Zernike polynomial
            surface = Surface(
                lambda u, v: np.array([
                    u, 
                    v, 
                    0.5 * zernike_polynomial(n, m, u, v)
                ]),
                u_range=(-1, 1),
                v_range=(-1, 1),
                resolution=(30, 30),
                fill_opacity=0.7,
                stroke_color=colors[i],
                stroke_width=1
            )
            
            # Color by height
            surface.set_fill_by_value(
                axes=axes,
                colorscale=[(BLUE, -0.5), (GREEN, 0), (RED, 0.5)]
            )
            
            # Create label
            label = Text(name, font_size=20, color=colors[i])
            label.to_edge(RIGHT, buff=0.5)
            label.shift(UP * (2 - i * 0.3))
            
            self.add_fixed_in_frame_mobjects(label)
            
            # Show the surface
            self.play(
                Create(surface),
                Write(label),
                run_time=1.5
            )
            
            # Rotate camera for better view
            if i == 0:
                self.begin_ambient_camera_rotation(rate=0.1)
                self.wait(3)
                self.stop_ambient_camera_rotation()
            else:
                self.wait(2)
            
            # Keep some surfaces for comparison
            if i < len(zernike_modes) - 1 and i % 3 == 2:
                self.wait(1)
            elif i < len(zernike_modes) - 1:
                self.play(FadeOut(surface), FadeOut(label))
        
        # Final rotation
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axis_labels)
        )
        
        self.wait(1)


class ZernikeOrthogonalityDemo(Scene):
    """
    Demonstration of Zernike polynomial orthogonality property.
    Shows that different modes are orthogonal to each other.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("Zernike多项式正交性", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Define orthogonality equation
        orthogonality_eq = MathTex(
            r"\int_0^{2\pi} \int_0^1 Z_n^m(r,\theta) Z_{n'}^{m'}(r,\theta) r dr d\theta = 0",
            font_size=28,
            color=BLACK
        )
        orthogonality_eq.next_to(title, DOWN, buff=0.5)
        
        condition_text = Text("当 (n,m) ≠ (n',m') 时", font_size=20, color=GRAY)
        condition_text.next_to(orthogonality_eq, DOWN, buff=0.2)
        
        self.play(Write(orthogonality_eq))
        self.play(Write(condition_text))
        self.wait(2)
        
        # Create visual representation
        axes1 = Axes(
            x_range=(-1, 1, 0.5),
            y_range=(-1, 1, 0.5),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=3,
            y_length=3
        )
        axes1.to_edge(LEFT, buff=0.5)
        axes1.shift(UP * 0.5)
        
        axes2 = Axes(
            x_range=(-1, 1, 0.5),
            y_range=(-1, 1, 0.5),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=3,
            y_length=3
        )
        axes2.to_edge(RIGHT, buff=0.5)
        axes2.shift(UP * 0.5)
        
        # Example: Defocus and Astigmatism
        def defocus_func(x, y):
            r2 = x**2 + y**2
            return 2 * r2 - 1 if r2 <= 1 else 0
        
        def astigmatism_func(x, y):
            return x**2 - y**2 if x**2 + y**2 <= 1 else 0
        
        # Create contour plots
        x_range = np.linspace(-1, 1, 50)
        y_range = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Defocus contour
        Z1 = defocus_func(X, Y)
        contour1 = axes1.get_contour_plot(
            lambda x, y: defocus_func(x, y),
            x_range=(-1, 1),
            y_range=(-1, 1),
            colors=[BLUE, GREEN, YELLOW, RED],
            stroke_width=2
        )
        
        # Astigmatism contour  
        Z2 = astigmatism_func(X, Y)
        contour2 = axes2.get_contour_plot(
            lambda x, y: astigmatism_func(x, y),
            x_range=(-1, 1),
            y_range=(-1, 1),
            colors=[BLUE, GREEN, YELLOW, RED],
            stroke_width=2
        )
        
        # Labels
        label1 = Text("离焦 (Defocus)", font_size=16, color=BLUE)
        label1.next_to(axes1, UP, buff=0.2)
        
        label2 = Text("像散 (Astigmatism)", font_size=16, color=GREEN)
        label2.next_to(axes2, UP, buff=0.2)
        
        self.play(
            Create(axes1),
            Create(axes2),
            Create(contour1),
            Create(contour2),
            Write(label1),
            Write(label2)
        )
        
        # Show orthogonality result
        result_text = Text("内积 = 0 (正交)", font_size=24, color=RED)
        result_text.to_edge(DOWN, buff=0.5)
        
        result_box = SurroundingRectangle(result_text, buff=0.2, color=RED)
        
        self.play(
            Write(result_text),
            Create(result_box)
        )
        
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(orthogonality_eq),
            FadeOut(condition_text),
            FadeOut(axes1),
            FadeOut(axes2),
            FadeOut(contour1),
            FadeOut(contour2),
            FadeOut(label1),
            FadeOut(label2),
            FadeOut(result_text),
            FadeOut(result_box)
        )
        
        self.wait(1)


class ZernikeWavefrontReconstruction(Scene):
    """
    Shows how Zernike polynomials can be combined to reconstruct
    a complex wavefront aberration.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("波前重构", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Create coordinate system
        axes = Axes(
            x_range=(-1, 1, 0.5),
            y_range=(-1, 1, 0.5),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=4,
            y_length=4
        )
        axes.to_edge(LEFT, buff=0.5)
        
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        self.play(Create(axes), Write(axis_labels))
        
        # Define Zernike functions (simplified)
        def zernike_piston(x, y):
            return 1.0 if x**2 + y**2 <= 1 else 0
        
        def zernike_tilt_x(x, y):
            return x if x**2 + y**2 <= 1 else 0
        
        def zernike_defocus(x, y):
            return 2*(x**2 + y**2) - 1 if x**2 + y**2 <= 1 else 0
        
        def zernike_astigmatism(x, y):
            return x**2 - y**2 if x**2 + y**2 <= 1 else 0
        
        # Coefficients for reconstruction
        coefficients = [1.0, 0.3, 0.5, 0.2]
        functions = [zernike_piston, zernike_tilt_x, zernike_defocus, zernike_astigmatism]
        names = ["活塞", "倾斜X", "离焦", "像散"]
        colors = [BLUE, GREEN, ORANGE, RED]
        
        # Show individual components
        component_plots = VGroup()
        component_labels = VGroup()
        
        for i, (func, name, coeff, color) in enumerate(zip(functions, names, coefficients, colors)):
            # Create contour plot
            contour = axes.get_contour_plot(
                func,
                x_range=(-1, 1),
                y_range=(-1, 1),
                colors=[color],
                stroke_width=2
            )
            
            # Create label
            label = MathTex(f"{coeff:.1f} \\times \\text{{{name}}}", font_size=16, color=color)
            label.to_edge(RIGHT, buff=0.5)
            label.shift(UP * (1.5 - i * 0.8))
            
            component_plots.add(contour)
            component_labels.add(label)
            
            self.play(
                Create(contour),
                Write(label),
                run_time=1
            )
        
        self.wait(2)
        
        # Show reconstruction
        def reconstructed_wavefront(x, y):
            result = 0
            for coeff, func in zip(coefficients, functions):
                result += coeff * func(x, y)
            return result
        
        # Create reconstruction axes
        recon_axes = Axes(
            x_range=(-1, 1, 0.5),
            y_range=(-1, 1, 0.5),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=4,
            y_length=4
        )
        recon_axes.to_edge(RIGHT, buff=0.5)
        
        recon_axis_labels = recon_axes.get_axis_labels(x_label="x", y_label="y")
        
        # Create reconstruction plot
        recon_contour = recon_axes.get_contour_plot(
            reconstructed_wavefront,
            x_range=(-1, 1),
            y_range=(-1, 1),
            colors=[PURPLE, BLUE, GREEN, YELLOW, RED],
            stroke_width=3
        )
        
        recon_label = Text("重构波前", font_size=20, color=PURPLE)
        recon_label.next_to(recon_axes, UP, buff=0.2)
        
        # Show reconstruction
        self.play(
            Create(recon_axes),
            Write(recon_axis_labels),
            Create(recon_contour),
            Write(recon_label)
        )
        
        # Show equation
        equation = MathTex(
            r"W(x,y) = 1.0P + 0.3T_x + 0.5D + 0.2A",
            font_size=20,
            color=BLACK
        )
        equation.to_edge(DOWN, buff=0.5)
        
        self.play(Write(equation))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(component_plots),
            FadeOut(component_labels),
            FadeOut(recon_axes),
            FadeOut(recon_axis_labels),
            FadeOut(recon_contour),
            FadeOut(recon_label),
            FadeOut(equation)
        )
        
        self.wait(1)