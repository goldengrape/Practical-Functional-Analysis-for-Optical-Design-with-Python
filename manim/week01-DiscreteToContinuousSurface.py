# DiscreteToContinuousSurface.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render week01-DiscreteToContinuousSurface.py DiscreteToContinuousTransition

from manim import *
import numpy as np
from scipy.interpolate import interp1d, bisplrep, bisplev

class DiscreteToContinuousTransition(Scene):
    """
    A Manim scene that visualizes the transition from discrete data points
    to a continuous surface, demonstrating the core concept of functional analysis
    in optical design.
    
    This animation shows:
    1. Discrete measurement points from an interferometer
    2. Spline interpolation creating a continuous surface
    3. The transformation from point cloud to smooth function
    """
    
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # --- 1. INTRODUCTION: Title and Concept ---
        title = Text("从离散到连续：光学曲面建模", font_size=48, color=BLACK)
        subtitle = Text("离散测量数据 → 连续函数", font_size=32, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # --- 2. DISCRETE DATA: Interferometer Measurement Points ---
        
        # Create a grid of discrete measurement points (simulating interferometer data)
        # These represent wavefront error measurements at discrete points
        grid_size = 8
        x_range = np.linspace(-3, 3, grid_size)
        y_range = np.linspace(-3, 3, grid_size)
        
        # Generate synthetic wavefront error data (combination of defocus and astigmatism)
        def wavefront_error(x, y):
            return 0.3 * (x**2 + y**2) + 0.2 * (x**2 - y**2)  # defocus + astigmatism
        
        # Create 3D points with color coding based on error magnitude
        discrete_points = VGroup()
        point_colors = []
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                z = wavefront_error(x, y)
                point = Dot3D(point=[x, y, z], radius=0.08)
                
                # Color based on wavefront error magnitude
                error_mag = abs(z)
                if error_mag < 0.5:
                    color = BLUE
                elif error_mag < 1.0:
                    color = GREEN
                elif error_mag < 1.5:
                    color = YELLOW
                else:
                    color = RED
                
                point.set_color(color)
                discrete_points.add(point)
                point_colors.append(color)
        
        # Label for discrete data
        discrete_label = Text("离散测量点 (干涉仪数据)", font_size=28, color=BLACK)
        discrete_label.to_edge(LEFT, buff=0.5)
        discrete_label.shift(UP * 2)
        
        self.play(FadeIn(discrete_label))
        self.play(LaggedStart(*[FadeIn(point) for point in discrete_points], lag_ratio=0.05))
        self.wait(2)
        
        # --- 3. COORDINATE SYSTEM: Show 3D axes ---
        
        # Create 3D axes for reference
        axes_3d = ThreeDAxes(
            x_range=(-4, 4, 1),
            y_range=(-4, 4, 1),
            z_range=(-2, 4, 1),
            x_length=8,
            y_length=8,
            z_length=6,
            axis_config={"color": GRAY, "stroke_width": 2}
        )
        
        axis_labels = axes_3d.get_axis_labels(x_label="x", y_label="y", z_label="W(x,y)")
        
        self.play(Create(axes_3d), Write(axis_labels))
        self.wait(1)
        
        # --- 4. TRANSITION: Show interpolation process ---
        
        # Create intermediate visualization showing interpolation
        transition_text = Text("样条插值：构建连续曲面", font_size=32, color=BLACK)
        transition_text.to_edge(RIGHT, buff=0.5)
        transition_text.shift(UP * 2)
        
        self.play(Write(transition_text))
        
        # Show interpolation process by creating connecting lines
        # First, show 1D interpolation concept
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)
        
        # Create a 2D slice to show interpolation concept
        slice_y = 0
        slice_points = VGroup()
        slice_x_vals = []
        slice_z_vals = []
        
        for i, x in enumerate(x_range):
            z = wavefront_error(x, slice_y)
            point_2d = Dot(point=[x, z, 0], radius=0.08, color=RED)
            slice_points.add(point_2d)
            slice_x_vals.append(x)
            slice_z_vals.append(z)
        
        # Show 1D interpolation curve
        interp_func = interp1d(slice_x_vals, slice_z_vals, kind='cubic')
        x_fine = np.linspace(-3, 3, 100)
        z_fine = interp_func(x_fine)
        
        interp_curve = VMobject()
        interp_curve.set_points_smoothly([np.array([x, z, 0]) for x, z in zip(x_fine, z_fine)])
        interp_curve.set_stroke(color=RED, width=3)
        
        self.play(
            FadeOut(discrete_label),
            FadeIn(slice_points)
        )
        self.wait(1)
        
        self.play(Create(interp_curve))
        self.wait(2)
        
        # --- 5. CONTINUOUS SURFACE: Show final result ---
        
        # Create the full 2D interpolated surface
        # Generate denser grid for smooth surface
        x_fine = np.linspace(-3, 3, 50)
        y_fine = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_fine, y_fine)
        
        # Use radial basis function interpolation for 2D surface
        from scipy.interpolate import Rbf
        
        # Create interpolation function from discrete points
        x_points = []
        y_points = []
        z_points = []
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                x_points.append(x)
                y_points.append(y)
                z_points.append(wavefront_error(x, y))
        
        rbf_interp = Rbf(x_points, y_points, z_points, function='multiquadric', epsilon=1.0)
        Z_fine = rbf_interp(X, Y)
        
        # Create the continuous surface
        continuous_surface = Surface(
            lambda u, v: np.array([u, v, rbf_interp(u, v)]),
            u_range=(-3, 3),
            v_range=(-3, 3),
            resolution=(40, 40),
            fill_opacity=0.7,
            stroke_color=BLUE,
            stroke_width=1
        )
        
        # Color the surface based on height
        continuous_surface.set_fill_by_value(
            axes=axes_3d,
            colorscale=[(BLUE, -2), (GREEN, 0), (YELLOW, 2), (RED, 4)]
        )
        
        # Final labels
        final_label = Text("连续函数 W(x,y)", font_size=32, color=BLACK)
        final_label.to_edge(RIGHT, buff=0.5)
        final_label.shift(UP * 2)
        
        # Transform from discrete to continuous
        self.play(
            FadeOut(slice_points),
            FadeOut(interp_curve),
            FadeOut(transition_text)
        )
        
        # Show the transformation
        self.play(
            Transform(discrete_points, continuous_surface),
            run_time=3
        )
        
        self.play(FadeIn(final_label))
        self.wait(2)
        
        # --- 6. MATHEMATICAL FORMULA: Show the concept ---
        
        # Show the mathematical concept
        formula = MathTex(
            r"W(x,y) = \sum_{i,j} c_{ij} \cdot \phi_{ij}(x,y)",
            font_size=36,
            color=BLACK
        )
        formula.to_edge(DOWN, buff=0.5)
        
        explanation = Text(
            "基函数展开：离散系数 → 连续函数",
            font_size=24,
            color=GRAY
        )
        explanation.next_to(formula, DOWN, buff=0.2)
        
        self.play(Write(formula))
        self.play(FadeIn(explanation))
        self.wait(3)
        
        # --- 7. CONCLUSION: Rotate camera for better view ---
        
        # Rotate camera to show the surface better
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # Fade out everything
        self.play(
            FadeOut(continuous_surface),
            FadeOut(axes_3d),
            FadeOut(axis_labels),
            FadeOut(final_label),
            FadeOut(formula),
            FadeOut(explanation),
            FadeOut(title_group)
        )
        
        self.wait(1)


class SplineInterpolationDemo(Scene):
    """
    A focused demonstration of spline interpolation showing how discrete points
    are connected by smooth curves, emphasizing C2 continuity important for optical surfaces.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("样条插值：C² 连续性保证", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        # Create control points (discrete data)
        control_points = [
            np.array([-4, -1, 0]),
            np.array([-2, 1, 0]),
            np.array([0, -0.5, 0]),
            np.array([2, 2, 0]),
            np.array([4, 0, 0])
        ]
        
        # Show control points
        dots = VGroup(*[Dot(point, color=RED, radius=0.08) for point in control_points])
        
        # Labels for control points
        labels = VGroup()
        for i, point in enumerate(control_points):
            label = Text(f"P_{i}", font_size=20, color=BLACK)
            label.next_to(point, DOWN, buff=0.1)
            labels.add(label)
        
        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(label) for label in labels], lag_ratio=0.1))
        self.wait(1)
        
        # Show different interpolation methods
        methods = [
            ("线性插值", "linear", BLUE),
            ("三次样条", "cubic", GREEN),
            ("B样条", "quadratic", PURPLE)
        ]
        
        x_vals = [point[0] for point in control_points]
        y_vals = [point[1] for point in control_points]
        
        for method_name, kind, color in methods:
            # Create interpolation
            if kind == "quadratic":
                # For B-spline, use scipy's splrep
                from scipy.interpolate import splrep, splev
                tck = splrep(x_vals, y_vals, k=2, s=0)
                x_fine = np.linspace(-4, 4, 200)
                y_fine = splev(x_fine, tck)
            else:
                interp_func = interp1d(x_vals, y_vals, kind=kind)
                x_fine = np.linspace(-4, 4, 200)
                y_fine = interp_func(x_fine)
            
            # Create curve
            curve = VMobject()
            curve.set_points_smoothly([np.array([x, y, 0]) for x, y in zip(x_fine, y_fine)])
            curve.set_stroke(color=color, width=3)
            
            method_label = Text(method_name, font_size=28, color=color)
            method_label.to_edge(RIGHT, buff=0.5)
            
            self.play(
                FadeIn(method_label),
                Create(curve),
                run_time=2
            )
            
            if method_name == "三次样条":
                # Highlight C2 continuity for cubic spline
                continuity_text = Text(
                    "C² 连续性：函数、一阶、二阶导数连续",
                    font_size=20,
                    color=GREEN
                )
                continuity_text.to_edge(DOWN, buff=0.5)
                self.play(FadeIn(continuity_text))
                self.wait(2)
                self.play(FadeOut(continuity_text))
            
            self.wait(1)
            
            if method_name != methods[-1][0]:  # Don't fade out the last one
                self.play(FadeOut(curve), FadeOut(method_label))
        
        # Final emphasis on optical importance
        optical_text = Text(
            "光学曲面需要光滑连接，避免光线散射",
            font_size=24,
            color=BLACK
        )
        optical_text.to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(optical_text))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(dots),
            FadeOut(labels),
            FadeOut(optical_text)
        )
        
        self.wait(1)