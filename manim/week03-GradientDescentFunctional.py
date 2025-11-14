# GradientDescentFunctional.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render GradientDescentFunctional.py FunctionalGradientDescentAnimation

from manim import *
import numpy as np
from scipy.integrate import quad

class FunctionalGradientDescentAnimation(Scene):
    """
    Animation demonstrating functional gradient descent in the context of
    optical surface optimization. Shows how we iteratively improve a function
    to minimize a functional.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("泛函梯度下降", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        subtitle = Text("在函数空间中寻找最优解", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        title_group = VGroup(title, subtitle)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Create coordinate system
        axes = Axes(
            x_range=(-3, 3),
            y_range=(-2, 3),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=8,
            y_length=4
        )
        axes.to_edge(LEFT, buff=0.5)
        
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        self.play(Create(axes), Write(axis_labels))
        
        # Define the functional to minimize
        def functional(f, df_dx):
            """
            Example functional: J[f] = ∫[(f')² + (f - sin(x))²] dx
            This measures both smoothness and fidelity to sin(x)
            """
            def integrand(x):
                return (df_dx(x))**2 + (f(x) - np.sin(x))**2
            
            value, _ = quad(integrand, -3, 3)
            return value
        
        # Initial function (poor approximation)
        def initial_function(x):
            return 0.5 * x + 0.2  # Linear approximation
        
        # Generate points for initial function
        x_vals = np.linspace(-3, 3, 100)
        initial_points = [axes.c2p(x, initial_function(x), 0) for x in x_vals]
        
        initial_curve = VMobject()
        initial_curve.set_points_smoothly(initial_points)
        initial_curve.set_stroke(color=RED, width=3)
        
        initial_label = Text("初始函数", font_size=18, color=RED)
        initial_label.next_to(initial_curve, UP, buff=0.2)
        
        # Target function (sin(x))
        target_points = [axes.c2p(x, np.sin(x), 0) for x in x_vals]
        target_curve = VMobject()
        target_curve.set_points_smoothly(target_points)
        target_curve.set_stroke(color=GRAY, width=2, opacity=0.5)
        
        target_label = Text("目标函数", font_size=16, color=GRAY)
        target_label.next_to(target_curve, DOWN, buff=0.2)
        
        self.play(
            Create(initial_curve),
            Write(initial_label),
            Create(target_curve),
            Write(target_label)
        )
        
        # Calculate initial functional value
        def df_dx_initial(x):
            h = 1e-5
            return (initial_function(x + h) - initial_function(x - h)) / (2 * h)
        
        initial_func_val = functional(initial_function, df_dx_initial)
        
        # Display functional value
        func_label = Text("泛函值:", font_size=20, color=BLACK)
        func_label.to_edge(RIGHT, buff=0.5)
        func_label.shift(UP * 2)
        
        func_value_text = Text(f"{initial_func_val:.4f}", font_size=18, color=RED)
        func_value_text.next_to(func_label, RIGHT, buff=0.1)
        
        self.play(Write(func_label), Write(func_value_text))
        
        # Gradient descent iterations
        learning_rate = 0.1
        num_iterations = 5
        
        current_function = initial_function
        current_curve = initial_curve
        current_label = initial_label
        current_func_val = initial_func_val
        
        colors = [ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        
        for i in range(num_iterations):
            # Compute gradient (simplified finite difference)
            def gradient_step(f_prev, x):
                h = 1e-5
                # Compute gradient of functional with respect to function values
                # This is a simplified approximation
                
                # Gradient of (f' )² term
                f_pp = (f_prev(x + h) - 2*f_prev(x) + f_prev(x - h)) / (h**2)
                
                # Gradient of (f - sin(x))² term  
                f_minus_sin = f_prev(x) - np.sin(x)
                
                # Combined gradient
                gradient = -2 * f_pp + 2 * f_minus_sin
                
                return f_prev(x) - learning_rate * gradient
            
            # Create new function
            def new_function(x):
                return gradient_step(current_function, x)
            
            # Generate points for new function
            new_points = [axes.c2p(x, new_function(x), 0) for x in x_vals]
            
            new_curve = VMobject()
            new_curve.set_points_smoothly(new_points)
            new_curve.set_stroke(color=colors[i], width=3)
            
            new_label = Text(f"迭代 {i+1}", font_size=18, color=colors[i])
            new_label.next_to(new_curve, UP, buff=0.2)
            
            # Calculate new functional value
            def df_dx_new(x):
                h = 1e-5
                return (new_function(x + h) - new_function(x - h)) / (2 * h)
            
            new_func_val = functional(new_function, df_dx_new)
            
            # Update functional value display
            new_func_value_text = Text(f"{new_func_val:.4f}", font_size=18, color=colors[i])
            new_func_value_text.next_to(func_label, RIGHT, buff=0.1)
            
            # Animate the transition
            self.play(
                Transform(current_curve, new_curve),
                Transform(current_label, new_label),
                Transform(func_value_text, new_func_value_text),
                run_time=1.5
            )
            
            # Update current function
            current_function = new_function
            current_curve = new_curve
            current_label = new_label
            current_func_val = new_func_val
            
            self.wait(0.5)
        
        # Final result
        final_text = Text("收敛！", font_size=24, color=GREEN)
        final_text.to_edge(RIGHT, buff=0.5)
        final_text.shift(DOWN * 2)
        
        final_box = SurroundingRectangle(final_text, buff=0.2, color=GREEN)
        
        self.play(
            Write(final_text),
            Create(final_box)
        )
        
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title_group),
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(current_curve),
            FadeOut(current_label),
            FadeOut(target_curve),
            FadeOut(target_label),
            FadeOut(func_label),
            FadeOut(func_value_text),
            FadeOut(final_text),
            FadeOut(final_box)
        )
        
        self.wait(1)


class GradientDescentVisualization3D(ThreeDScene):
    """
    3D visualization of functional gradient descent showing the optimization
    landscape and the path taken by the algorithm.
    """
    
    def construct(self):
        # Set up 3D scene
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES, zoom=0.8)
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("3D泛函优化景观", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=(-2, 2, 0.5),
            y_range=(-2, 2, 0.5),
            z_range=(-1, 3, 0.5),
            x_length=6,
            y_length=6,
            z_length=4
        )
        
        axis_labels = axes.get_axis_labels(x_label="a", y_label="b", z_label="J")
        
        self.play(Create(axes), Write(axis_labels))
        
        # Create a functional landscape
        def functional_landscape(a, b):
            """
            Example functional: J(a,b) = (a-1)² + (b+0.5)² + 0.5*sin(2a)*cos(2b)
            """
            return (a - 1)**2 + (b + 0.5)**2 + 0.5 * np.sin(2*a) * np.cos(2*b)
        
        # Create surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, functional_landscape(u, v)),
            u_range=(-2, 2),
            v_range=(-2, 2),
            resolution=(30, 30),
            fill_opacity=0.7,
            stroke_color=BLUE,
            stroke_width=1
        )
        
        # Color by height
        surface.set_fill_by_value(
            axes=axes,
            colorscale=[(BLUE, -1), (GREEN, 0), (YELLOW, 1), (RED, 3)]
        )
        
        self.play(Create(surface))
        
        # Gradient descent path
        def gradient(a, b):
            """Compute gradient of functional"""
            h = 1e-5
            grad_a = (functional_landscape(a + h, b) - functional_landscape(a - h, b)) / (2*h)
            grad_b = (functional_landscape(a, b + h) - functional_landscape(a, b - h)) / (2*h)
            return np.array([grad_a, grad_b])
        
        # Perform gradient descent
        current_point = np.array([-1.5, 1.5])  # Starting point
        learning_rate = 0.1
        num_steps = 20
        
        path_points = [axes.c2p(current_point[0], current_point[1], 
                               functional_landscape(current_point[0], current_point[1]))]
        
        # Create starting point
        start_point = Dot3D(
            point=path_points[0],
            color=RED,
            radius=0.08
        )
        
        self.play(FadeIn(start_point))
        
        # Follow gradient descent path
        for i in range(num_steps):
            # Compute gradient
            grad = gradient(current_point[0], current_point[1])
            
            # Update position
            current_point = current_point - learning_rate * grad
            
            # Add to path
            path_points.append(axes.c2p(current_point[0], current_point[1], 
                                        functional_landscape(current_point[0], current_point[1])))
            
            # Create path segment
            if i > 0:
                path_segment = Line(path_points[-2], path_points[-1], 
                                  color=YELLOW, stroke_width=3)
                self.play(Create(path_segment), run_time=0.1)
        
        # Create final point
        final_point = Dot3D(
            point=path_points[-1],
            color=GREEN,
            radius=0.08
        )
        
        self.play(FadeIn(final_point))
        
        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(surface),
            FadeOut(start_point),
            FadeOut(final_point)
        )
        
        self.wait(1)