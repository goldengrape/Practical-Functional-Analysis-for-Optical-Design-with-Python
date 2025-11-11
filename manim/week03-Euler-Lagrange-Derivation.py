# -*- coding: utf-8 -*-
from manim import *
import numpy as np
import icontract

# ==============================================================================
# 1. Core Mathematical Logic (Functional & Data-Oriented Programming)
# ==============================================================================
# The core logic is implemented as pure functions operating on NumPy arrays.

@icontract.require(lambda start_point, end_point: len(start_point) == 2 and len(end_point) == 2)
def optimal_path_func(x: np.ndarray, start_point: tuple, end_point: tuple) -> np.ndarray:
    """
    Calculates the y-values for the optimal path (a straight line) between two points.
    
    :param x: NumPy array of x-coordinates.
    :param start_point: Tuple (x0, y0).
    :param end_point: Tuple (x1, y1).
    :return: NumPy array of y-coordinates.
    """
    x0, y0 = start_point
    x1, y1 = end_point
    # Handle vertical line case to avoid division by zero, although unlikely in this viz
    if np.isclose(x1, x0):
        return np.full_like(x, (y0 + y1) / 2)
    slope = (y1 - y0) / (x1 - x0)
    return slope * (x - x0) + y0

@icontract.require(lambda x, x_min, x_max: x_max > x_min)
def perturbation_func(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """
    Calculates a perturbation function eta(x) that is zero at the boundaries.
    A sine wave is a common and visually clear choice.
    
    :param x: NumPy array of x-coordinates.
    :param x_min: The starting x-coordinate where eta(x) should be zero.
    :param x_max: The ending x-coordinate where eta(x) should be zero.
    :return: NumPy array of perturbation values.
    """
    # Ensure the sine wave completes one half-period over the interval [x_min, x_max]
    return np.sin(np.pi * (x - x_min) / (x_max - x_min))

@icontract.require(lambda y_vals, x_vals: y_vals.shape == x_vals.shape)
def calculate_functional_J(y_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """
    Calculates the value of the functional J[y] for the shortest path problem.
    J[y] = integral( sqrt(1 + (y')^2) dx )
    
    :param y_vals: Discretized y-values of the function.
    :param x_vals: Discretized x-values of the function.
    :return: The scalar value of the functional (total path length).
    """
    # Numerically calculate the derivative y' using finite differences
    y_prime = np.gradient(y_vals, x_vals)
    
    # Define the Lagrangian L(x, y, y') for the shortest path
    lagrangian = np.sqrt(1 + y_prime**2)
    
    # Integrate the Lagrangian over the interval using the trapezoidal rule
    return np.trapz(lagrangian, x_vals)


# ==============================================================================
# 2. Manim Visualization Scene (Axiomatic Design: Separating Concerns)
# ==============================================================================
# The Manim scene is responsible for visualization only. 
# It calls the pure mathematical functions defined above.

class EulerLagrangePerturbation(Scene):
    """
    A Manim scene to visualize the core principle of the Euler-Lagrange equation:
    for an optimal function f, the first variation of the functional J is zero.
    """
    def construct(self):
        # --- Configuration ---
        start_point = (-4, -1)
        end_point = (4, 1)
        x_range = (start_point[0], end_point[0], 0.01) # (min, max, step)
        
        # --- Manim Axes Setup ---
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": np.arange(-6, 7, 2)},
            y_axis_config={"numbers_to_include": np.arange(-3, 4, 1)},
        ).add_coordinates()
        axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")
        
        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # --- Define Paths and Data using our functional core ---
        x_coords = np.arange(*x_range)
        
        # Path 1: The Optimal Path f(x)
        optimal_y = optimal_path_func(x_coords, start_point, end_point)
        optimal_graph = axes.plot_line_graph(x_coords, optimal_y, line_color=GREEN)
        optimal_label = MathTex("f(x)", color=GREEN).next_to(optimal_graph, UP, buff=0.2)

        # Path 2: The Perturbation eta(x)
        eta_y = perturbation_func(x_coords, start_point[0], end_point[0])
        eta_graph = axes.plot_line_graph(x_coords, eta_y, line_color=YELLOW, line_opacity=0.7)
        eta_label = MathTex("\\eta(x)", color=YELLOW).next_to(eta_graph.get_top(), UP, buff=0.2)
        
        # --- Animation Step 1: Introduce the Optimal Path ---
        self.play(
            Write(Tex("考虑两点间最短路径问题", font_size=36).to_edge(UP)),
        )
        self.play(Create(optimal_graph), Write(optimal_label))
        self.wait(1)

        # --- Animation Step 2: Introduce the Perturbation ---
        self.play(
            Write(Tex("我们给最优路径施加一个微小扰动", font_size=36).to_edge(UP).shift(DOWN*0.8))
        )
        self.play(Create(eta_graph), Write(eta_label))
        self.wait(1)

        # --- Animation Step 3: Show the Perturbed Path and Functional Value ---
        epsilon = ValueTracker(0.0)
        
        # The perturbed path is a dynamic object that updates with epsilon
        perturbed_graph = always_redraw(
            lambda: axes.plot_line_graph(
                x_coords,
                optimal_y + epsilon.get_value() * eta_y,
                line_color=RED
            )
        )
        perturbed_label = always_redraw(
            lambda: MathTex("f(x) + \\epsilon \\eta(x)", color=RED).next_to(
                perturbed_graph, DOWN, buff=0.2
            )
        )

        # Text to display dynamic values of epsilon and J
        text_group = VGroup(
            MathTex("\\epsilon = "),
            DecimalNumber(epsilon.get_value(), num_decimal_places=2, show_ellipsis=False),
            MathTex("J[f+\\epsilon\\eta] = ")
        ).arrange(RIGHT).to_edge(DOWN)
        
        cost_value = always_redraw(
            lambda: DecimalNumber(
                calculate_functional_J(optimal_y + epsilon.get_value() * eta_y, x_coords),
                num_decimal_places=4
            ).next_to(text_group, RIGHT)
        )
        text_group[1].add_updater(lambda d: d.set_value(epsilon.get_value()))
        
        self.play(
            FadeOut(eta_graph, eta_label),
            FadeIn(perturbed_graph, perturbed_label, text_group, cost_value),
            run_time=1.5
        )
        self.wait(1)

        # Animate epsilon changing, showing the effect on the path and cost
        self.play(epsilon.animate.set_value(1.5), run_time=3)
        self.play(epsilon.animate.set_value(-1.5), run_time=3)
        self.play(epsilon.animate.set_value(0.0), run_time=2)
        self.wait(1)

        # --- Animation Step 4: The Conclusion - First Variation is Zero ---
        conclusion_text = Tex(
            "当路径最优时，泛函 $J$ 的一阶变分 $\\delta J$ 为零。",
            "即 $\\frac{dJ}{d\\epsilon} \\bigg|_{\\epsilon=0} = 0$",
            font_size=36
        ).to_edge(UP)
        
        self.play(FadeOut(text_group, cost_value, perturbed_label), Transform(perturbed_graph, optimal_graph))
        self.play(Write(conclusion_text[0]))
        self.wait(1)
        self.play(Write(conclusion_text[1]))
        
        # Draw the tangent line at epsilon=0 on a conceptual plot (optional but effective)
        # Here we add a conceptual plot of J vs epsilon
        j_vs_eps_axes = Axes(
            x_range=[-2, 2, 0.5], y_range=[8, 9, 0.2], 
            x_length=5, y_length=3,
            axis_config={"include_tip": False}
        ).to_corner(DR)
        j_vs_eps_labels = j_vs_eps_axes.get_axis_labels(x_label="\\epsilon", y_label="J")
        
        eps_vals = np.linspace(-1.5, 1.5, 50)
        j_vals = [calculate_functional_J(optimal_y + e * eta_y, x_coords) for e in eps_vals]
        
        j_curve = j_vs_eps_axes.plot_line_graph(eps_vals, j_vals, line_color=ORANGE)
        tangent = j_vs_eps_axes.get_secant_slope_group(
            x=0.0, graph=j_curve, dx=0.01, secant_line_length=2, secant_line_color=YELLOW
        )

        self.play(
            FadeIn(j_vs_eps_axes, j_vs_eps_labels),
            Create(j_curve)
        )
        self.play(Create(tangent))
        self.wait(3)