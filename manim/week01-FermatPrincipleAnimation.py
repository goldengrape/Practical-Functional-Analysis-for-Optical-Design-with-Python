# FermatPrincipleAnimation.py
#
# To run this animation, ensure you have Manim and scipy installed:
# pip install manim scipy icontract
#
# Then, execute the following command in your terminal:
# manim -pqh render FermatPrincipleAnimation.py FermatPrincipleVisualization

from manim import *
import numpy as np
from scipy.optimize import fsolve
import icontract
from typing import Callable, Tuple

# Design by Contract: Define a pure function for OPL calculation.
# This function has a clear contract: given points and refractive indices, it returns a float.
def calculate_opl(p_test: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n1: float, n2: float) -> float:
    """Calculates the Optical Path Length (OPL) for a path from A to B via P."""
    # OPL = n1 * distance(A, P) + n2 * distance(P, B)
    l1 = np.linalg.norm(p_a - p_test)
    l2 = np.linalg.norm(p_b - p_test)
    return n1 * l1 + n2 * l2

# Design by Contract & Axiomatic Design:
# This function is a self-contained solver for the physical problem (FR: Find minimal path).
# It is independent of any visualization logic.
@icontract.require(lambda n1, n2: n1 > 0 and n2 > 0)
def solve_snell_intercept(p_a: np.ndarray, p_b: np.ndarray, n1: float, n2: float) -> float:
    """
    Finds the x-coordinate on the interface (y=0) that satisfies Snell's Law
    by finding the root of the OPL derivative.

    This is the point where the OPL is stationary (minimal in this case).

    The derivative of OPL(x) w.r.t. x is:
    d(OPL)/dx = n1 * (x - p_a[0]) / L1 + n2 * (x - p_b[0]) / L2
    This is equivalent to n1*sin(theta1) - n2*sin(theta2) = 0.
    """
    
    def opl_derivative(x: float) -> float:
        """The derivative of the OPL with respect to the intercept's x-coordinate."""
        # This is a pure function used for root finding.
        p_test = np.array([x, 0, 0])
        l1 = np.linalg.norm(p_a - p_test)
        l2 = np.linalg.norm(p_b - p_test)

        # Avoid division by zero if a point is on the interface
        if l1 < 1e-9 or l2 < 1e-9:
            return 0.0

        term1 = n1 * (x - p_a[0]) / l1
        term2 = n2 * (x - p_b[0]) / l2
        return term1 + term2

    # Use a robust numerical solver to find the root of the derivative.
    # Initial guess is the geometric mean of the x-coordinates.
    initial_guess = (p_a[0] + p_b[0]) / 2.0
    solution = fsolve(opl_derivative, initial_guess)
    return solution[0]


class FermatPrincipleVisualization(Scene):
    """
    A Manim scene to visualize Fermat's Principle of Least Time.
    It shows that light follows the path that minimizes the Optical Path Length (OPL)
    when traveling between two points in different media.
    """
    def construct(self):
        # --- 1. SETUP: Define the physical and visual parameters ---
        # Data-Oriented: The entire animation is driven by these initial data points.
        n1, n2 = 1.00, 1.50  # Refractive indices (e.g., air to glass)
        color_1, color_2 = BLUE_E, TEAL_E
        
        # Coordinates of start (A) and end (B) points
        p_a = np.array([-4, 2.5, 0])
        p_b = np.array([4, -2.5, 0])

        # --- 2. AXIOMATIC DESIGN: Create independent visual components ---
        
        # The interface between the two media
        interface = Line(LEFT * 7, RIGHT * 7, color=BLACK, stroke_width=2)
        
        # Visual representation of the media
        medium_1 = Rectangle(height=4, width=14, stroke_width=0,
                             fill_color=color_1, fill_opacity=0.2).to_edge(UP, buff=0)
        medium_2 = Rectangle(height=4, width=14, stroke_width=0,
                             fill_color=color_2, fill_opacity=0.3).to_edge(DOWN, buff=0)

        # Start and end points and their labels
        dot_a = Dot(p_a, color=RED)
        dot_b = Dot(p_b, color=RED)
        label_a = MathTex(f"A", "(n_1={n1:.2f})", color=BLACK).next_to(dot_a, UL, buff=0.2)
        label_b = MathTex(f"B", "(n_2={n2:.2f})", color=BLACK).next_to(dot_b, DR, buff=0.2)
        
        self.add(medium_1, medium_2, interface, dot_a, dot_b, label_a, label_b)

        # --- 3. DATA-DRIVEN DYNAMICS: Use a ValueTracker for the test point ---
        # The x-coordinate of the intersection point P is our primary state data.
        test_x_tracker = ValueTracker(-5.0)

        # Functional Programming: Use a getter function (lambda) to derive state.
        get_test_p = lambda: np.array([test_x_tracker.get_value(), 0, 0])

        # Functional Programming: `always_redraw` takes a function to generate the Mobject.
        # The path is a *function* of the tracker's value.
        test_path = always_redraw(
            lambda: VGroup(
                Line(p_a, get_test_p(), color=GRAY, stroke_width=2),
                Line(get_test_p(), p_b, color=GRAY, stroke_width=2)
            )
        )
        test_dot = always_redraw(lambda: Dot(get_test_p(), color=GRAY, radius=0.08))

        # --- 4. REAL-TIME FEEDBACK: OPL display with an updater ---
        
        # OPL Title and value display
        opl_title = Tex("Optical Path Length (OPL):", color=BLACK).to_edge(UP, buff=0.5)
        opl_value_text = MathTex("0.000", color=BLACK).next_to(opl_title, DOWN, buff=0.2)

        # Functional Programming: `add_updater` attaches a function that is called every frame.
        # This updater modifies the text object based on the tracker's state.
        def opl_updater(mob: MathTex) -> None:
            # This function has a side effect (modifying `mob`), which is typical for updaters.
            current_opl = calculate_opl(get_test_p(), p_a, p_b, n1, n2)
            new_text = MathTex(f"{current_opl:.3f}", color=BLACK).move_to(mob)
            mob.become(new_text)

        opl_value_text.add_updater(opl_updater)
        self.add(test_path, test_dot, opl_title, opl_value_text)
        self.wait(1)

        # --- 5. ANIMATION SEQUENCE: Explore the functional's landscape ---
        
        # Animate the tracker, causing the entire scene to update.
        # The viewer "sees" the OPL value change as the input path (defined by x) changes.
        self.play(
            test_x_tracker.animate.set_value(5.0),
            run_time=6,
            rate_func=rate_functions.there_and_back_with_pause
        )
        self.wait(1)

        # --- 6. THE SOLUTION: Find and display the optimal path ---
        
        # Occam's Razor: Use the dedicated solver to find the correct answer directly.
        correct_x = solve_snell_intercept(p_a, p_b, n1, n2)
        p_correct = np.array([correct_x, 0, 0])
        
        # Create the visual representation of the final, correct path.
        correct_path = VGroup(
            Line(p_a, p_correct, color=RED, stroke_width=5),
            Line(p_correct, p_b, color=RED, stroke_width=5)
        )
        correct_dot = Dot(p_correct, color=RED, radius=0.1)

        # Animate the "snap" to the correct solution.
        self.play(
            test_x_tracker.animate(run_time=1.5, rate_func=rate_functions.ease_in_out_sine).set_value(correct_x)
        )
        self.wait(0.5)
        
        # Clean up the scene, leaving only the solution.
        self.remove(test_path, test_dot)
        opl_value_text.clear_updaters() # Stop the text from updating
        
        # Display the final results.
        min_opl_value = calculate_opl(p_correct, p_a, p_b, n1, n2)
        final_text = MathTex(f"{min_opl_value:.3f}", color=RED).move_to(opl_value_text)
        
        result_box = SurroundingRectangle(final_text, buff=0.1, color=RED)
        result_label = Tex("Minimum OPL", color=RED, font_size=36).next_to(result_box, DOWN)

        self.play(
            FadeIn(correct_path, correct_dot, shift=UP),
            Transform(opl_value_text, final_text),
            Create(result_box)
        )
        self.play(Write(result_label))
        self.wait(3)