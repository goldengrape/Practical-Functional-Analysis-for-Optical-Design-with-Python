# EulerLagrangeDerivation.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render EulerLagrangeDerivation.py EulerLagrangeDerivation

from manim import *
import numpy as np
from scipy.optimize import minimize_scalar

class EulerLagrangeDerivation(Scene):
    """
    Step-by-step derivation of the Euler-Lagrange equation from the calculus of variations.
    Shows how minimizing a functional leads to the fundamental differential equation.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("欧拉-拉格朗日方程推导", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Step 1: Functional definition
        step1 = Text("步骤 1: 泛函定义", font_size=24, color=BLUE)
        step1.to_edge(LEFT, buff=0.5)
        step1.shift(UP * 2.5)
        
        functional_def = MathTex(
            r"J[y(x)] = \int_{x_1}^{x_2} F(x, y, y') dx",
            font_size=28,
            color=BLACK
        )
        functional_def.next_to(step1, DOWN, buff=0.3)
        
        self.play(Write(step1))
        self.play(Write(functional_def))
        self.wait(2)
        
        # Step 2: Variation of the function
        step2 = Text("步骤 2: 函数变分", font_size=24, color=GREEN)
        step2.next_to(functional_def, DOWN, buff=0.5)
        
        variation_def = MathTex(
            r"\tilde{y}(x) = y(x) + \epsilon \eta(x)",
            font_size=28,
            color=BLACK
        )
        variation_def.next_to(step2, DOWN, buff=0.3)
        
        # Boundary conditions
        boundary_cond = MathTex(
            r"\eta(x_1) = \eta(x_2) = 0",
            font_size=24,
            color=GRAY
        )
        boundary_cond.next_to(variation_def, DOWN, buff=0.2)
        
        self.play(Write(step2))
        self.play(Write(variation_def))
        self.play(Write(boundary_cond))
        self.wait(2)
        
        # Step 3: First variation of the functional
        step3 = Text("步骤 3: 泛函的一阶变分", font_size=24, color=ORANGE)
        step3.next_to(boundary_cond, DOWN, buff=0.5)
        
        delta_j = MathTex(
            r"\delta J = \frac{d}{d\epsilon} J[y + \epsilon \eta] \bigg|_{\epsilon=0}",
            font_size=28,
            color=BLACK
        )
        delta_j.next_to(step3, DOWN, buff=0.3)
        
        self.play(Write(step3))
        self.play(Write(delta_j))
        self.wait(2)
        
        # Step 4: Expand the integrand
        step4 = Text("步骤 4: 展开被积函数", font_size=24, color=PURPLE)
        step4.to_edge(RIGHT, buff=0.5)
        step4.shift(UP * 2.5)
        
        taylor_expansion = MathTex(
            r"F(x, y + \epsilon \eta, y' + \epsilon \eta') \\",
            r"= F(x, y, y') + \epsilon \eta \frac{\partial F}{\partial y} + \epsilon \eta' \frac{\partial F}{\partial y'} + O(\epsilon^2)",
            font_size=24,
            color=BLACK
        )
        taylor_expansion.next_to(step4, DOWN, buff=0.3)
        
        self.play(Write(step4))
        self.play(Write(taylor_expansion))
        self.wait(2)
        
        # Step 5: Integration by parts
        step5 = Text("步骤 5: 分部积分", font_size=24, color=RED)
        step5.next_to(taylor_expansion, DOWN, buff=0.5)
        
        integration_by_parts = MathTex(
            r"\int_{x_1}^{x_2} \eta' \frac{\partial F}{\partial y'} dx \\",
            r"= \left[\eta \frac{\partial F}{\partial y'}\right]_{x_1}^{x_2} - \int_{x_1}^{x_2} \eta \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) dx \\",
            r"= -\int_{x_1}^{x_2} \eta \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) dx",
            font_size=22,
            color=BLACK
        )
        integration_by_parts.next_to(step5, DOWN, buff=0.3)
        
        self.play(Write(step5))
        self.play(Write(integration_by_parts))
        self.wait(2)
        
        # Step 6: Collect terms
        step6 = Text("步骤 6: 合并同类项", font_size=24, color=TEAL)
        step6.next_to(integration_by_parts, DOWN, buff=0.5)
        
        final_variation = MathTex(
            r"\delta J = \epsilon \int_{x_1}^{x_2} \eta \left[ \frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) \right] dx = 0",
            font_size=20,
            color=BLACK
        )
        final_variation.next_to(step6, DOWN, buff=0.3)
        
        self.play(Write(step6))
        self.play(Write(final_variation))
        self.wait(2)
        
        # Step 7: Fundamental lemma
        step7 = Text("步骤 7: 基本引理", font_size=24, color=PINK)
        step7.to_edge(DOWN, buff=0.5)
        step7.shift(LEFT * 2)
        
        fundamental_lemma = MathTex(
            r"\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0",
            font_size=28,
            color=BLACK
        )
        fundamental_lemma.next_to(step7, DOWN, buff=0.3)
        
        # Highlight the final equation
        box = SurroundingRectangle(fundamental_lemma, buff=0.2, color=RED)
        
        self.play(Write(step7))
        self.play(Write(fundamental_lemma))
        self.play(Create(box))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(step1),
            FadeOut(functional_def),
            FadeOut(step2),
            FadeOut(variation_def),
            FadeOut(boundary_cond),
            FadeOut(step3),
            FadeOut(delta_j),
            FadeOut(step4),
            FadeOut(taylor_expansion),
            FadeOut(step5),
            FadeOut(integration_by_parts),
            FadeOut(step6),
            FadeOut(final_variation),
            FadeOut(step7),
            FadeOut(fundamental_lemma),
            FadeOut(box)
        )
        
        self.wait(1)


class EulerLagrangeVisualization(Scene):
    """
    Visual demonstration of the Euler-Lagrange equation showing how small variations
    affect the functional and lead to the extremum condition.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("欧拉-拉格朗日方程可视化", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Create coordinate system
        axes = Axes(
            x_range=(-1, 5),
            y_range=(-1, 3),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=8,
            y_length=4
        )
        axes.to_edge(LEFT, buff=0.5)
        
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        self.play(Create(axes), Write(axis_labels))
        
        # Define the extremal function (solution to E-L equation)
        def extremal_function(x):
            return np.sin(x)  # Simple example
        
        # Define a competing function with parameter
        def competing_function(x, epsilon):
            return np.sin(x) + epsilon * x * (4 - x)  # Variation that vanishes at boundaries
        
        # Plot the extremal function
        x_vals = np.linspace(0, 4, 100)
        extremal_points = [axes.c2p(x, extremal_function(x), 0) for x in x_vals]
        
        extremal_curve = VMobject()
        extremal_curve.set_points_smoothly(extremal_points)
        extremal_curve.set_stroke(color=RED, width=4)
        
        extremal_label = Text("极值函数", font_size=20, color=RED)
        extremal_label.next_to(extremal_curve, UP, buff=0.2)
        
        self.play(Create(extremal_curve), Write(extremal_label))
        
        # Create a ValueTracker for epsilon
        epsilon_tracker = ValueTracker(0.0)
        
        # Plot competing function with updater
        competing_curve = always_redraw(lambda: self.create_competing_curve(
            axes, x_vals, lambda x: competing_function(x, epsilon_tracker.get_value()), BLUE
        ))
        
        competing_label = Text("变分函数", font_size=20, color=BLUE)
        competing_label.next_to(competing_curve, DOWN, buff=0.2)
        
        self.play(Create(competing_curve), Write(competing_label))
        
        # Calculate and display functional values
        def functional_value(func):
            # Example functional: J[y] = ∫[0 to 4] (y'² - y²) dx
            from scipy.integrate import quad
            
            def integrand(x):
                h = 1e-5
                y_prime = (func(x + h) - func(x - h)) / (2 * h)
                return y_prime**2 - func(x)**2
            
            value, _ = quad(integrand, 0, 4)
            return value
        
        # Display functional values
        extremal_func_val = functional_value(extremal_function)
        
        functional_label = Text("泛函值:", font_size=20, color=BLACK)
        functional_label.to_edge(RIGHT, buff=0.5)
        functional_label.shift(UP * 2)
        
        extremal_value_text = always_redraw(lambda: Text(
            f"极值: {extremal_func_val:.4f}", font_size=18, color=RED
        ).next_to(functional_label, DOWN, buff=0.2))
        
        competing_value_text = always_redraw(lambda: Text(
            f"变分: {functional_value(lambda x: competing_function(x, epsilon_tracker.get_value())):.4f}", 
            font_size=18, color=BLUE
        ).next_to(extremal_value_text, DOWN, buff=0.2))
        
        self.play(
            Write(functional_label),
            Write(extremal_value_text),
            Write(competing_value_text)
        )
        
        # Animate epsilon changes
        self.play(
            epsilon_tracker.animate.set_value(0.5),
            run_time=2
        )
        self.play(
            epsilon_tracker.animate.set_value(-0.5),
            run_time=2
        )
        self.play(
            epsilon_tracker.animate.set_value(0.0),
            run_time=2
        )
        
        # Show that extremal gives minimum
        minimum_text = Text("极值函数使泛函取极值", font_size=20, color=BLACK)
        minimum_text.to_edge(RIGHT, buff=0.5)
        minimum_text.shift(DOWN * 2)
        
        box = SurroundingRectangle(minimum_text, buff=0.2, color=GREEN)
        
        self.play(Write(minimum_text), Create(box))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(extremal_curve),
            FadeOut(extremal_label),
            FadeOut(competing_curve),
            FadeOut(competing_label),
            FadeOut(functional_label),
            FadeOut(extremal_value_text),
            FadeOut(competing_value_text),
            FadeOut(minimum_text),
            FadeOut(box)
        )
        
        self.wait(1)
    
    def create_competing_curve(self, axes, x_vals, func, color):
        """Helper method to create a curve from a function."""
        points = [axes.c2p(x, func(x), 0) for x in x_vals]
        curve = VMobject()
        curve.set_points_smoothly(points)
        curve.set_stroke(color=color, width=3)
        return curve