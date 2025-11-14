# FermatSnellScene.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render FermatSnellScene.py FermatToSnellDerivation

from manim import *
import numpy as np
from scipy.optimize import fsolve

class FermatToSnellDerivation(Scene):
    """
    Mathematical derivation showing how Fermat's Principle of least time
    leads to Snell's Law of refraction using the Euler-Lagrange equation.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("费马原理 → 斯涅耳定律", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Step 1: Fermat's Principle
        step1 = Text("步骤 1: 费马原理", font_size=24, color=BLUE)
        step1.to_edge(LEFT, buff=0.5)
        step1.shift(UP * 2.5)
        
        fermat_principle = Text(
            "光线沿时间最短的路径传播",
            font_size=20,
            color=BLACK
        )
        fermat_principle.next_to(step1, DOWN, buff=0.3)
        
        opl_functional = MathTex(
            r"\Delta[y(x)] = \int_A^B n(x,y) \sqrt{1 + (y')^2} dx",
            font_size=24,
            color=BLACK
        )
        opl_functional.next_to(fermat_principle, DOWN, buff=0.3)
        
        self.play(Write(step1))
        self.play(Write(fermat_principle))
        self.play(Write(opl_functional))
        self.wait(2)
        
        # Step 2: Apply Euler-Lagrange equation
        step2 = Text("步骤 2: 应用欧拉-拉格朗日方程", font_size=24, color=GREEN)
        step2.next_to(opl_functional, DOWN, buff=0.5)
        
        el_equation = MathTex(
            r"\frac{d}{dx}\left(\frac{\partial L}{\partial y'}\right) - \frac{\partial L}{\partial y} = 0",
            font_size=24,
            color=BLACK
        )
        el_equation.next_to(step2, DOWN, buff=0.3)
        
        # Identify the Lagrangian
        lagrangian = MathTex(
            r"L = n(y) \sqrt{1 + (y')^2}",
            font_size=24,
            color=GRAY
        )
        lagrangian.next_to(el_equation, DOWN, buff=0.2)
        
        self.play(Write(step2))
        self.play(Write(el_equation))
        self.play(Write(lagrangian))
        self.wait(2)
        
        # Step 3: Calculate partial derivatives
        step3 = Text("步骤 3: 计算偏导数", font_size=24, color=ORANGE)
        step3.to_edge(RIGHT, buff=0.5)
        step3.shift(UP * 2.5)
        
        partial_derivatives = VGroup(
            MathTex(
                r"\frac{\partial L}{\partial y} = \frac{dn}{dy} \sqrt{1 + (y')^2}",
                font_size=20,
                color=BLACK
            ),
            MathTex(
                r"\frac{\partial L}{\partial y'} = n(y) \frac{y'}{\sqrt{1 + (y')^2}}",
                font_size=20,
                color=BLACK
            )
        )
        partial_derivatives.arrange(DOWN, buff=0.2)
        partial_derivatives.next_to(step3, DOWN, buff=0.3)
        
        self.play(Write(step3))
        self.play(Write(partial_derivatives))
        self.wait(2)
        
        # Step 4: Substitute into E-L equation
        step4 = Text("步骤 4: 代入E-L方程", font_size=24, color=PURPLE)
        step4.next_to(partial_derivatives, DOWN, buff=0.5)
        
        substitution = MathTex(
            r"\frac{d}{dx}\left(n(y) \frac{y'}{\sqrt{1 + (y')^2}}\right) - \frac{dn}{dy} \sqrt{1 + (y')^2} = 0",
            font_size=18,
            color=BLACK
        )
        substitution.next_to(step4, DOWN, buff=0.3)
        
        self.play(Write(step4))
        self.play(Write(substitution))
        self.wait(2)
        
        # Step 5: Simplify for constant n (homogeneous medium)
        step5 = Text("步骤 5: 均匀介质简化 (n = 常数)", font_size=24, color=PINK)
        step5.next_to(substitution, DOWN, buff=0.5)
        
        simplified = MathTex(
            r"\frac{d}{dx}\left(\frac{y'}{\sqrt{1 + (y')^2}}\right) = 0",
            font_size=20,
            color=BLACK
        )
        simplified.next_to(step5, DOWN, buff=0.3)
        
        self.play(Write(step5))
        self.play(Write(simplified))
        self.wait(2)
        
        # Step 6: First integral
        step6 = Text("步骤 6: 首次积分", font_size=24, color=TEAL)
        step6.to_edge(LEFT, buff=0.5)
        step6.shift(DOWN * 2)
        
        first_integral = MathTex(
            r"\frac{y'}{\sqrt{1 + (y')^2}} = \text{常数} = C_1",
            font_size=20,
            color=BLACK
        )
        first_integral.next_to(step6, DOWN, buff=0.3)
        
        self.play(Write(step6))
        self.play(Write(first_integral))
        self.wait(2)
        
        # Step 7: Solve for y'
        step7 = Text("步骤 7: 解得", font_size=24, color=YELLOW)
        step7.next_to(first_integral, DOWN, buff=0.5)
        
        solution = MathTex(
            r"y' = \frac{C_1}{\sqrt{1 - C_1^2}} = \text{常数} = \tan\theta",
            font_size=20,
            color=BLACK
        )
        solution.next_to(step7, DOWN, buff=0.3)
        
        self.play(Write(step7))
        self.play(Write(solution))
        self.wait(2)
        
        # Step 8: Snell's Law
        step8 = Text("步骤 8: 斯涅耳定律", font_size=24, color=RED)
        step8.to_edge(RIGHT, buff=0.5)
        step8.shift(DOWN * 2)
        
        snell_law = MathTex(
            r"n_1 \sin\theta_1 = n_2 \sin\theta_2",
            font_size=28,
            color=RED
        )
        snell_law.next_to(step8, DOWN, buff=0.3)
        
        # Highlight the final result
        box = SurroundingRectangle(snell_law, buff=0.2, color=RED)
        
        self.play(Write(step8))
        self.play(Write(snell_law))
        self.play(Create(box))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(step1),
            FadeOut(fermat_principle),
            FadeOut(opl_functional),
            FadeOut(step2),
            FadeOut(el_equation),
            FadeOut(lagrangian),
            FadeOut(step3),
            FadeOut(partial_derivatives),
            FadeOut(step4),
            FadeOut(substitution),
            FadeOut(step5),
            FadeOut(simplified),
            FadeOut(step6),
            FadeOut(first_integral),
            FadeOut(step7),
            FadeOut(solution),
            FadeOut(step8),
            FadeOut(snell_law),
            FadeOut(box)
        )
        
        self.wait(1)


class FermatSnellVisualization(Scene):
    """
    Visual demonstration of Fermat's principle leading to Snell's law.
    Shows light rays bending at an interface to minimize travel time.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("费马原理可视化", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Create two media
        medium1 = Rectangle(
            width=8, height=3, stroke_width=2,
            stroke_color=BLACK, fill_color=BLUE, fill_opacity=0.2
        )
        medium1.to_edge(UP, buff=1)
        
        medium2 = Rectangle(
            width=8, height=3, stroke_width=2,
            stroke_color=BLACK, fill_color=GREEN, fill_opacity=0.2
        )
        medium2.to_edge(DOWN, buff=1)
        
        # Interface line
        interface = Line(LEFT * 4, RIGHT * 4, color=BLACK, stroke_width=3)
        
        # Labels
        n1_label = Text("n₁ = 1.0 (空气)", font_size=20, color=BLUE)
        n1_label.next_to(medium1, LEFT, buff=0.2)
        
        n2_label = Text("n₂ = 1.5 (玻璃)", font_size=20, color=GREEN)
        n2_label.next_to(medium2, LEFT, buff=0.2)
        
        self.play(
            Create(medium1),
            Create(medium2),
            Create(interface),
            Write(n1_label),
            Write(n2_label)
        )
        
        # Define start and end points
        start_point = np.array([-2, 1.5, 0])  # In medium 1
        end_point = np.array([2, -1.5, 0])   # In medium 2
        
        start_dot = Dot(start_point, color=RED, radius=0.08)
        end_dot = Dot(end_point, color=RED, radius=0.08)
        
        self.play(FadeIn(start_dot), FadeIn(end_dot))
        
        # Create a ValueTracker for the interface point
        interface_x_tracker = ValueTracker(0.0)
        
        # Function to get current interface point
        get_interface_point = lambda: np.array([interface_x_tracker.get_value(), 0, 0])
        
        # Create light ray with updater
        light_ray = always_redraw(lambda: VGroup(
            Line(start_point, get_interface_point(), color=YELLOW, stroke_width=3),
            Line(get_interface_point(), end_point, color=YELLOW, stroke_width=3)
        ))
        
        # Calculate and display OPL
        opl_label = Text("光程:", font_size=18, color=BLACK)
        opl_label.to_edge(RIGHT, buff=0.5)
        opl_label.shift(UP * 2)
        
        def calculate_opl():
            interface_pt = get_interface_point()
            
            # Calculate path lengths
            path1_length = np.linalg.norm(start_point - interface_pt)
            path2_length = np.linalg.norm(interface_pt - end_point)
            
            # Calculate OPL (n₁ × length₁ + n₂ × length₂)
            opl = 1.0 * path1_length + 1.5 * path2_length
            return opl
        
        opl_value_text = always_redraw(lambda: Text(
            f"{calculate_opl():.3f}", font_size=16, color=BLACK
        ).next_to(opl_label, RIGHT, buff=0.1))
        
        self.play(
            Create(light_ray),
            Write(opl_label),
            Write(opl_value_text)
        )
        
        # Animate the interface point to find minimum
        self.play(
            interface_x_tracker.animate.set_value(1.0),
            run_time=2
        )
        self.play(
            interface_x_tracker.animate.set_value(-1.0),
            run_time=2
        )
        
        # Find the optimal point
        def find_optimal_interface():
            def opl_function(x):
                interface_pt = np.array([x, 0, 0])
                path1_length = np.linalg.norm(start_point - interface_pt)
                path2_length = np.linalg.norm(interface_pt - end_point)
                return 1.0 * path1_length + 1.5 * path2_length
            
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(opl_function, bounds=(-3, 3), method='bounded')
            return result.x
        
        optimal_x = find_optimal_interface()
        
        # Snap to optimal position
        self.play(
            interface_x_tracker.animate.set_value(optimal_x),
            run_time=1.5
        )
        
        # Show angles and Snell's law
        current_interface = get_interface_point()
        
        # Calculate incident and refracted angles
        incident_vector = current_interface - start_point
        refracted_vector = end_point - current_interface
        
        # Normalize and calculate angles with normal (y-axis)
        incident_angle = np.arctan2(incident_vector[0], -incident_vector[1])
        refracted_angle = np.arctan2(refracted_vector[0], -refracted_vector[1])
        
        # Draw normal line
        normal = Line(current_interface + np.array([0, -1, 0]), 
                     current_interface + np.array([0, 1, 0]), 
                     color=GRAY, stroke_width=2, stroke_opacity=0.7)
        
        # Draw angle arcs
        incident_arc = Arc(
            radius=0.5, start_angle=np.pi/2, angle=incident_angle,
            color=BLUE, stroke_width=2
        ).move_to(current_interface)
        
        refracted_arc = Arc(
            radius=0.5, start_angle=np.pi/2, angle=refracted_angle,
            color=GREEN, stroke_width=2
        ).move_to(current_interface)
        
        self.play(
            Create(normal),
            Create(incident_arc),
            Create(refracted_arc)
        )
        
        # Show Snell's law verification
        snell_verification = MathTex(
            f"1.0 \cdot \sin({incident_angle:.2f}) = 1.5 \cdot \sin({refracted_angle:.2f})",
            font_size=16,
            color=BLACK
        )
        snell_verification.to_edge(RIGHT, buff=0.5)
        snell_verification.shift(DOWN * 2)
        
        left_side = 1.0 * np.sin(incident_angle)
        right_side = 1.5 * np.sin(refracted_angle)
        
        verification_result = Text(
            f"{left_side:.3f} ≈ {right_side:.3f} ✓",
            font_size=16,
            color=GREEN
        )
        verification_result.next_to(snell_verification, DOWN, buff=0.1)
        
        self.play(
            Write(snell_verification),
            Write(verification_result)
        )
        
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(medium1),
            FadeOut(medium2),
            FadeOut(interface),
            FadeOut(n1_label),
            FadeOut(n2_label),
            FadeOut(start_dot),
            FadeOut(end_dot),
            FadeOut(light_ray),
            FadeOut(opl_label),
            FadeOut(opl_value_text),
            FadeOut(normal),
            FadeOut(incident_arc),
            FadeOut(refracted_arc),
            FadeOut(snell_verification),
            FadeOut(verification_result)
        )
        
        self.wait(1)