# manim -pql scene.py FermatSnellScene

from manim import *

# --- Configuration & Style ---
# A centralized place for styling constants improves maintainability.
config.background_color = "#1E1E1E"
TEXT_COLOR = WHITE
ACCENT_COLOR = YELLOW
PATH_COLOR = ORANGE

# Properties for the two media
MEDIUM_1_COLOR = BLUE_E
MEDIUM_2_COLOR = BLUE_D
MEDIUM_1_OPACITY = 0.3
MEDIUM_2_OPACITY = 0.4

class FermatSnellScene(Scene):
    """
    An animation demonstrating the derivation of Snell's Law from Fermat's Principle.
    The narrative structure is as follows:
    1.  Introduce Fermat's Principle of Least Time.
    2.  Set up the physical scene: two media, a boundary, and two points A and B.
    3.  Define the light path and the geometric variables (x, h1, h2, d, angles).
    4.  Construct the total time equation T(x) as a function of the crossing point x.
    5.  Apply the principle of calculus to find the minimum: dT/dx = 0.
    6.  Visually connect the mathematical terms in the derivative to the geometric sines of the angles.
    7.  Arrive at the final form of Snell's Law.
    """
    def construct(self):
        """Orchestrates the animation sequence."""
        self.show_title()
        self.setup_scene()
        self.build_path_and_variables()
        self.derive_time_equation()
        self.perform_calculus()
        self.connect_geometry_to_math()
        self.show_final_result()
        self.wait(3)

    def show_title(self) -> None:
        """
        Displays the introductory title and Fermat's Principle.
        This function's contract is to introduce the topic to the viewer.
        """
        title = Text("费马原理推导斯涅尔定律", font_size=48)
        principle = Text("光线在两点之间传播的路径，是所需时间最短的路径。", font_size=28, color=ACCENT_COLOR)
        
        VGroup(title, principle).arrange(DOWN, buff=0.5)

        self.play(Write(title))
        self.play(Write(principle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(principle))

    def setup_scene(self) -> None:
        """
        Creates the physical environment for the derivation.
        This includes the media, boundary, and the start/end points.
        Its contract is to prepare a static visual stage for the dynamic parts.
        """
        # Create the boundary line
        self.boundary = Line(LEFT * 7, RIGHT * 7, color=WHITE)

        # Define the two media as rectangles
        medium1 = Rectangle(width=14, height=4).move_to(UP * 2)
        medium1.set_fill(MEDIUM_1_COLOR, opacity=MEDIUM_1_OPACITY).set_stroke(width=0)
        medium2 = Rectangle(width=14, height=4).move_to(DOWN * 2)
        medium2.set_fill(MEDIUM_2_COLOR, opacity=MEDIUM_2_OPACITY).set_stroke(width=0)
        
        # Labels for the media properties
        props1 = MathTex("n_1, v_1 = c/n_1").move_to(UP * 3 + LEFT * 5)
        props2 = MathTex("n_2, v_2 = c/n_2").move_to(DOWN * 3 + LEFT * 5)

        # Define start and end points
        self.A_coord = UP * 2.5 + LEFT * 4
        self.B_coord = DOWN * 2.5 + RIGHT * 4
        self.dot_A = Dot(self.A_coord, color=YELLOW)
        self.dot_B = Dot(self.B_coord, color=YELLOW)
        label_A = MathTex("A").next_to(self.dot_A, UP)
        label_B = MathTex("B").next_to(self.dot_B, DOWN)

        self.play(
            Create(self.boundary),
            FadeIn(medium1, medium2),
            Write(props1), Write(props2),
            FadeIn(self.dot_A, label_A),
            FadeIn(self.dot_B, label_B)
        )
        self.wait(1)

        # Store key mobjects for later access
        self.add(self.boundary, self.dot_A, self.dot_B, label_A, label_B)

    def build_path_and_variables(self) -> None:
        """
        Constructs the light path and all associated geometric variables.
        Its contract is to overlay the geometric model onto the physical scene.
        """
        # Define the crossing point on the boundary, this is our variable.
        self.crossing_point_x = -1.0 
        crossing_coord = self.boundary.n2p(self.crossing_point_x) # n2p: number to point
        
        # Create the light ray segments
        self.ray1 = Line(self.A_coord, crossing_coord, color=PATH_COLOR, stroke_width=5)
        self.ray2 = Line(crossing_coord, self.B_coord, color=PATH_COLOR, stroke_width=5)

        # Create construction lines
        perp_A = DashedLine(self.A_coord, self.A_coord.real * RIGHT, color=WHITE)
        perp_B = DashedLine(self.B_coord, self.B_coord.real * RIGHT, color=WHITE)
        perp_crossing = DashedLine(crossing_coord + UP*3, crossing_coord + DOWN*3, color=WHITE, stroke_width=2)

        # Label distances
        h1_line = Brace(perp_A, direction=LEFT, buff=0.2)
        h2_line = Brace(perp_B, direction=LEFT, buff=0.2)
        h1_label = h1_line.get_tex("h_1")
        h2_label = h2_line.get_tex("h_2")

        x_line = Brace(Line(self.A_coord.real*RIGHT, crossing_coord), direction=DOWN, buff=0.2)
        dx_line = Brace(Line(crossing_coord, self.B_coord.real*RIGHT), direction=DOWN, buff=0.2)
        x_label = x_line.get_tex("x")
        dx_label = dx_line.get_tex("d-x")
        
        # Label angles
        self.angle1 = Angle(perp_crossing, self.ray1, radius=1.0, quadrant=(-1,-1), color=WHITE)
        self.angle2 = Angle(perp_crossing, self.ray2, radius=1.0, quadrant=(-1,1), color=WHITE)
        theta1_label = MathTex(r"\theta_1").next_to(self.angle1, RIGHT, buff=-0.2).shift(DOWN*0.5)
        theta2_label = MathTex(r"\theta_2").next_to(self.angle2, RIGHT, buff=-0.2).shift(UP*0.5)

        self.play(
            Create(self.ray1), Create(self.ray2),
            Create(perp_crossing)
        )
        self.play(
            Create(self.angle1), Write(theta1_label),
            Create(self.angle2), Write(theta2_label)
        )
        self.play(
            Create(perp_A), Create(perp_B),
            Create(h1_line), Write(h1_label),
            Create(h2_line), Write(h2_label)
        )
        self.play(
            Create(x_line), Write(x_label),
            Create(dx_line), Write(dx_label)
        )
        self.wait(2)
        
        # Group construction lines for easier fading later
        self.construction_group = VGroup(
            perp_A, perp_B, perp_crossing, h1_line, h2_line, h1_label, h2_label,
            x_line, dx_line, x_label, dx_label
        )

    def derive_time_equation(self) -> None:
        """
        Shows the step-by-step derivation of the total time equation T(x).
        Its contract is to formulate the objective function for minimization.
        """
        eq_pos = UP * 2.5 + RIGHT * 2
        
        # Step-by-step derivation of the time equation
        eq1 = MathTex(r"T(x) = t_1 + t_2").move_to(eq_pos)
        eq2 = MathTex(r"T(x) = \frac{L_1}{v_1} + \frac{L_2}{v_2}").move_to(eq_pos)
        eq3 = MathTex(r"T(x) = \frac{\sqrt{h_1^2 + x^2}}{v_1} + \frac{\sqrt{h_2^2 + (d-x)^2}}{v_2}").move_to(eq_pos)
        self.time_equation = eq3 # Save for later
        
        self.play(Write(eq1))
        self.wait(1)
        self.play(TransformMatchingTex(eq1, eq2))
        self.wait(1.5)
        self.play(TransformMatchingTex(eq2, eq3))
        self.wait(2)

    def perform_calculus(self) -> None:
        """
        States the minimization condition and shows the result of the differentiation.
        Its contract is to apply the core calculus principle.
        """
        # State the condition for minimum time
        condition = MathTex(r"\text{要使时间最短, 我们需要: } \frac{dT}{dx} = 0").next_to(self.time_equation, DOWN, buff=0.5)
        self.play(Write(condition))
        self.wait(1.5)

        # Show the result of the derivative
        derivative_eq = MathTex(
            r"\frac{dT}{dx} = \frac{1}{v_1} \frac{x}{\sqrt{h_1^2 + x^2}} - \frac{1}{v_2} \frac{d-x}{\sqrt{h_2^2 + (d-x)^2}} = 0"
        ).next_to(condition, DOWN, buff=0.5)
        self.derivative_equation = derivative_eq # Save for later

        self.play(Write(derivative_eq))
        self.wait(2)
        
        # Clean up the previous equations
        self.play(FadeOut(self.time_equation), FadeOut(condition))
        self.play(self.derivative_equation.animate.to_edge(UP))

    def connect_geometry_to_math(self) -> None:
        """
        Visually links the terms in the derivative equation to sin(theta) from the diagram.
        This is the "aha!" moment of the proof. Its contract is to bridge algebra and geometry.
        """
        # Isolate the sin(theta_1) part of the equation
        term1 = self.derivative_equation.get_part_by_tex(r"\frac{x}{\sqrt{h_1^2 + x^2}}")
        box1 = SurroundingRectangle(term1, color=ACCENT_COLOR)
        
        # Connect to geometry
        geo_term1 = MathTex(r"\sin\theta_1 = \frac{\text{对边}}{\text{斜边}} = \frac{x}{\sqrt{h_1^2 + x^2}}").to_edge(LEFT, buff=0.5).shift(UP)
        
        self.play(Create(box1))
        self.play(Indicate(self.angle1, color=ACCENT_COLOR))
        self.play(Write(geo_term1))
        self.wait(2)

        # Isolate the sin(theta_2) part
        term2 = self.derivative_equation.get_part_by_tex(r"\frac{d-x}{\sqrt{h_2^2 + (d-x)^2}}")
        box2 = SurroundingRectangle(term2, color=ACCENT_COLOR)
        
        # Connect to geometry
        geo_term2 = MathTex(r"\sin\theta_2 = \frac{\text{对边}}{\text{斜边}} = \frac{d-x}{\sqrt{h_2^2 + (d-x)^2}}").next_to(geo_term1, DOWN, buff=0.5, aligned_edge=LEFT)
        
        self.play(ReplacementTransform(box1, box2))
        self.play(Indicate(self.angle2, color=ACCENT_COLOR))
        self.play(Write(geo_term2))
        self.wait(2)

        self.play(FadeOut(box2, self.construction_group))
        self.geo_terms = VGroup(geo_term1, geo_term2)

    def show_final_result(self) -> None:
        """
        Substitutes the geometric terms back into the equation and simplifies to Snell's Law.
        Its contract is to present the final, conclusive result of the derivation.
        """
        # Substitute the sin terms back into the derivative equation
        sub_eq = MathTex(r"\frac{\sin\theta_1}{v_1} - \frac{\sin\theta_2}{v_2} = 0").move_to(self.derivative_equation.get_center())
        final_eq1 = MathTex(r"\frac{\sin\theta_1}{v_1} = \frac{\sin\theta_2}{v_2}").move_to(sub_eq.get_center())
        final_eq2 = MathTex(r"\frac{\sin\theta_1}{c/n_1} = \frac{\sin\theta_2}{c/n_2}").move_to(final_eq1.get_center())
        snell_law = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2").scale(1.5).move_to(ORIGIN)
        
        self.play(
            FadeOut(self.geo_terms),
            TransformMatchingTex(self.derivative_equation, sub_eq)
        )
        self.wait(1)
        self.play(TransformMatchingTex(sub_eq, final_eq1))
        self.wait(1)
        self.play(TransformMatchingTex(final_eq1, final_eq2))
        self.wait(1.5)
        
        # Clear the scene for the final result
        self.play(
            FadeOut(VGroup(*[m for m in self.mobjects if m is not snell_law]))
        )
        
        # The grand finale
        self.play(Write(snell_law))
        self.play(Create(SurroundingRectangle(snell_law, color=YELLOW, buff=0.2)))