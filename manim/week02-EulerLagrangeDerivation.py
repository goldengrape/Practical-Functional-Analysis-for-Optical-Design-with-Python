# manim -pql scene.py EulerLagrangeDerivation

from manim import *
import numpy as np

# Set the global theme for the animation
config.background_color = "#1E1E1E" # A dark, modern background
TEXT_COLOR = WHITE
ACCENT_COLOR = YELLOW
PATH_COLOR = BLUE
PERTURBATION_COLOR = GREEN
PERTURBED_PATH_COLOR = ORANGE

class EulerLagrangeDerivation(Scene):
    """
    An animation demonstrating the derivation of the Euler-Lagrange equation
    using the path perturbation method, as described in the course material.
    The animation is structured to follow the logical steps of the derivation:
    1.  Introduce the problem: finding an optimal path y(x).
    2.  Define a perturbed path y(x, epsilon) using a variation eta(x).
    3.  Show that this turns the functional I[y] into a function I(epsilon).
    4.  Apply the minimum condition from calculus: dI/d_epsilon = 0 at epsilon = 0.
    5.  Visually step through the mathematical derivation to arrive at the E-L equation.
    """
    def construct(self):
        self.show_title()
        self.setup_problem()
        self.introduce_perturbation()
        self.functional_to_function()
        self.apply_calculus_condition()
        self.mathematical_derivation()
        self.show_conclusion()

    def show_title(self):
        """Displays the main title of the animation."""
        title = Text("变分法核心：欧拉-拉格朗日方程的推导", color=TEXT_COLOR, font_size=40)
        subtitle = Text("Intuitive Derivation of the Euler-Lagrange Equation", color=ACCENT_COLOR, font_size=24)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.4)
        self.play(Write(title_group))
        self.wait(2)
        self.play(FadeOut(title_group))

    def setup_problem(self):
        """Sets up the initial problem: finding the path y(x) that minimizes a functional."""
        # Create axes and labels
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            axis_config={"color": BLUE},
            x_length=10,
            y_length=6
        ).add_coordinates()
        
        axes_labels = self.axes.get_axis_labels(x_label="x", y_label="y")

        # Define start and end points
        self.start_point = self.axes.c2p(1, 2)
        self.end_point = self.axes.c2p(9, 4)
        dot_a = Dot(self.start_point, color=ACCENT_COLOR).set_z_index(1)
        dot_b = Dot(self.end_point, color=ACCENT_COLOR).set_z_index(1)
        label_a = MathTex("A", color=TEXT_COLOR).next_to(dot_a, DOWN)
        label_b = MathTex("B", color=TEXT_COLOR).next_to(dot_b, UP)
        
        # Define the optimal path y(x)
        self.optimal_path = self.axes.plot(
            lambda x: 0.025 * (x - 5)**2 + 1.75, x_range=[1, 9], color=PATH_COLOR
        )
        path_label = MathTex("y(x)", color=PATH_COLOR).next_to(self.optimal_path, UP, buff=0.3)
        
        # Define the functional to be minimized
        functional_text = MathTex(
            r"I[y] = \int_a^b F(x, y, y') dx",
            color=TEXT_COLOR
        ).to_corner(UL)
        
        # Animate the setup
        self.play(
            Create(self.axes),
            Create(axes_labels),
            Write(functional_text)
        )
        self.play(
            FadeIn(dot_a, scale=0.5), Write(label_a),
            FadeIn(dot_b, scale=0.5), Write(label_b)
        )
        self.play(Create(self.optimal_path), Write(path_label))
        self.wait(1)

        # Store objects for later use
        self.add(functional_text, dot_a, dot_b, label_a, label_b, path_label)
        self.path_label = path_label

    def introduce_perturbation(self):
        """Introduces the perturbation function eta(x) and the perturbed path."""
        # 1. Define the perturbation function eta(x)
        self.eta_path = self.axes.plot(
            lambda x: 0.5 * np.sin(PI * (x - 1) / 4), x_range=[1, 9], color=PERTURBATION_COLOR
        )
        eta_label = MathTex(r"\eta(x)", color=PERTURBATION_COLOR).next_to(self.eta_path, UP, buff=0.3)
        
        # Show that eta is zero at the boundaries
        boundary_conditions = MathTex(
            r"\eta(a) = 0, \eta(b) = 0",
            color=PERTURBATION_COLOR
        ).next_to(functional_text, DOWN, aligned_edge=LEFT)
        
        self.play(Create(self.eta_path), Write(eta_label))
        self.play(Write(boundary_conditions))
        self.wait(1)
        
        # 2. Define the perturbed path
        epsilon = ValueTracker(0.5) # The perturbation amount
        self.perturbed_path = always_redraw(
            lambda: self.axes.plot(
                lambda x: (0.025 * (x - 5)**2 + 1.75) + epsilon.get_value() * (0.5 * np.sin(PI * (x - 1) / 4)),
                x_range=[1, 9],
                color=PERTURBED_PATH_COLOR
            )
        )
        
        perturbed_path_eq = MathTex(
            r"y(x, \epsilon) = y(x) + \epsilon\eta(x)",
            color=PERTURBED_PATH_COLOR
        ).next_to(boundary_conditions, DOWN, aligned_edge=LEFT)
        
        self.play(
            FadeOut(eta_label),
            Transform(self.eta_path, self.perturbed_path),
            Write(perturbed_path_eq)
        )
        self.add(self.perturbed_path)
        self.play(self.eta_path.animate.set_opacity(0)) # Hide the original eta path
        self.wait(1)
        
        # Animate the effect of epsilon
        self.play(epsilon.animate.set_value(0.0), run_time=1.5)
        self.play(epsilon.animate.set_value(-0.7), run_time=1.5)
        self.play(epsilon.animate.set_value(0.0), run_time=1.5)
        self.wait(1)

        # Store objects
        self.add(perturbed_path_eq)

    def functional_to_function(self):
        """Shows how the functional becomes a simple function of epsilon."""
        # Transform the functional text
        functional_of_epsilon = MathTex(
            r"I(\epsilon) = \int_a^b F(x, y(x, \epsilon), y'(x, \epsilon)) dx",
            color=TEXT_COLOR
        ).to_corner(UL)
        
        self.play(Transform(self.find_mobject(r"I[y]"), functional_of_epsilon))
        self.wait(2)
        
        # Create a new graph for I(epsilon) vs epsilon
        self.epsilon_axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": BLUE}
        ).to_corner(DR)
        
        epsilon_labels = self.epsilon_axes.get_axis_labels(x_label=r"\epsilon", y_label=r"I(\epsilon)")
        
        # This curve represents I(epsilon), with a minimum at epsilon=0
        i_of_epsilon_curve = self.epsilon_axes.plot(lambda e: 2 * e**2 + 1, x_range=[-1, 1], color=ACCENT_COLOR)
        
        self.play(Create(self.epsilon_axes), Write(epsilon_labels))
        self.play(Create(i_of_epsilon_curve))
        self.wait(1)

    def apply_calculus_condition(self):
        """Applies the condition for a minimum from standard calculus."""
        # Highlight the minimum at epsilon = 0
        min_point = Dot(self.epsilon_axes.c2p(0, 1), color=RED)
        min_text = MathTex(r"\epsilon=0", color=RED).next_to(min_point, UP)
        
        # Draw the tangent line
        tangent = self.epsilon_axes.get_horizontal_line(self.epsilon_axes.c2p(0, 1), color=RED)
        
        # State the condition
        condition = MathTex(
            r"\left. \frac{dI}{d\epsilon} \right|_{\epsilon=0} = 0",
            color=ACCENT_COLOR,
            font_size=48
        ).move_to(self.epsilon_axes.get_center() + UP*2)
        
        self.play(Create(min_point), Write(min_text))
        self.play(Create(tangent))
        self.play(Write(condition))
        self.wait(2)

        # Clean up the scene for the main derivation
        self.play(
            FadeOut(self.axes, self.optimal_path, self.perturbed_path, self.path_label),
            FadeOut(self.find_mobject("A"), self.find_mobject("B")),
            FadeOut(self.find_mobject(r"\eta(a)"), self.find_mobject(r"y(x, \epsilon)")),
            FadeOut(self.epsilon_axes, self.find_mobject(r"\epsilon"), self.find_mobject(r"I(\epsilon)")),
            FadeOut(min_point, min_text, tangent),
            # Move the key equations to the top
            VGroup(
                self.find_mobject(r"I(\epsilon)"),
                condition
            ).animate.arrange(DOWN, buff=0.5).to_edge(UP)
        )
        self.wait(1)

    def mathematical_derivation(self):
        """Steps through the mathematical derivation of the E-L equation."""
        # Define equations using Tex to handle alignment and complex fractions
        deriv_steps = VGroup(
            # Step 1: Differentiate under the integral sign
            Tex(r"$\frac{dI}{d\epsilon} = \int_a^b \left( \frac{\partial F}{\partial y} \frac{\partial y}{\partial \epsilon} + \frac{\partial F}{\partial y'} \frac{\partial y'}{\partial \epsilon} \right) dx$"),
            # Step 2: Substitute partials
            Tex(r"$\frac{dI}{d\epsilon} = \int_a^b \left( \frac{\partial F}{\partial y} \eta(x) + \frac{\partial F}{\partial y'} \eta'(x) \right) dx$"),
            # Step 3: Set epsilon to 0 (implicit) and set to 0
            Tex(r"$\int_a^b \left( \frac{\partial F}{\partial y} \eta(x) + \frac{\partial F}{\partial y'} \eta'(x) \right) dx = 0$"),
            # Step 4: Integration by Parts on the second term
            Tex(r"$\int_a^b \frac{\partial F}{\partial y'} \eta'(x) dx = \left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b - \int_a^b \eta(x) \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) dx$"),
            # Step 5: Boundary term vanishes
            Tex(r"$\left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b = 0 \quad (\text{since } \eta(a)=\eta(b)=0)$"),
            # Step 6: Substitute back and combine
            Tex(r"$\int_a^b \left( \frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) \right) \eta(x) dx = 0$"),
            # Step 7: Fundamental Lemma of Calculus of Variations
            Tex(r"$\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0$")
        ).scale(0.8).arrange(DOWN, buff=0.5, aligned_edge=LEFT).next_to(self.find_mobject(r"\left."), DOWN, buff=0.5, aligned_edge=LEFT)
        
        # Animate each step of the derivation
        self.play(Write(deriv_steps[0]))
        self.wait(1)
        self.play(TransformMatchingTex(deriv_steps[0].copy(), deriv_steps[1]))
        self.wait(1)
        self.play(TransformMatchingTex(deriv_steps[1].copy(), deriv_steps[2]))
        self.wait(2)
        
        # Integration by parts explanation
        ibp_title = Text("分部积分 (Integration by Parts)", color=ACCENT_COLOR, font_size=28).next_to(deriv_steps[2], DOWN, buff=1)
        self.play(Write(ibp_title))
        self.play(Write(deriv_steps[3]))
        self.wait(2)
        self.play(Write(deriv_steps[4]))
        self.wait(2)
        
        # Fade out explanation and show the combined result
        self.play(FadeOut(ibp_title), FadeOut(deriv_steps[3]), FadeOut(deriv_steps[4]))
        self.play(Write(deriv_steps[5]))
        self.wait(2)
        
        # Fundamental Lemma
        lemma_text = Text("根据变分法基本引理...", color=ACCENT_COLOR, font_size=28).next_to(deriv_steps[5], DOWN, buff=1)
        self.play(Write(lemma_text))
        self.wait(1)
        
        # The final equation
        final_equation = deriv_steps[6]
        box = SurroundingRectangle(final_equation, color=YELLOW, buff=0.1)
        self.play(TransformMatchingTex(deriv_steps[5].copy(), final_equation))
        self.play(Create(box))
        self.wait(3)

        # Store for the conclusion
        self.final_group = VGroup(final_equation, box, lemma_text)

    def show_conclusion(self):
        """Shows the conclusion and what has been achieved."""
        self.play(
            FadeOut(self.find_mobject(r"I(\epsilon)")),
            FadeOut(self.find_mobject(r"\left.")),
            FadeOut(self.find_mobject(r"\int_a^b"))
        )

        conclusion_text = VGroup(
            Text("泛函最小化问题", font_size=32),
            Tex(r"$\min_{y(x)} I[y]$", font_size=36),
            Text("转化为", font_size=32),
            Text("求解一个微分方程", font_size=32),
        ).arrange(DOWN, buff=0.4).to_edge(LEFT, buff=1)

        arrow = Arrow(conclusion_text.get_right(), self.final_group.get_left(), buff=0.5, color=WHITE)

        self.play(
            self.final_group.animate.to_edge(RIGHT, buff=1),
            Write(conclusion_text)
        )
        self.play(GrowArrow(arrow))
        self.wait(4)

    def find_mobject(self, tex_string):
        """Helper to find a mobject on screen from its LaTeX string."""
        for mobj in self.mobjects:
            if isinstance(mobj, (MathTex, Tex)):
                try:
                    # Check if the desired TeX string is a substring of the mobject's TeX string
                    if tex_string in mobj.tex_string:
                        return mobj
                except:
                    pass
        return None