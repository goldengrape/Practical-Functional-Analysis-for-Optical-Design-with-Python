from manim import *
import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt

class IntegralEquationsMultilayer(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("多层光学设计中的积分方程", font_size=36, color=BLUE)
        subtitle = Text("Integral Equations in Multilayer Optical Design", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Basic Integral Equation Form
        part1_title = Text("1. 多层膜积分方程", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        integral_eq = MathTex(
            r"E(z) = E_0(z) + \int_0^L K(z,z') \epsilon(z') E(z') dz'",
            font_size=32, color=BLACK
        )
        integral_eq.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        self.play(Write(integral_eq))
        self.wait(3)
        
        # Part 2: Kernel Function Explanation
        part2_title = Text("2. 核函数 (Kernel Function)", font_size=28, color=GREEN)
        part2_title.next_to(integral_eq, DOWN, buff=0.8)
        
        kernel_def = MathTex(
            r"K(z,z') = \frac{\omega^2}{c^2} G(z,z') \chi(z')",
            font_size=28, color=BLACK
        )
        kernel_def.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(kernel_def))
        self.wait(3)
        
        # Part 3: Green's Function
        part3_title = Text("3. 格林函数", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        greens_function = MathTex(
            r"G(z,z') = \frac{e^{ik|z-z'|}}{4\pi|z-z'|}",
            font_size=28, color=BLACK
        )
        greens_function.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(integral_eq), 
                 FadeOut(part2_title), FadeOut(kernel_def))
        self.play(FadeIn(part3_title))
        self.play(Write(greens_function))
        self.wait(2)
        
        # Create visualization of Green's function
        axes = Axes(
            x_range=[-3, 3, 0.5],
            y_range=[0, 2, 0.2],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(greens_function, DOWN, buff=0.5)
        
        # Plot Green's function
        def greens_func(x, z_prime=0, k=2):
            return np.exp(-k*np.abs(x - z_prime)) / (4*np.pi*np.abs(x - z_prime) + 0.1)
        
        greens_plot = axes.plot(lambda x: greens_func(x, z_prime=0, k=2), 
                               color=BLUE, stroke_width=3)
        
        greens_label = MathTex(r"G(z,0)", font_size=24, color=BLUE)
        greens_label.next_to(axes, RIGHT, buff=0.3)
        
        self.play(Create(axes))
        self.play(Create(greens_plot), Write(greens_label))
        self.wait(3)
        
        # Part 4: Discretization
        part4_title = Text("4. 离散化方法", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        discretization = MathTex(
            r"E_i = E_{0,i} + \sum_j K_{ij} \epsilon_j E_j \Delta z",
            font_size=28, color=BLACK
        )
        discretization.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        self.play(Write(discretization))
        self.wait(3)
        
        # Part 5: Matrix Form
        part5_title = Text("5. 矩阵形式", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        matrix_form = MathTex(
            r"(I - K\epsilon\Delta z) E = E_0",
            font_size=32, color=BLACK
        )
        matrix_form.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(matrix_form))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class MultilayerStructureVisualization(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("多层膜结构可视化", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D visualization of multilayer structure
        layers = []
        layer_colors = [BLUE, GREEN, RED, YELLOW, PURPLE]
        layer_thickness = [0.5, 0.3, 0.7, 0.4, 0.6]
        
        current_z = 0
        for i, (thickness, color) in enumerate(zip(layer_thickness, layer_colors)):
            layer = Prism(
                dimensions=[4, 4, thickness],
                fill_opacity=0.7,
                fill_color=color,
                stroke_width=2
            )
            layer.move_to([0, 0, current_z + thickness/2])
            layers.append(layer)
            current_z += thickness
        
        # Create axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[0, 3, 0.5],
            x_length=6,
            y_length=6,
            z_length=6
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Animate layer creation
        for i, layer in enumerate(layers):
            layer_label = Text(f"层 {i+1}", font_size=20, color=WHITE)
            layer_label.move_to(layer.get_center())
            
            self.play(Create(layer), Write(layer_label))
            self.wait(0.5)
        
        self.wait(2)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class IntegralEquationSolution(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("积分方程求解方法", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Part 1: Neumann Series
        part1_title = Text("1. 诺依曼级数法", font_size=28, color=GREEN)
        part1_title.next_to(title, DOWN, buff=0.5)
        
        neumann_series = MathTex(
            r"E = E_0 + KE_0 + K^2E_0 + K^3E_0 + \cdots",
            font_size=28, color=BLACK
        )
        neumann_series.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(part1_title))
        self.play(Write(neumann_series))
        self.wait(3)
        
        # Part 2: Iterative Solution
        part2_title = Text("2. 迭代求解", font_size=28, color=GREEN)
        part2_title.next_to(neumann_series, DOWN, buff=0.8)
        
        iterative_eq = MathTex(
            r"E^{(n+1)} = E_0 + K E^{(n)}",
            font_size=28, color=BLACK
        )
        iterative_eq.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(iterative_eq))
        self.wait(3)
        
        # Part 3: Convergence
        part3_title = Text("3. 收敛条件", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        convergence_cond = MathTex(
            r"\|K\| < 1",
            font_size=32, color=BLACK
        )
        convergence_cond.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(neumann_series), 
                 FadeOut(part2_title), FadeOut(iterative_eq))
        self.play(FadeIn(part3_title))
        self.play(Write(convergence_cond))
        self.wait(3)
        
        # Part 4: Numerical Solution
        part4_title = Text("4. 数值求解", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        numerical_method = MathTex(
            r"E = (I - K)^{-1} E_0",
            font_size=32, color=BLACK
        )
        numerical_method.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        self.play(Write(numerical_method))
        self.wait(3)
        
        # Create visualization of convergence
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 2, 0.2],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(convergence_cond, DOWN, buff=0.8)
        
        # Plot convergence behavior
        def convergence_error(n):
            return 0.8**n  # Exponential convergence
        
        convergence_plot = axes.plot(convergence_error, color=GREEN, stroke_width=3)
        
        convergence_label = MathTex(r"\|E^{(n)} - E^*\|", font_size=24, color=GREEN)
        convergence_label.next_to(axes, RIGHT, buff=0.3)
        
        self.play(Create(axes))
        self.play(Create(convergence_plot), Write(convergence_label))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))