from manim import *
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

class VariationalMethodsOptimization(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("变分法与优化", font_size=36, color=BLUE)
        subtitle = Text("Variational Methods and Optimization in Optical Design", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Basic Variational Problem
        part1_title = Text("1. 变分问题基本形式", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        variational_problem = MathTex(
            r"\min J[y] = \int_a^b F(x, y, y') dx",
            font_size=32, color=BLACK
        )
        variational_problem.next_to(part1_title, DOWN, buff=0.5)
        
        boundary_conditions = MathTex(
            r"y(a) = y_a, \quad y(b) = y_b",
            font_size=28, color=BLACK
        )
        boundary_conditions.next_to(variational_problem, DOWN, buff=0.3)
        
        self.play(FadeIn(part1_title))
        self.play(Write(variational_problem))
        self.play(Write(boundary_conditions))
        self.wait(3)
        
        # Part 2: Euler-Lagrange Equation
        part2_title = Text("2. 欧拉-拉格朗日方程", font_size=28, color=GREEN)
        part2_title.next_to(boundary_conditions, DOWN, buff=0.8)
        
        euler_lagrange = MathTex(
            r"\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0",
            font_size=32, color=BLACK
        )
        euler_lagrange.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(euler_lagrange))
        self.wait(3)
        
        # Part 3: Optical Application - Light Path
        part3_title = Text("3. 光学应用：光线最短路径", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        optical_functional = MathTex(
            r"J[y] = \int n(x,y) \sqrt{1 + (y')^2} dx",
            font_size=28, color=BLACK
        )
        optical_functional.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(variational_problem), 
                 FadeOut(boundary_conditions), FadeOut(part2_title), FadeOut(euler_lagrange))
        self.play(FadeIn(part3_title))
        self.play(Write(optical_functional))
        self.wait(2)
        
        # Create visualization of light path optimization
        axes = Axes(
            x_range=[0, 4, 0.5],
            y_range=[-1, 2, 0.3],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(optical_functional, DOWN, buff=0.5)
        
        # Define refractive index profile
        def refractive_index(x, y):
            return 1.0 + 0.3 * np.exp(-((x-2)**2 + y**2))
        
        # Straight line path (initial guess)
        def straight_path(x):
            return 0.5 * x - 0.5
        
        # Optimized path (solution to variational problem)
        def optimized_path(x):
            # Approximate solution for gradient index medium
            return -0.2 * (x-2)**2 + 0.8
        
        # Plot paths
        straight_line = axes.plot(straight_path, color=RED, stroke_width=3)
        optimized_line = axes.plot(optimized_path, color=BLUE, stroke_width=3)
        
        straight_label = MathTex(r"\text{直线路径}", font_size=20, color=RED)
        optimized_label = MathTex(r"\text{优化路径}", font_size=20, color=BLUE)
        
        straight_label.next_to(axes, RIGHT, buff=0.3).shift(UP*0.5)
        optimized_label.next_to(straight_label, DOWN, buff=0.2)
        
        self.play(Create(axes))
        self.play(Create(straight_line), Write(straight_label))
        self.wait(2)
        self.play(Create(optimized_line), Write(optimized_label))
        self.wait(3)
        
        # Part 4: Numerical Optimization
        part4_title = Text("4. 数值优化方法", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        numerical_methods = VGroup(
            MathTex(r"\text{梯度下降：} y_{n+1} = y_n - \alpha \nabla J[y_n]", font_size=24, color=BLACK),
            MathTex(r"\text{牛顿法：} y_{n+1} = y_n - [H J[y_n]]^{-1} \nabla J[y_n]", font_size=24, color=BLACK),
            MathTex(r"\text{共轭梯度法：} \text{正交搜索方向}", font_size=24, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        numerical_methods.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        for method in numerical_methods:
            self.play(Write(method))
            self.wait(1)
        self.wait(3)
        
        # Part 5: Constrained Optimization
        part5_title = Text("5. 约束优化", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        constrained_problem = MathTex(
            r"\min J[y] \quad \text{s.t.} \quad g_i(y) = 0, \quad h_j(y) \leq 0",
            font_size=28, color=BLACK
        )
        constrained_problem.next_to(part5_title, UP, buff=0.3)
        
        lagrange_multipliers = MathTex(
            r"\mathcal{L}[y, \lambda, \mu] = J[y] + \sum \lambda_i g_i(y) + \sum \mu_j h_j(y)",
            font_size=24, color=BLACK
        )
        lagrange_multipliers.next_to(constrained_problem, DOWN, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(constrained_problem))
        self.play(Write(lagrange_multipliers))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class OpticalSystemOptimization(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("光学系统优化实例", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D visualization of lens optimization
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-1, 1, 0.2],
            x_length=8,
            y_length=8,
            z_length=4
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Initial lens surface (spherical)
        def initial_lens(x, y):
            R = 2.0  # Radius of curvature
            return (x**2 + y**2) / (2 * R)
        
        # Create initial lens surface
        initial_surface = Surface(
            lambda u, v: axes.c2p(u, v, initial_lens(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=30,
            fill_opacity=0.7,
            fill_color=BLUE
        )
        
        initial_label = Text("初始球面透镜", font_size=20, color=BLUE)
        initial_label.next_to(axes, LEFT)
        
        self.play(Create(initial_surface), Write(initial_label))
        self.wait(2)
        
        # Optimized lens surface (aspheric)
        def optimized_lens(x, y):
            # Aspheric surface with optimized coefficients
            r2 = x**2 + y**2
            return r2/(2*2.0) - 0.01*r2**2 + 0.001*r2**3
        
        # Create optimized surface
        optimized_surface = Surface(
            lambda u, v: axes.c2p(u, v, optimized_lens(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=30,
            fill_opacity=0.7,
            fill_color=RED
        )
        
        optimized_label = Text("优化非球面透镜", font_size=20, color=RED)
        optimized_label.next_to(axes, RIGHT)
        
        # Transform to optimized surface
        self.play(Transform(initial_surface, optimized_surface), 
                 Transform(initial_label, optimized_label))
        self.wait(3)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class ConvergenceVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("优化算法收敛性", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for convergence plot
        axes = Axes(
            x_range=[0, 20, 2],
            y_range=[-3, 1, 0.5],
            x_length=8,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # Different convergence behaviors
        def linear_convergence(x):
            return -x * 0.1
        
        def quadratic_convergence(x):
            return -0.5 * (x**2) * 0.01
        
        def exponential_convergence(x):
            return -np.exp(0.2 * x) * 0.1
        
        # Plot convergence curves
        linear_plot = axes.plot(linear_convergence, color=BLUE, stroke_width=3)
        quadratic_plot = axes.plot(quadratic_convergence, color=RED, stroke_width=3)
        exponential_plot = axes.plot(exponential_convergence, color=GREEN, stroke_width=3)
        
        # Labels
        linear_label = MathTex(r"\text{线性收敛}", font_size=20, color=BLUE)
        quadratic_label = MathTex(r"\text{二次收敛}", font_size=20, color=RED)
        exponential_label = MathTex(r"\text{指数收敛}", font_size=20, color=GREEN)
        
        labels = VGroup(linear_label, quadratic_label, exponential_label)
        labels.arrange(DOWN, buff=0.2)
        labels.next_to(axes, RIGHT, buff=0.3)
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Animate convergence curves
        self.play(Create(linear_plot), Write(linear_label))
        self.wait(1)
        self.play(Create(quadratic_plot), Write(quadratic_label))
        self.wait(1)
        self.play(Create(exponential_plot), Write(exponential_label))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))