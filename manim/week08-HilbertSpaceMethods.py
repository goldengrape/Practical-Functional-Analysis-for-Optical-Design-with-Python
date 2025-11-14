from manim import *
import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt

class HilbertSpaceMethods(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("希尔伯特空间方法", font_size=36, color=BLUE)
        subtitle = Text("Hilbert Space Methods in Optical Analysis", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Hilbert Space Definition
        part1_title = Text("1. 希尔伯特空间定义", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        hilbert_def = VGroup(
            MathTex(r"\text{完备内积空间}", font_size=32, color=BLACK),
            MathTex(r"\langle f, g \rangle = \int f^*(x) g(x) dx", font_size=28, color=BLACK),
            MathTex(r"\|f\|^2 = \langle f, f \rangle", font_size=28, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        hilbert_def.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        for eq in hilbert_def:
            self.play(Write(eq))
            self.wait(1)
        self.wait(2)
        
        # Part 2: Orthogonal Projection
        part2_title = Text("2. 正交投影", font_size=28, color=GREEN)
        part2_title.next_to(hilbert_def, DOWN, buff=0.8)
        
        projection_formula = MathTex(
            r"P_V f = \sum_{i=1}^n \langle e_i, f \rangle e_i",
            font_size=32, color=BLACK
        )
        projection_formula.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(projection_formula))
        self.wait(3)
        
        # Part 3: Orthonormal Basis
        part3_title = Text("3. 标准正交基", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        orthonormal_conditions = VGroup(
            MathTex(r"\langle e_i, e_j \rangle = \delta_{ij}", font_size=28, color=BLACK),
            MathTex(r"\|e_i\| = 1", font_size=28, color=BLACK),
            MathTex(r"\text{span}\{e_i\} \text{ 稠密}}", font_size=28, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        orthonormal_conditions.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(hilbert_def), 
                 FadeOut(part2_title), FadeOut(projection_formula))
        self.play(FadeIn(part3_title))
        for condition in orthonormal_conditions:
            self.play(Write(condition))
            self.wait(1)
        self.wait(2)
        
        # Part 4: Parseval's Identity
        part4_title = Text("4. 帕塞瓦尔恒等式", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        parseval_identity = MathTex(
            r"\|f\|^2 = \sum_{i=1}^{\infty} |\langle e_i, f \rangle|^2",
            font_size=32, color=BLACK
        )
        parseval_identity.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        self.play(Write(parseval_identity))
        self.wait(3)
        
        # Part 5: Riesz Representation Theorem
        part5_title = Text("5. 里斯表示定理", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        riesz_theorem = MathTex(
            r"\forall L \in H^*, \exists! g \in H: L(f) = \langle g, f \rangle",
            font_size=28, color=BLACK
        )
        riesz_theorem.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(riesz_theorem))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class OrthogonalProjectionDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("正交投影演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 2D coordinate system
        axes = Axes(
            x_range=[-3, 3, 0.5],
            y_range=[-3, 3, 0.5],
            x_length=6,
            y_length=6,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Define a function and its projection
        def original_function(x):
            return 0.5 * x**2 - 0.5 * x + 1
        
        def projection_component(x):
            # Projection onto linear subspace
            return 0.8 * x + 0.5
        
        def orthogonal_component(x):
            return original_function(x) - projection_component(x)
        
        # Plot original function
        original_plot = axes.plot(original_function, color=BLUE, stroke_width=3)
        original_label = MathTex(r"f(x)", font_size=24, color=BLUE)
        original_label.next_to(original_plot, RIGHT, buff=0.2)
        
        self.play(Create(original_plot), Write(original_label))
        self.wait(2)
        
        # Plot projection
        projection_plot = axes.plot(projection_component, color=RED, stroke_width=3)
        projection_label = MathTex(r"P_V f(x)", font_size=24, color=RED)
        projection_label.next_to(projection_plot, LEFT, buff=0.2)
        
        self.play(Create(projection_plot), Write(projection_label))
        self.wait(2)
        
        # Plot orthogonal component
        orthogonal_plot = axes.plot(orthogonal_component, color=GREEN, stroke_width=2, dashed=True)
        orthogonal_label = MathTex(r"f(x) - P_V f(x)", font_size=24, color=GREEN)
        orthogonal_label.next_to(orthogonal_plot, DOWN, buff=0.2)
        
        self.play(Create(orthogonal_plot), Write(orthogonal_label))
        self.wait(3)
        
        # Show projection error
        error_formula = MathTex(
            r"\|f - P_V f\|^2 = \min_{g \in V} \|f - g\|^2",
            font_size=24, color=BLACK
        )
        error_formula.to_edge(DOWN)
        
        self.play(Write(error_formula))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class FourierBasisDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("傅里叶基函数演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Fourier basis functions
        def fourier_cosine(n, x):
            return np.cos(n * np.pi * x)
        
        def fourier_sine(n, x):
            return np.sin(n * np.pi * x)
        
        # Plot several basis functions
        colors = [BLUE, RED, GREEN, ORANGE]
        
        for n in range(1, 5):
            color = colors[n-1]
            
            # Cosine function
            cos_plot = axes.plot(lambda x: fourier_cosine(n, x), color=color, stroke_width=2)
            cos_label = MathTex(f"\\cos({n}\\pi x)", font_size=20, color=color)
            cos_label.next_to(cos_plot, UP, buff=0.1)
            
            self.play(Create(cos_plot), Write(cos_label))
            self.wait(1)
            
            if n <= 2:  # Only show first few to avoid clutter
                # Sine function
                sin_plot = axes.plot(lambda x: fourier_sine(n, x), color=color, stroke_width=2, dashed=True)
                sin_label = MathTex(f"\\sin({n}\\pi x)", font_size=20, color=color)
                sin_label.next_to(sin_plot, DOWN, buff=0.1)
                
                self.play(Create(sin_plot), Write(sin_label))
                self.wait(1)
        
        # Show orthogonality
        orthogonality = MathTex(
            r"\int_{-1}^1 \cos(m\\pi x) \cos(n\\pi x) dx = \delta_{mn}",
            font_size=24, color=BLACK
        )
        orthogonality.to_edge(DOWN)
        
        self.play(Write(orthogonality))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class HilbertSpaceOperations(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("希尔伯特空间运算", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D visualization of function space
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
        
        # Define two functions
        def function_f(x, y):
            return np.exp(-(x**2 + y**2)) * np.sin(3*x)
        
        def function_g(x, y):
            return np.exp(-(x**2 + y**2)) * np.cos(3*y)
        
        # Create surfaces for functions
        f_surface = Surface(
            lambda u, v: axes.c2p(u, v, function_f(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=25,
            fill_opacity=0.7,
            fill_color=BLUE
        )
        
        g_surface = Surface(
            lambda u, v: axes.c2p(u, v, function_g(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=25,
            fill_opacity=0.7,
            fill_color=RED
        )
        
        # Labels
        f_label = MathTex(r"f(x,y)", font_size=24, color=BLUE)
        g_label = MathTex(r"g(x,y)", font_size=24, color=RED)
        
        f_label.next_to(f_surface, LEFT)
        g_label.next_to(g_surface, RIGHT)
        
        # Show individual functions
        self.play(Create(f_surface), Write(f_label))
        self.wait(2)
        self.play(Create(g_surface), Write(g_label))
        self.wait(3)
        
        # Show inner product visualization
        inner_product_text = MathTex(
            r"\langle f, g \rangle = \int f^*(x,y) g(x,y) dx dy",
            font_size=24, color=BLACK
        )
        inner_product_text.to_edge(DOWN)
        
        self.play(Write(inner_product_text))
        self.wait(3)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class GramSchmidtProcess(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("Gram-Schmidt 正交化过程", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Show the Gram-Schmidt algorithm steps
        steps = VGroup(
            MathTex(r"\text{1. } e_1 = \frac{v_1}{\|v_1\|}", font_size=24, color=BLACK),
            MathTex(r"\text{2. } u_2 = v_2 - \langle e_1, v_2 \rangle e_1", font_size=24, color=BLACK),
            MathTex(r"\text{3. } e_2 = \frac{u_2}{\|u_2\|}", font_size=24, color=BLACK),
            MathTex(r"\text{4. } u_k = v_k - \sum_{j=1}^{k-1} \langle e_j, v_k \rangle e_j", font_size=24, color=BLACK),
            MathTex(r"\text{5. } e_k = \frac{u_k}{\|u_k\|}", font_size=24, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        steps.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        
        for step in steps:
            self.play(Write(step))
            self.wait(1)
        
        # Show geometric interpretation
        geometric_intuition = Text(
            "几何意义：逐步减去投影分量，确保正交性",
            font_size=20, color=GREEN
        )
        geometric_intuition.next_to(steps, DOWN, buff=0.5)
        
        self.play(Write(geometric_intuition))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))