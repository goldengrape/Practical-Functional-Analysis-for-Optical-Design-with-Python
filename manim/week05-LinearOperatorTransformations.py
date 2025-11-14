from manim import *
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

class LinearOperatorTransformations(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("线性算子变换", font_size=36, color=BLUE)
        subtitle = Text("Linear Operator Transformations in Optical Systems", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Basic Linear Operator Definition
        part1_title = Text("1. 线性算子定义", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        operator_def = MathTex(
            r"L: V \rightarrow W, \quad L(af + bg) = aL(f) + bL(g)",
            font_size=32, color=BLACK
        )
        operator_def.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        self.play(Write(operator_def))
        self.wait(3)
        
        # Part 2: Optical System Example - Fourier Transform
        part2_title = Text("2. 光学傅里叶变换", font_size=28, color=GREEN)
        part2_title.next_to(operator_def, DOWN, buff=0.8)
        
        fourier_transform = MathTex(
            r"\mathcal{F}\{f(x)\}(k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} dx",
            font_size=28, color=BLACK
        )
        fourier_transform.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(fourier_transform))
        self.wait(3)
        
        # Part 3: Eigenfunction Analysis
        part3_title = Text("3. 特征函数分析", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        eigenfunction_eq = MathTex(
            r"L\phi_n = \lambda_n \phi_n",
            font_size=32, color=BLACK
        )
        eigenfunction_eq.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(operator_def), 
                 FadeOut(part2_title), FadeOut(fourier_transform))
        self.play(FadeIn(part3_title))
        self.play(Write(eigenfunction_eq))
        self.wait(2)
        
        # Create visualization of eigenfunctions
        axes = Axes(
            x_range=[-3, 3, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(eigenfunction_eq, DOWN, buff=0.5)
        
        # Define eigenfunctions for a simple operator (d/dx)
        def eigenfunction_1(x):
            return np.exp(x)  # eigenfunction of d/dx with eigenvalue 1
        
        def eigenfunction_2(x):
            return np.exp(2*x)  # eigenfunction of d/dx with eigenvalue 2
        
        # Plot eigenfunctions
        eigen1 = axes.plot(eigenfunction_1, color=BLUE, stroke_width=3)
        eigen2 = axes.plot(eigenfunction_2, color=RED, stroke_width=3)
        
        eigen1_label = MathTex(r"\phi_1(x) = e^x, \lambda_1 = 1", font_size=24, color=BLUE)
        eigen2_label = MathTex(r"\phi_2(x) = e^{2x}, \lambda_2 = 2", font_size=24, color=RED)
        
        eigen1_label.next_to(axes, RIGHT, buff=0.3)
        eigen2_label.next_to(eigen1_label, DOWN, buff=0.2)
        
        self.play(Create(axes))
        self.play(Create(eigen1), Write(eigen1_label))
        self.wait(2)
        self.play(Create(eigen2), Write(eigen2_label))
        self.wait(3)
        
        # Part 4: Matrix Representation
        part4_title = Text("4. 矩阵表示", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        matrix_rep = MathTex(
            r"L_{ij} = \langle \phi_i | L | \phi_j \rangle",
            font_size=32, color=BLACK
        )
        matrix_rep.next_to(part4_title, DOWN, buff=0.5)
        
        # Create a simple matrix example
        matrix_example = MathTex(
            r"L = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}",
            font_size=28, color=BLACK
        )
        matrix_example.next_to(matrix_rep, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        self.play(Write(matrix_rep))
        self.play(Write(matrix_example))
        self.wait(3)
        
        # Part 5: Spectral Decomposition
        part5_title = Text("5. 谱分解", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        spectral_decomp = MathTex(
            r"L = \sum_n \lambda_n |\phi_n\rangle \langle \phi_n|",
            font_size=32, color=BLACK
        )
        spectral_decomp.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(spectral_decomp))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class OpticalFieldTransformation(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("光学场变换", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D axes for field visualization
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
        
        # Define initial optical field (Gaussian beam)
        def initial_field(x, y):
            return np.exp(-(x**2 + y**2)) * np.cos(2*np.pi*x)
        
        # Create surface for initial field
        surface_initial = Surface(
            lambda u, v: axes.c2p(u, v, initial_field(u, v)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=30,
            fill_opacity=0.8,
            fill_color=BLUE
        )
        
        initial_label = Text("初始光场", font_size=24, color=BLUE)
        initial_label.next_to(axes, LEFT)
        
        self.play(Create(surface_initial), Write(initial_label))
        self.wait(2)
        
        # Apply linear transformation (lens operation)
        def lens_transformation(x, y):
            # Quadratic phase factor (thin lens)
            return np.exp(-(x**2 + y**2)) * np.cos(2*np.pi*x * (1 - 0.5*(x**2 + y**2)))
        
        # Create transformed surface
        surface_transformed = Surface(
            lambda u, v: axes.c2p(u, v, lens_transformation(u, v)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=30,
            fill_opacity=0.8,
            fill_color=RED
        )
        
        transformed_label = Text("变换后光场", font_size=24, color=RED)
        transformed_label.next_to(axes, RIGHT)
        
        # Transform the surface
        self.play(Transform(surface_initial, surface_transformed), 
                 Transform(initial_label, transformed_label))
        self.wait(3)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class EigenvalueDecompositionDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("特征值分解演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create a 2x2 matrix
        matrix = np.array([[3, 1], [1, 2]])
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(matrix)
        
        # Display the matrix
        matrix_tex = MathTex(
            r"A = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}",
            font_size=36, color=BLACK
        )
        matrix_tex.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title), Write(matrix_tex))
        self.wait(2)
        
        # Display eigenvalues
        eigenvalue_text = Text("特征值:", font_size=24, color=GREEN)
        eigenvalue_text.next_to(matrix_tex, DOWN, buff=0.5).to_edge(LEFT)
        
        eig1 = MathTex(f"\\lambda_1 = {eigenvalues[0]:.3f}", font_size=24, color=BLACK)
        eig2 = MathTex(f"\\lambda_2 = {eigenvalues[1]:.3f}", font_size=24, color=BLACK)
        
        eig1.next_to(eigenvalue_text, DOWN, buff=0.3)
        eig2.next_to(eig1, DOWN, buff=0.2)
        
        self.play(Write(eigenvalue_text))
        self.play(Write(eig1), Write(eig2))
        self.wait(2)
        
        # Display eigenvectors
        eigenvector_text = Text("特征向量:", font_size=24, color=RED)
        eigenvector_text.next_to(eigenvalue_text, RIGHT, buff=1.5)
        
        vec1 = MathTex(
            f"v_1 = \\begin{{pmatrix}} {eigenvectors[0,0]:.3f} \\\\ {eigenvectors[1,0]:.3f} \\end{{pmatrix}}",
            font_size=24, color=BLACK
        )
        vec2 = MathTex(
            f"v_2 = \\begin{{pmatrix}} {eigenvectors[0,1]:.3f} \\\\ {eigenvectors[1,1]:.3f} \\end{{pmatrix}}",
            font_size=24, color=BLACK
        )
        
        vec1.next_to(eigenvector_text, DOWN, buff=0.3)
        vec2.next_to(vec1, DOWN, buff=0.2)
        
        self.play(Write(eigenvector_text))
        self.play(Write(vec1), Write(vec2))
        self.wait(3)
        
        # Show diagonalization
        diagonalization = MathTex(
            r"A = PDP^{-1}",
            font_size=32, color=BLUE
        )
        diagonalization.next_to(eig2, DOWN, buff=0.8)
        
        self.play(Write(diagonalization))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))