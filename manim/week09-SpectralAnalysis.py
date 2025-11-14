from manim import *
import numpy as np
from scipy.linalg import eig, eigvals
import matplotlib.pyplot as plt

class SpectralAnalysis(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("谱分析", font_size=36, color=BLUE)
        subtitle = Text("Spectral Analysis in Optical Systems", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Spectrum Definition
        part1_title = Text("1. 谱的定义", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        spectrum_def = VGroup(
            MathTex(r"\sigma(T) = \{\lambda \in \mathbb{C} : (T - \lambda I) \text{ 不可逆}\}", font_size=28, color=BLACK),
            MathTex(r"\text{点谱：} \sigma_p(T) = \{\lambda : \exists v \neq 0, Tv = \lambda v\}", font_size=24, color=BLACK),
            MathTex(r"\text{连续谱：} \sigma_c(T)\text{，剩余谱：} \sigma_r(T)", font_size=24, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        spectrum_def.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        for eq in spectrum_def:
            self.play(Write(eq))
            self.wait(1)
        self.wait(2)
        
        # Part 2: Eigenvalue Problem
        part2_title = Text("2. 特征值问题", font_size=28, color=GREEN)
        part2_title.next_to(spectrum_def, DOWN, buff=0.8)
        
        eigenvalue_problem = MathTex(
            r"Av = \lambda v, \quad v \neq 0",
            font_size=32, color=BLACK
        )
        eigenvalue_problem.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        self.play(Write(eigenvalue_problem))
        self.wait(3)
        
        # Part 3: Spectral Decomposition
        part3_title = Text("3. 谱分解", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        spectral_decomp = MathTex(
            r"A = \sum_{i=1}^n \lambda_i P_i",
            font_size=32, color=BLACK
        )
        spectral_decomp.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(spectrum_def), 
                 FadeOut(part2_title), FadeOut(eigenvalue_problem))
        self.play(FadeIn(part3_title))
        self.play(Write(spectral_decomp))
        self.wait(2)
        
        # Create visualization of eigenvalues
        axes = Axes(
            x_range=[-3, 3, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(spectral_decomp, DOWN, buff=0.5)
        
        # Create a matrix and compute its eigenvalues
        matrix = np.array([[2, 1], [1, 3]])
        eigenvalues, eigenvectors = eig(matrix)
        
        # Plot eigenvalues on complex plane
        eigenvalue_dots = VGroup()
        for i, (ev, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            dot = Dot(axes.c2p(ev.real, ev.imag), color=RED, radius=0.1)
            eigenvalue_dots.add(dot)
        
        eigenvalue_label = MathTex(r"\sigma(A) = \{1.38, 3.62\}", font_size=24, color=RED)
        eigenvalue_label.next_to(axes, RIGHT, buff=0.3)
        
        self.play(Create(axes))
        self.play(Create(eigenvalue_dots), Write(eigenvalue_label))
        self.wait(3)
        
        # Part 4: Spectral Theorem
        part4_title = Text("4. 谱定理", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        spectral_theorem = MathTex(
            r"A = U \Lambda U^*, \quad U^*U = I",
            font_size=32, color=BLACK
        )
        spectral_theorem.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        self.play(Write(spectral_theorem))
        self.wait(3)
        
        # Part 5: Applications in Optics
        part5_title = Text("5. 光学中的应用", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        optical_application = MathTex(
            r"\text{共振模式：} H \psi = \omega^2 \psi",
            font_size=28, color=BLACK
        )
        optical_application.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(optical_application))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class EigenvalueEvolution(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("特征值演化", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for parameter-dependent eigenvalues
        axes = Axes(
            x_range=[0, 5, 0.5],
            y_range=[-1, 4, 0.5],
            x_length=8,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # Parameter-dependent matrix
        def param_matrix(alpha):
            return np.array([[2 + alpha, 1 - alpha], [1 - alpha, 3 + 0.5*alpha]])
        
        # Compute eigenvalues for different parameter values
        alpha_values = np.linspace(0, 5, 50)
        eigenvalue_tracks = [[], []]
        
        for alpha in alpha_values:
            matrix = param_matrix(alpha)
            eigenvals = eigvals(matrix)
            eigenvalue_tracks[0].append(eigenvals[0])
            eigenvalue_tracks[1].append(eigenvals[1])
        
        # Plot eigenvalue trajectories
        track1 = VMobject()
        track1.set_points_smoothly([
            axes.c2p(alpha, eig) for alpha, eig in zip(alpha_values, eigenvalue_tracks[0])
        ])
        track1.set_color(BLUE)
        track1.set_stroke(width=3)
        
        track2 = VMobject()
        track2.set_points_smoothly([
            axes.c2p(alpha, eig) for alpha, eig in zip(alpha_values, eigenvalue_tracks[1])
        ])
        track2.set_color(RED)
        track2.set_stroke(width=3)
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Animate the eigenvalue evolution
        self.play(Create(track1), Create(track2))
        
        # Labels
        param_label = MathTex(r"\alpha \text{ (参数)}", font_size=24, color=BLACK)
        param_label.next_to(axes, DOWN, buff=0.3)
        
        eigenvalue_label = MathTex(r"\lambda \text{ (特征值)}", font_size=24, color=BLACK)
        eigenvalue_label.next_to(axes, LEFT, buff=0.3)
        
        track1_label = MathTex(r"\lambda_1(\alpha)", font_size=20, color=BLUE)
        track1_label.next_to(track1, RIGHT, buff=0.2)
        
        track2_label = MathTex(r"\lambda_2(\alpha)", font_size=20, color=RED)
        track2_label.next_to(track2, LEFT, buff=0.2)
        
        self.play(Write(param_label), Write(eigenvalue_label))
        self.play(Write(track1_label), Write(track2_label))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class SpectralDecompositionDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("谱分解演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create a symmetric matrix
        matrix = np.array([[3, 1], [1, 2]])
        eigenvalues, eigenvectors = eig(matrix)
        
        # Display original matrix
        matrix_tex = MathTex(
            r"A = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}",
            font_size=36, color=BLACK
        )
        matrix_tex.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title), Write(matrix_tex))
        self.wait(2)
        
        # Display eigenvalues and eigenvectors
        eigen_info = VGroup(
            MathTex(f"\\lambda_1 = {eigenvalues[0]:.3f}", font_size=24, color=BLUE),
            MathTex(f"v_1 = \\begin{{pmatrix}} {eigenvectors[0,0]:.3f} \\\\ {eigenvectors[1,0]:.3f} \\end{{pmatrix}}", font_size=24, color=BLUE),
            MathTex(f"\\lambda_2 = {eigenvalues[1]:.3f}", font_size=24, color=RED),
            MathTex(f"v_2 = \\begin{{pmatrix}} {eigenvectors[0,1]:.3f} \\\\ {eigenvectors[1,1]:.3f} \\end{{pmatrix}}", font_size=24, color=RED)
        ).arrange(DOWN, buff=0.2)
        
        eigen_info.next_to(matrix_tex, DOWN, buff=0.8)
        
        for info in eigen_info:
            self.play(Write(info))
            self.wait(1)
        
        # Show spectral decomposition
        decomposition = VGroup(
            MathTex(r"A = \lambda_1 v_1 v_1^T + \lambda_2 v_2 v_2^T", font_size=24, color=BLACK),
            MathTex(r"A = \sum_{i=1}^n \lambda_i P_i", font_size=24, color=BLACK),
            Text("其中 \(P_i = v_i v_i^T\) 是投影矩阵", font_size=20, color=GREEN)
        ).arrange(DOWN, buff=0.3)
        
        decomposition.next_to(eigen_info, DOWN, buff=0.8)
        
        for eq in decomposition:
            self.play(Write(eq))
            self.wait(1)
        
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class OpticalResonanceModes(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("光学共振模式", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D visualization of resonator modes
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
        
        # Define different resonance modes
        def mode_00(x, y):
            # Fundamental mode
            return np.exp(-(x**2 + y**2))
        
        def mode_10(x, y):
            # First order mode in x
            return x * np.exp(-(x**2 + y**2))
        
        def mode_01(x, y):
            # First order mode in y
            return y * np.exp(-(x**2 + y**2))
        
        def mode_11(x, y):
            # First order mode in both
            return x * y * np.exp(-(x**2 + y**2))
        
        modes = [mode_00, mode_10, mode_01, mode_11]
        mode_names = ["TEM_{00}", "TEM_{10}", "TEM_{01}", "TEM_{11}"]
        colors = [BLUE, RED, GREEN, ORANGE]
        
        # Show each mode
        for i, (mode, name, color) in enumerate(zip(modes, mode_names, colors)):
            surface = Surface(
                lambda u, v: axes.c2p(u, v, mode(u, v)),
                u_range=[-1.5, 1.5],
                v_range=[-1.5, 1.5],
                resolution=25,
                fill_opacity=0.7,
                fill_color=color
            )
            
            mode_label = MathTex(f"\text{{{name}}}", font_size=24, color=color)
            mode_label.next_to(surface, RIGHT)
            
            self.play(Create(surface), Write(mode_label))
            self.wait(2)
            
            if i < len(modes) - 1:
                self.play(FadeOut(surface), FadeOut(mode_label))
        
        # Show mode equation
        mode_equation = MathTex(
            r"\nabla^2 \psi + k^2 \psi = \lambda \psi",
            font_size=28, color=BLACK
        )
        mode_equation.to_edge(DOWN)
        
        self.play(Write(mode_equation))
        self.wait(3)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class SpectralGapVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("谱隙与收敛性", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 3, 0.3],
            x_length=8,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # Define eigenvalues with spectral gap
        lambda1 = 2.5
        lambda2 = 2.0
        gap = lambda1 - lambda2
        
        # Plot eigenvalues as horizontal lines
        line1 = axes.plot(lambda x: lambda1, color=BLUE, stroke_width=4)
        line2 = axes.plot(lambda x: lambda2, color=RED, stroke_width=4)
        
        # Highlight the spectral gap
        gap_region = axes.get_area(
            line1, x_range=[0, 10], 
            bounded_graph=line2, 
            color=YELLOW, 
            opacity=0.3
        )
        
        # Labels
        gap_label = MathTex(f"\\text{{谱隙}} = {gap:.1f}", font_size=24, color=YELLOW)
        gap_label.next_to(gap_region, RIGHT, buff=0.3)
        
        lambda1_label = MathTex(f"\\lambda_1 = {lambda1}", font_size=20, color=BLUE)
        lambda1_label.next_to(line1, RIGHT, buff=0.2)
        
        lambda2_label = MathTex(f"\\lambda_2 = {lambda2}", font_size=20, color=RED)
        lambda2_label.next_to(line2, RIGHT, buff=0.2)
        
        convergence_text = Text(
            "谱隙越大，收敛速度越快",
            font_size=20, color=GREEN
        )
        convergence_text.to_edge(DOWN)
        
        self.play(Write(title))
        self.play(Create(axes))
        self.play(Create(line1), Create(line2))
        self.play(Create(gap_region), Write(gap_label))
        self.play(Write(lambda1_label), Write(lambda2_label))
        self.play(Write(convergence_text))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))