from manim import *
import numpy as np
from scipy.linalg import lstsq, pinv
from scipy.sparse import diags
import matplotlib.pyplot as plt

class InverseProblems(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("逆问题", font_size=36, color=BLUE)
        subtitle = Text("Inverse Problems in Optical Design", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Forward vs Inverse Problem
        part1_title = Text("1. 正问题 vs 逆问题", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        forward_inverse = VGroup(
            MathTex(r"\text{正问题：} d = G(m)", font_size=28, color=BLACK),
            MathTex(r"\text{逆问题：} m = G^{-1}(d)", font_size=28, color=BLACK),
            Text("已知模型参数，预测观测数据", font_size=20, color=BLUE),
            Text("已知观测数据，反推模型参数", font_size=20, color=RED)
        ).arrange(DOWN, buff=0.3)
        
        forward_inverse.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        for eq in forward_inverse:
            self.play(Write(eq))
            self.wait(1)
        self.wait(2)
        
        # Part 2: Ill-posedness
        part2_title = Text("2. 病态问题特征", font_size=28, color=GREEN)
        part2_title.next_to(forward_inverse, DOWN, buff=0.8)
        
        ill_posedness = VGroup(
            Text("1. 解不存在", font_size=24, color=BLACK),
            Text("2. 解不唯一", font_size=24, color=BLACK),
            Text("3. 解不稳定", font_size=24, color=BLACK),
            MathTex(r"\text{小扰动} \delta d \Rightarrow \text{大扰动} \delta m", font_size=24, color=RED)
        ).arrange(DOWN, buff=0.3)
        
        ill_posedness.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        for condition in ill_posedness:
            self.play(Write(condition))
            self.wait(1)
        self.wait(2)
        
        # Part 3: Linear Inverse Problem
        part3_title = Text("3. 线性逆问题", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        linear_inverse = MathTex(
            r"d = G m + \epsilon",
            font_size=32, color=BLACK
        )
        linear_inverse.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(forward_inverse), 
                 FadeOut(part2_title), FadeOut(ill_posedness))
        self.play(FadeIn(part3_title))
        self.play(Write(linear_inverse))
        self.wait(2)
        
        # Create visualization of matrix condition
        axes = Axes(
            x_range=[0, 5, 0.5],
            y_range=[0, 2, 0.2],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        axes.next_to(linear_inverse, DOWN, buff=0.5)
        
        # Singular values decay
        def singular_values(k):
            return 2.0 * np.exp(-0.5 * k)
        
        sv_plot = axes.plot(singular_values, color=BLUE, stroke_width=3)
        
        sv_label = MathTex(r"\sigma_k(G)", font_size=24, color=BLUE)
        sv_label.next_to(axes, RIGHT, buff=0.3)
        
        condition_number = MathTex(r"\kappa(G) = \frac{\sigma_{max}}{\sigma_{min}}", font_size=24, color=RED)
        condition_number.next_to(axes, DOWN, buff=0.3)
        
        self.play(Create(axes))
        self.play(Create(sv_plot), Write(sv_label))
        self.play(Write(condition_number))
        self.wait(3)
        
        # Part 4: Regularization
        part4_title = Text("4. 正则化方法", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        regularization = VGroup(
            MathTex(r"\text{Tikhonov：} (G^T G + \alpha I) m = G^T d", font_size=24, color=BLACK),
            MathTex(r"\text{截断SVD：} m = \sum_{i=1}^k \frac{u_i^T d}{\sigma_i} v_i", font_size=24, color=BLACK),
            MathTex(r"\text{迭代法：} m_{k+1} = m_k + \beta G^T (d - G m_k)", font_size=24, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        regularization.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        for method in regularization:
            self.play(Write(method))
            self.wait(1)
        self.wait(3)
        
        # Part 5: Bayesian Approach
        part5_title = Text("5. 贝叶斯方法", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        bayesian = MathTex(
            r"P(m|d) \propto P(d|m) P(m)",
            font_size=32, color=BLACK
        )
        bayesian.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        self.play(Write(bayesian))
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class DeconvolutionDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("反卷积问题演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for 1D signal
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-0.5, 1.5, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # Original signal (spike)
        def original_signal(x):
            return np.exp(-(x-5)**2 * 2)
        
        # Point spread function (Gaussian blur)
        def psf(x):
            return np.exp(-(x-5)**2 * 0.5)
        
        # Blurred signal (convolution)
        def blurred_signal(x):
            # Approximate convolution result
            return 0.6 * np.exp(-(x-5)**2 * 0.3)
        
        # Plot signals
        original_plot = axes.plot(original_signal, color=BLUE, stroke_width=3)
        psf_plot = axes.plot(psf, color=GREEN, stroke_width=2)
        blurred_plot = axes.plot(blurred_signal, color=RED, stroke_width=3)
        
        # Labels
        original_label = MathTex(r"f(x)", font_size=20, color=BLUE)
        psf_label = MathTex(r"h(x)", font_size=20, color=GREEN)
        blurred_label = MathTex(r"g(x) = (f * h)(x)", font_size=20, color=RED)
        
        labels = VGroup(original_label, psf_label, blurred_label)
        labels.arrange(DOWN, buff=0.1)
        labels.next_to(axes, RIGHT, buff=0.3)
        
        # Show original signal
        self.play(Create(original_plot), Write(original_label))
        self.wait(2)
        
        # Show PSF
        self.play(Create(psf_plot), Write(psf_label))
        self.wait(2)
        
        # Show blurred signal
        self.play(Create(blurred_plot), Write(blurred_label))
        self.wait(3)
        
        # Show inverse problem equation
        inverse_eq = MathTex(
            r"f = h^{-1} * g",
            font_size=28, color=BLACK
        )
        inverse_eq.next_to(axes, DOWN, buff=0.5)
        
        self.play(Write(inverse_eq))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class TomographyVisualization(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("层析成像逆问题", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create 3D visualization of tomography
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
        
        # Create a simple phantom (object to be reconstructed)
        def phantom_object(x, y, z):
            # Two spheres
            sphere1 = np.exp(-((x+0.5)**2 + y**2 + z**2) * 4)
            sphere2 = np.exp(-((x-0.5)**2 + y**2 + z**2) * 4)
            return 0.7 * (sphere1 + sphere2)
        
        # Create phantom surface
        phantom_surface = Surface(
            lambda u, v: axes.c2p(u, v, phantom_object(u, v, 0)),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=25,
            fill_opacity=0.7,
            fill_color=BLUE
        )
        
        phantom_label = Text("原始物体", font_size=20, color=BLUE)
        phantom_label.next_to(phantom_surface, LEFT)
        
        self.play(Create(phantom_surface), Write(phantom_label))
        self.wait(2)
        
        # Show projection lines (X-ray paths)
        projection_lines = VGroup()
        for i in range(-2, 3):
            for j in range(-2, 3):
                line = Line3D(
                    start=axes.c2p(i*0.4, j*0.4, -1),
                    end=axes.c2p(i*0.4, j*0.4, 1),
                    color=RED,
                    stroke_width=1
                )
                projection_lines.add(line)
        
        projection_label = Text("投影数据", font_size=20, color=RED)
        projection_label.next_to(projection_lines, RIGHT)
        
        self.play(Create(projection_lines), Write(projection_label))
        self.wait(3)
        
        # Show reconstruction equation
        tomography_eq = MathTex(
            r"p = A f, \quad f = A^{-1} p",
            font_size=28, color=BLACK
        )
        tomography_eq.to_edge(DOWN)
        
        self.play(Write(tomography_eq))
        self.wait(3)
        
        # Rotate camera for better view
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES, run_time=3)
        self.wait(2)


class RegularizationComparison(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("正则化方法比较", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create comparison table
        comparison_text = VGroup(
            Text("Tikhonov正则化：", font_size=24, color=BLUE),
            Text("• 稳定但可能过平滑", font_size=20, color=BLACK),
            Text("• 需要选择正则化参数", font_size=20, color=BLACK),
            Text("", font_size=16),
            Text("截断SVD：", font_size=24, color=RED),
            Text("• 保留主要特征值", font_size=20, color=BLACK),
            Text("• 可能丢失细节信息", font_size=20, color=BLACK),
            Text("", font_size=16),
            Text("迭代法：", font_size=24, color=GREEN),
            Text("• 计算效率高", font_size=20, color=BLACK),
            Text("• 需要合适的停止准则", font_size=20, color=BLACK)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        
        comparison_text.to_edge(LEFT).shift(RIGHT*0.5)
        
        self.play(Write(title))
        
        for text in comparison_text:
            self.play(Write(text))
            self.wait(0.5)
        
        # Show parameter selection
        parameter_selection = VGroup(
            MathTex(r"\text{L曲线法：} \|Gm - d\|^2 + \alpha^2 \|m\|^2", font_size=20, color=BLACK),
            MathTex(r"\text{交叉验证：} \text{最小化预测误差}", font_size=20, color=BLACK),
            MathTex(r"\text{偏差原理：} \|Gm - d\| \approx \delta", font_size=20, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        parameter_selection.to_edge(RIGHT).shift(LEFT*0.5)
        
        for eq in parameter_selection:
            self.play(Write(eq))
            self.wait(1)
        
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class InverseProblemExample(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("光学逆问题实例", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for demonstrating the problem
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        self.play(Write(title))
        self.play(Create(axes))
        
        # True refractive index profile
        def true_profile(x):
            return 1.0 + 0.3 * np.exp(-(x-5)**2 * 0.5)
        
        # Measured data (with noise)
        def measured_data(x):
            return true_profile(x) + 0.05 * np.random.normal(0, 1, len(x) if hasattr(x, '__len__') else 1)
        
        # Reconstructed profile (regularized solution)
        def reconstructed_profile(x):
            return 1.0 + 0.25 * np.exp(-(x-5)**2 * 0.4)
        
        # Create plots
        x_vals = np.linspace(0, 10, 100)
        true_plot = axes.plot_line_graph(
            x_vals, [true_profile(x) for x in x_vals],
            line_color=BLUE, stroke_width=3
        )
        
        measured_plot = axes.plot_line_graph(
            x_vals, [measured_data(x) for x in x_vals],
            line_color=RED, stroke_width=2
        )
        
        reconstructed_plot = axes.plot_line_graph(
            x_vals, [reconstructed_profile(x) for x in x_vals],
            line_color=GREEN, stroke_width=3
        )
        
        # Labels
        true_label = MathTex(r"n_{true}(x)", font_size=20, color=BLUE)
        measured_label = MathTex(r"d_{measured}(x)", font_size=20, color=RED)
        reconstructed_label = MathTex(r"n_{reconstructed}(x)", font_size=20, color=GREEN)
        
        labels = VGroup(true_label, measured_label, reconstructed_label)
        labels.arrange(DOWN, buff=0.1)
        labels.next_to(axes, RIGHT, buff=0.3)
        
        # Show plots
        self.play(Create(true_plot), Write(true_label))
        self.wait(1)
        self.play(Create(measured_plot), Write(measured_label))
        self.wait(1)
        self.play(Create(reconstructed_plot), Write(reconstructed_label))
        self.wait(3)
        
        # Show solution quality
        quality_text = Text(
            "正则化平衡了数据拟合与解的平滑性",
            font_size=20, color=BLACK
        )
        quality_text.next_to(axes, DOWN, buff=0.5)
        
        self.play(Write(quality_text))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))