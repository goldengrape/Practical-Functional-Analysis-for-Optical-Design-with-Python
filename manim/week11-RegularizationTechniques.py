from manim import *
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt

class RegularizationTechniques(Scene):
    def construct(self):
        # Set up the scene with a clean background
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("正则化技术", font_size=36, color=BLUE)
        subtitle = Text("Regularization Techniques in Optical Inverse Problems", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Part 1: Need for Regularization
        part1_title = Text("1. 正则化的必要性", font_size=28, color=GREEN)
        part1_title.next_to(title_group, DOWN, buff=0.8)
        
        ill_posed_problem = VGroup(
            MathTex(r"\min_m \|Gm - d\|^2", font_size=32, color=BLACK),
            Text("病态问题：小扰动导致大误差", font_size=24, color=RED),
            MathTex(r"\text{解不稳定：} \kappa(G) = \frac{\sigma_{max}}{\sigma_{min}} \gg 1", font_size=24, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        ill_posed_problem.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part1_title))
        for eq in ill_posed_problem:
            self.play(Write(eq))
            self.wait(1)
        self.wait(2)
        
        # Part 2: Tikhonov Regularization
        part2_title = Text("2. Tikhonov 正则化", font_size=28, color=GREEN)
        part2_title.next_to(ill_posed_problem, DOWN, buff=0.8)
        
        tikhonov = VGroup(
            MathTex(r"\min_m \|Gm - d\|^2 + \alpha^2 \|Lm\|^2", font_size=28, color=BLACK),
            MathTex(r"\text{正则化解：} m_\alpha = (G^T G + \alpha^2 L^T L)^{-1} G^T d", font_size=24, color=BLACK),
            Text("L = I (零阶), L = D (一阶), L = D² (二阶)", font_size=20, color=BLUE)
        ).arrange(DOWN, buff=0.3)
        
        tikhonov.next_to(part2_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part2_title))
        for eq in tikhonov:
            self.play(Write(eq))
            self.wait(1)
        self.wait(3)
        
        # Part 3: Parameter Selection
        part3_title = Text("3. 正则化参数选择", font_size=28, color=GREEN)
        part3_title.to_edge(LEFT).shift(UP * 1.5)
        
        parameter_methods = VGroup(
            MathTex(r"\text{L曲线法：} \log\|Gm_\alpha - d\| \text{ vs } \log\|Lm_\alpha\|", font_size=20, color=BLACK),
            MathTex(r"\text{交叉验证：} \text{留一法或K折验证}", font_size=20, color=BLACK),
            MathTex(r"\text{偏差原理：} \|Gm_\alpha - d\| \approx \delta", font_size=20, color=BLACK),
            MathTex(r"\text{广义交叉验证：} \text{GCV函数最小化}", font_size=20, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        parameter_methods.next_to(part3_title, DOWN, buff=0.5)
        
        self.play(FadeOut(part1_title), FadeOut(ill_posed_problem), 
                 FadeOut(part2_title), FadeOut(tikhonov))
        self.play(FadeIn(part3_title))
        for method in parameter_methods:
            self.play(Write(method))
            self.wait(1)
        self.wait(2)
        
        # Part 4: Iterative Regularization
        part4_title = Text("4. 迭代正则化", font_size=28, color=GREEN)
        part4_title.to_edge(RIGHT).shift(UP * 1.5)
        
        iterative_methods = VGroup(
            MathTex(r"\text{Landweber迭代：} m_{k+1} = m_k + \omega G^T(d - Gm_k)", font_size=20, color=BLACK),
            MathTex(r"\text{共轭梯度法：} \text{早期停止}", font_size=20, color=BLACK),
            MathTex(r"\text{LSQR：} \text{最小二乘QR分解}", font_size=20, color=BLACK),
            Text("迭代次数作为正则化参数", font_size=18, color=BLUE)
        ).arrange(DOWN, buff=0.3)
        
        iterative_methods.next_to(part4_title, DOWN, buff=0.5)
        
        self.play(FadeIn(part4_title))
        for method in iterative_methods:
            self.play(Write(method))
            self.wait(1)
        self.wait(3)
        
        # Part 5: Total Variation
        part5_title = Text("5. 全变分正则化", font_size=28, color=GREEN)
        part5_title.to_edge(DOWN).shift(UP * 0.5)
        
        total_variation = VGroup(
            MathTex(r"\min_m \|Gm - d\|^2 + \alpha \text{TV}(m)", font_size=24, color=BLACK),
            MathTex(r"\text{TV}(m) = \sum_i \sqrt{(\nabla_x m)_i^2 + (\nabla_y m)_i^2}", font_size=20, color=BLACK),
            Text("保持边缘信息，适合分段常数解", font_size=18, color=GREEN)
        ).arrange(DOWN, buff=0.3)
        
        total_variation.next_to(part5_title, UP, buff=0.3)
        
        self.play(FadeIn(part5_title))
        for eq in total_variation:
            self.play(Write(eq))
            self.wait(1)
        self.wait(3)
        
        # Fade out all elements
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)


class LCurveDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("L曲线参数选择", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create L-curve plot
        axes = Axes(
            x_range=[-2, 1, 0.5],
            y_range=[-2, 1, 0.5],
            x_length=6,
            y_length=6,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # L-curve function (log-log scale)
        def l_curve(x):
            # Typical L-curve shape
            if x < -1:
                return -0.5 * x - 1.5
            else:
                return -2 * x - 1
        
        # Plot L-curve
        l_curve_plot = axes.plot(l_curve, color=BLUE, stroke_width=3)
        
        # Mark the corner (optimal point)
        corner_point = Dot(axes.c2p(-1, 1), color=RED, radius=0.1)
        corner_label = MathTex(r"\alpha_{opt}", font_size=24, color=RED)
        corner_label.next_to(corner_point, UR, buff=0.2)
        
        # Axes labels
        x_label = MathTex(r"\log\|Lm_\alpha\|", font_size=24, color=BLACK)
        y_label = MathTex(r"\log\|Gm_\alpha - d\|", font_size=24, color=BLACK)
        
        x_label.next_to(axes, DOWN, buff=0.3)
        y_label.next_to(axes, LEFT, buff=0.3)
        
        self.play(Write(title))
        self.play(Create(axes))
        self.play(Create(l_curve_plot))
        self.play(Create(corner_point), Write(corner_label))
        self.play(Write(x_label), Write(y_label))
        
        # Explanation
        explanation = Text(
            "L曲线拐角处平衡了残差与解的范数",
            font_size=18, color=GREEN
        )
        explanation.next_to(axes, DOWN, buff=0.8)
        
        self.play(Write(explanation))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class IterativeRegularizationDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("迭代正则化演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for error vs iteration
        axes = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 1, 0.1],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # Error curves
        def data_error(k):
            return 0.8 * np.exp(-0.2 * k)
        
        def solution_error(k):
            # Initially decreases, then increases (semi-convergence)
            return 0.1 + 0.05 * (k - 8)**2 if k > 8 else 0.1 + 0.01 * k
        
        # Plot error curves
        data_error_plot = axes.plot(data_error, color=BLUE, stroke_width=3)
        solution_error_plot = axes.plot(solution_error, color=RED, stroke_width=3)
        
        # Labels
        data_label = MathTex(r"\|Gm_k - d\|", font_size=20, color=BLUE)
        solution_label = MathTex(r"\|m_k - m_{true}\|", font_size=20, color=RED)
        
        data_label.next_to(data_error_plot, UR, buff=0.2)
        solution_label.next_to(solution_error_plot, DR, buff=0.2)
        
        # Optimal stopping point
        optimal_k = 8
        optimal_line = axes.get_vertical_line(axes.c2p(optimal_k, solution_error(optimal_k)), color=GREEN, stroke_width=2)
        optimal_label = MathTex(r"k_{opt} = 8", font_size=20, color=GREEN)
        optimal_label.next_to(optimal_line, DOWN, buff=0.2)
        
        self.play(Write(title))
        self.play(Create(axes))
        self.play(Create(data_error_plot), Write(data_label))
        self.play(Create(solution_error_plot), Write(solution_label))
        self.play(Create(optimal_line), Write(optimal_label))
        
        # Axes labels
        x_label = MathTex(r"\text{迭代次数 } k", font_size=24, color=BLACK)
        y_label = MathTex(r"\text{误差}", font_size=24, color=BLACK)
        
        x_label.next_to(axes, DOWN, buff=0.3)
        y_label.next_to(axes, LEFT, buff=0.3)
        
        self.play(Write(x_label), Write(y_label))
        
        # Explanation
        explanation = Text(
            "半收敛现象：过早停止vs过拟合",
            font_size=18, color=GREEN
        )
        explanation.next_to(axes, DOWN, buff=0.8)
        
        self.play(Write(explanation))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class TotalVariationDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("全变分正则化演示", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create axes for 1D signal comparison
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "stroke_width": 2}
        )
        
        # True piecewise constant signal
        def true_signal(x):
            if x < 3:
                return 0.5
            elif x < 7:
                return 1.5
            else:
                return 0.8
        
        # Tikhonov regularized solution (smooth)
        def tikhonov_solution(x):
            # Smooth approximation
            return 0.5 + 0.8 * np.exp(-(x-5)**2 * 0.1)
        
        # TV regularized solution (piecewise constant)
        def tv_solution(x):
            # Better preserves edges
            if x < 2.8:
                return 0.52
            elif x < 7.2:
                return 1.48
            else:
                return 0.82
        
        # Create plots
        x_vals = np.linspace(0, 10, 100)
        true_plot = axes.plot_line_graph(
            x_vals, [true_signal(x) for x in x_vals],
            line_color=BLUE, stroke_width=3
        )
        
        tikhonov_plot = axes.plot_line_graph(
            x_vals, [tikhonov_solution(x) for x in x_vals],
            line_color=RED, stroke_width=2
        )
        
        tv_plot = axes.plot_line_graph(
            x_vals, [tv_solution(x) for x in x_vals],
            line_color=GREEN, stroke_width=3
        )
        
        # Labels
        true_label = MathTex(r"f_{true}(x)", font_size=20, color=BLUE)
        tikhonov_label = MathTex(r"f_{Tikhonov}(x)", font_size=20, color=RED)
        tv_label = MathTex(r"f_{TV}(x)", font_size=20, color=GREEN)
        
        labels = VGroup(true_label, tikhonov_label, tv_label)
        labels.arrange(DOWN, buff=0.1)
        labels.next_to(axes, RIGHT, buff=0.3)
        
        # Show plots
        self.play(Write(title))
        self.play(Create(axes))
        self.play(Create(true_plot), Write(true_label))
        self.wait(1)
        self.play(Create(tikhonov_plot), Write(tikhonov_label))
        self.wait(1)
        self.play(Create(tv_plot), Write(tv_label))
        self.wait(3)
        
        # Show TV definition
        tv_definition = MathTex(
            r"\text{TV}(f) = \int |\nabla f| dx \approx \sum_i |f_{i+1} - f_i|",
            font_size=20, color=BLACK
        )
        tv_definition.next_to(axes, DOWN, buff=0.5)
        
        self.play(Write(tv_definition))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))


class SparseRegularizationDemo(Scene):
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("稀疏正则化", font_size=32, color=BLUE)
        title.to_edge(UP)
        
        # Create comparison of different norms
        norms_comparison = VGroup(
            MathTex(r"\text{L2范数：} \|x\|_2^2 = \sum_i x_i^2", font_size=24, color=BLUE),
            MathTex(r"\text{L1范数：} \|x\|_1 = \sum_i |x_i|", font_size=24, color=RED),
            MathTex(r"\text{Lp范数：} \|x\|_p^p = \sum_i |x_i|^p, \; 0 < p < 1", font_size=24, color=GREEN),
            Text("", font_size=16),
            Text("L1范数促进稀疏性", font_size=20, color=BLACK),
            Text("Lp范数 (p<1) 更强的稀疏性", font_size=20, color=BLACK)
        ).arrange(DOWN, buff=0.3)
        
        norms_comparison.to_edge(LEFT).shift(RIGHT*0.5)
        
        self.play(Write(title))
        
        for norm in norms_comparison:
            self.play(Write(norm))
            self.wait(1)
        
        # Show LASSO formulation
        lasso_formulation = VGroup(
            MathTex(r"\text{LASSO：} \min_x \|Ax - b\|^2 + \lambda \|x\|_1", font_size=20, color=BLACK),
            MathTex(r"\text{压缩感知：} \min_x \|x\|_1 \text{ s.t. } Ax = b", font_size=20, color=BLACK),
            Text("稀疏表示：只有少数非零元素", font_size=18, color=GREEN)
        ).arrange(DOWN, buff=0.3)
        
        lasso_formulation.to_edge(RIGHT).shift(LEFT*0.5)
        
        for eq in lasso_formulation:
            self.play(Write(eq))
            self.wait(1)
        
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(Group(*self.mobjects)))