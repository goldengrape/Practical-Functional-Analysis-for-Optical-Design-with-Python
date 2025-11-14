# BrachistochroneScene.py
#
# To run this animation, ensure you have Manim installed:
# pip install manim scipy
#
# Then, execute the following command in your terminal:
# manim -pqh render BrachistochroneScene.py BrachistochroneAnimation

from manim import *
import numpy as np
from scipy.optimize import minimize_scalar
import icontract

# Physical constants for the brachistochrone problem
GRAVITY = 9.81  # m/s²

@icontract.require(lambda x: x >= 0)
def cycloid_time(x: float, y_end: float = -1.0) -> float:
    """
    Calculate the time for a bead to slide along a cycloid curve.
    This is the analytical solution to the brachistochrone problem.
    """
    # Cycloid parameter: x = a*(t - sin(t)), y = -a*(1 - cos(t))
    # We need to find the parameter 'a' such that the curve passes through (x, y_end)
    
    # For a cycloid starting at (0,0) and ending at (x, y_end):
    # The parameter 'a' is determined by the endpoint condition
    if x == 0:
        return float('inf')  # Invalid case
    
    # Numerically solve for the cycloid parameter
    def cycloid_constraint(a):
        # Find t such that y = -a*(1 - cos(t)) = y_end
        # and x = a*(t - sin(t))
        if a <= 0:
            return float('inf')
        
        # For given 'a', find 't' that satisfies y constraint
        from scipy.optimize import fsolve
        def find_t(t):
            return -a * (1 - np.cos(t)) - y_end
        
        try:
            t_solution = fsolve(find_t, np.pi)[0]
            # Return the difference in x coordinates
            return a * (t_solution - np.sin(t_solution)) - x
        except:
            return float('inf')
    
    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda a: abs(cycloid_constraint(a)), bounds=(0.01, 10), method='bounded')
        a_optimal = result.x
        
        # Find the corresponding t value
        def find_t_final(t):
            return -a_optimal * (1 - np.cos(t)) - y_end
        
        t_final = fsolve(find_t_final, np.pi)[0]
        
        # Calculate the time using the brachistochrone formula
        # T = sqrt(a/g) * t_final
        time = np.sqrt(a_optimal / GRAVITY) * t_final
        return time
    except:
        return float('inf')

def straight_line_time(x: float, y_end: float = -1.0) -> float:
    """Calculate time for straight line path."""
    distance = np.sqrt(x**2 + y_end**2)
    # Average velocity: v_avg = sqrt(g * |y_end| / 2)
    v_avg = np.sqrt(GRAVITY * abs(y_end) / 2)
    return distance / v_avg

def parabola_time(x: float, y_end: float = -1.0) -> float:
    """Calculate time for parabolic path."""
    # Parabola: y = -k * x², where k is chosen to pass through (x, y_end)
    if x == 0:
        return float('inf')
    k = -y_end / (x**2)
    
    # Time integral for parabolic path
    # T = integral[0 to x] sqrt((1 + (dy/dx)²) / (2g(kx² - ky²))) dx
    # This is a numerical approximation
    def integrand(xi):
        y = -k * xi**2
        dy_dx = -2 * k * xi
        return np.sqrt((1 + dy_dx**2) / (2 * GRAVITY * (k * x**2 - k * xi**2)))
    
    from scipy.integrate import quad
    time, _ = quad(integrand, 0, x)
    return time

class BrachistochroneAnimation(Scene):
    """
    Animation demonstrating the brachistochrone problem - finding the curve 
    of fastest descent between two points under gravity.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # --- Title and Setup ---
        title = Text("最速降线问题", font_size=48, color=BLACK)
        subtitle = Text("哪条路径下降最快？", font_size=32, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # --- Coordinate System ---
        axes = Axes(
            x_range=(-1, 6),
            y_range=(-3, 1),
            axis_config={"color": GRAY, "stroke_width": 2},
            x_length=8,
            y_length=6
        )
        
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Start and end points
        start_point = np.array([0, 0, 0])
        end_point = np.array([4, -2, 0])
        
        start_dot = Dot(axes.c2p(*start_point), color=RED, radius=0.1)
        end_dot = Dot(axes.c2p(*end_point), color=RED, radius=0.1)
        
        start_label = Text("起点", font_size=24, color=BLACK).next_to(start_dot, LEFT)
        end_label = Text("终点", font_size=24, color=BLACK).next_to(end_dot, RIGHT)
        
        self.play(
            Create(axes),
            Write(axis_labels),
            FadeIn(start_dot),
            FadeIn(end_dot),
            Write(start_label),
            Write(end_label)
        )
        self.wait(1)
        
        # --- Different Path Candidates ---
        
        # 1. Straight line path
        straight_line = Line(
            axes.c2p(*start_point),
            axes.c2p(*end_point),
            color=BLUE,
            stroke_width=4
        )
        
        straight_label = Text("直线路径", font_size=20, color=BLUE)
        straight_label.next_to(straight_line, UP, buff=0.2)
        
        # Calculate time for straight line
        straight_time = straight_line_time(4, -2)
        straight_time_text = Text(f"时间: {straight_time:.2f}s", font_size=18, color=BLUE)
        straight_time_text.next_to(straight_label, UP, buff=0.1)
        
        self.play(
            Create(straight_line),
            Write(straight_label),
            Write(straight_time_text)
        )
        self.wait(1.5)
        
        # 2. Parabolic path
        parabola_x = np.linspace(0, 4, 100)
        parabola_y = -0.125 * parabola_x**2  # Parabola through (0,0) and (4,-2)
        parabola_points = [axes.c2p(x, y, 0) for x, y in zip(parabola_x, parabola_y)]
        
        parabola_curve = VMobject()
        parabola_curve.set_points_smoothly(parabola_points)
        parabola_curve.set_stroke(color=GREEN, width=4)
        
        parabola_label = Text("抛物线路径", font_size=20, color=GREEN)
        parabola_label.next_to(parabola_curve, DOWN, buff=0.2)
        
        # Calculate time for parabola
        parabola_time_val = parabola_time(4, -2)
        parabola_time_text = Text(f"时间: {parabola_time_val:.2f}s", font_size=18, color=GREEN)
        parabola_time_text.next_to(parabola_label, DOWN, buff=0.1)
        
        self.play(
            Create(parabola_curve),
            Write(parabola_label),
            Write(parabola_time_text)
        )
        self.wait(1.5)
        
        # 3. Cycloid path (brachistochrone solution)
        # Parametric cycloid: x = a*(t - sin(t)), y = -a*(1 - cos(t))
        # We need to find the parameter 'a' such that the curve passes through (4, -2)
        
        # Solve for cycloid parameter
        from scipy.optimize import fsolve
        
        def find_cycloid_params():
            def equations(vars):
                a, t_final = vars
                eq1 = a * (t_final - np.sin(t_final)) - 4  # x constraint
                eq2 = -a * (1 - np.cos(t_final)) + 2   # y constraint
                return [eq1, eq2]
            
            solution = fsolve(equations, [1.0, np.pi])
            return solution[0], solution[1]
        
        a_param, t_final = find_cycloid_params()
        
        # Generate cycloid points
        t_vals = np.linspace(0, t_final, 100)
        cycloid_x = a_param * (t_vals - np.sin(t_vals))
        cycloid_y = -a_param * (1 - np.cos(t_vals))
        cycloid_points = [axes.c2p(x, y, 0) for x, y in zip(cycloid_x, cycloid_y)]
        
        cycloid_curve = VMobject()
        cycloid_curve.set_points_smoothly(cycloid_points)
        cycloid_curve.set_stroke(color=RED, width=4)
        
        cycloid_label = Text("摆线路径 (最速降线)", font_size=20, color=RED)
        cycloid_label.next_to(cycloid_curve, RIGHT, buff=0.2)
        
        # Calculate time for cycloid (analytical solution)
        cycloid_time_val = np.sqrt(a_param / GRAVITY) * t_final
        cycloid_time_text = Text(f"时间: {cycloid_time_val:.2f}s", font_size=18, color=RED)
        cycloid_time_text.next_to(cycloid_label, UP, buff=0.1)
        
        self.play(
            Create(cycloid_curve),
            Write(cycloid_label),
            Write(cycloid_time_text)
        )
        self.wait(2)
        
        # --- Animation of Beads Sliding Down ---
        
        # Create beads for each path
        straight_bead = Dot(axes.c2p(*start_point), color=BLUE, radius=0.08)
        parabola_bead = Dot(axes.c2p(*start_point), color=GREEN, radius=0.08)
        cycloid_bead = Dot(axes.c2p(*start_point), color=RED, radius=0.08)
        
        # Animate beads sliding down
        self.play(
            FadeIn(straight_bead),
            FadeIn(parabola_bead),
            FadeIn(cycloid_bead)
        )
        
        # Create motion along paths
        straight_animation = straight_bead.animate(rate_func=rate_functions.ease_in_quad).move_to(
            axes.c2p(*end_point)
        )
        
        parabola_animation = UpdateFromAlphaFunc(
            parabola_bead,
            lambda bead, alpha: bead.move_to(
                axes.c2p(parabola_x[int(alpha * (len(parabola_x) - 1))], 
                        parabola_y[int(alpha * (len(parabola_y) - 1))])
            ),
            rate_func=rate_functions.ease_in_quad
        )
        
        cycloid_animation = UpdateFromAlphaFunc(
            cycloid_bead,
            lambda bead, alpha: bead.move_to(
                axes.c2p(cycloid_x[int(alpha * (len(cycloid_x) - 1))], 
                        cycloid_y[int(alpha * (len(cycloid_y) - 1))])
            ),
            rate_func=rate_functions.ease_in_quad
        )
        
        # Run animations simultaneously but with different durations
        self.play(
            straight_animation,
            parabola_animation,
            cycloid_animation,
            run_time=3
        )
        
        self.wait(1)
        
        # --- Conclusion: Highlight the Winner ---
        
        # Create a results box
        results_box = Rectangle(
            width=4,
            height=2,
            stroke_color=BLACK,
            stroke_width=2,
            fill_color=WHITE,
            fill_opacity=0.9
        )
        results_box.to_edge(RIGHT, buff=0.5)
        
        results_title = Text("结果对比", font_size=24, color=BLACK)
        results_title.next_to(results_box, UP, buff=0.2)
        
        # Results text
        results_text = VGroup(
            Text(f"直线: {straight_time:.2f}s", font_size=18, color=BLUE),
            Text(f"抛物线: {parabola_time_val:.2f}s", font_size=18, color=GREEN),
            Text(f"摆线: {cycloid_time_val:.2f}s ✓", font_size=18, color=RED)
        )
        results_text.arrange(DOWN, buff=0.1, center=True)
        results_text.move_to(results_box.get_center())
        
        self.play(
            Create(results_box),
            Write(results_title),
            Write(results_text)
        )
        
        self.wait(2)
        
        # --- Mathematical Formula ---
        
        # Show the cycloid parametric equations
        formula_title = Text("摆线参数方程", font_size=24, color=BLACK)
        formula_title.to_edge(LEFT, buff=0.5)
        formula_title.shift(UP * 2)
        
        cycloid_formulas = MathTex(
            r"x &= a(t - \sin t) \\\n y &= -a(1 - \cos t)",
            font_size=20,
            color=BLACK
        )
        cycloid_formulas.next_to(formula_title, DOWN, buff=0.3)
        
        self.play(
            Write(formula_title),
            Write(cycloid_formulas)
        )
        
        self.wait(2)
        
        # Fade out
        self.play(
            FadeOut(title_group),
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(start_dot),
            FadeOut(end_dot),
            FadeOut(start_label),
            FadeOut(end_label),
            FadeOut(straight_line),
            FadeOut(straight_label),
            FadeOut(straight_time_text),
            FadeOut(parabola_curve),
            FadeOut(parabola_label),
            FadeOut(parabola_time_text),
            FadeOut(cycloid_curve),
            FadeOut(cycloid_label),
            FadeOut(cycloid_time_text),
            FadeOut(straight_bead),
            FadeOut(parabola_bead),
            FadeOut(cycloid_bead),
            FadeOut(results_box),
            FadeOut(results_title),
            FadeOut(results_text),
            FadeOut(formula_title),
            FadeOut(cycloid_formulas)
        )
        
        self.wait(1)


class BrachistochroneDerivation(Scene):
    """
    Mathematical derivation of the brachistochrone problem using calculus of variations.
    Shows how the Euler-Lagrange equation leads to the cycloid solution.
    """
    
    def construct(self):
        self.camera.background_color = "#fefcfb"
        
        # Title
        title = Text("最速降线推导", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        
        # Step 1: Time functional
        step1 = Text("1. 时间泛函", font_size=24, color=BLUE)
        step1.to_edge(LEFT, buff=0.5)
        step1.shift(UP * 2)
        
        time_functional = MathTex(
            r"T[y(x)] = \int_{x_1}^{x_2} \frac{\sqrt{1 + (y')^2}}{\sqrt{2g(y_1 - y)}} dx",
            font_size=24,
            color=BLACK
        )
        time_functional.next_to(step1, DOWN, buff=0.3)
        
        self.play(Write(step1))
        self.play(Write(time_functional))
        self.wait(2)
        
        # Step 2: Euler-Lagrange equation
        step2 = Text("2. 欧拉-拉格朗日方程", font_size=24, color=GREEN)
        step2.next_to(time_functional, DOWN, buff=0.5)
        
        el_equation = MathTex(
            r"\frac{d}{dx}\left(\frac{\partial L}{\partial y'}\right) - \frac{\partial L}{\partial y} = 0",
            font_size=24,
            color=BLACK
        )
        el_equation.next_to(step2, DOWN, buff=0.3)
        
        self.play(Write(step2))
        self.play(Write(el_equation))
        self.wait(2)
        
        # Step 3: Solution
        step3 = Text("3. 解得摆线", font_size=24, color=RED)
        step3.next_to(el_equation, DOWN, buff=0.5)
        
        solution = MathTex(
            r"x &= a(t - \sin t) \\\n y &= a(1 - \cos t)",
            font_size=24,
            color=BLACK
        )
        solution.next_to(step3, DOWN, buff=0.3)
        
        self.play(Write(step3))
        self.play(Write(solution))
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(step1),
            FadeOut(time_functional),
            FadeOut(step2),
            FadeOut(el_equation),
            FadeOut(step3),
            FadeOut(solution)
        )
        
        self.wait(1)