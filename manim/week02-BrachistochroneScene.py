# manim -pql scene.py BrachistochroneScene

from manim import *
from scipy.optimize import newton
import numpy as np

# --- Configuration & Style ---
# Adhering to the principle of separating configuration from logic.
config.background_color = "#1E1E1E"
TEXT_COLOR = WHITE
ACCENT_COLOR = YELLOW
LINE_COLOR = RED_C
ARC_COLOR = GREEN_C
CYCLOID_COLOR = BLUE_C
GRAVITY = 9.81

# --- Core Logic Functions (Data-Oriented & Functional) ---

def calculate_travel_time(path: VMobject, g: float = GRAVITY) -> float:
    """
    Calculates the travel time of a bead along a given path under gravity.
    This function is a pure data transformation: Path -> Time.
    It embodies the functional principle of mapping input data (path geometry)
    to output data (a scalar time value) without side effects.

    The physics is based on v = sqrt(2*g*y), where y is the vertical distance fallen.
    The total time is the integral of ds/v.

    Args:
        path (VMobject): The Manim path object representing the curve.
        g (float): The acceleration due to gravity.

    Returns:
        float: The total calculated time for a bead to traverse the path.
    """
    # Get the discrete points that make up the Manim path object.
    points = path.get_points()
    total_time = 0.0

    # The y-coordinate of the start point (highest point).
    # We assume the y-axis is inverted, so the start_y is the minimum y-value.
    start_y = points[0][1]

    # Iterate over each small segment of the path.
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        # Calculate the length of the segment (ds).
        segment_length = np.linalg.norm(p2 - p1)
        
        # If segment length is zero, it contributes no time.
        if segment_length < 1e-9:
            continue

        # Calculate the average y-coordinate for this segment.
        midpoint_y = (p1[1] + p2[1]) / 2
        
        # The vertical distance fallen is the difference from the start.
        vertical_drop = midpoint_y - start_y
        
        # To avoid division by zero or sqrt of negative number at the very start,
        # we add a small epsilon. This is an application of Occam's Razor:
        # the simplest solution to a numerical instability.
        if vertical_drop <= 1e-9:
            vertical_drop = 1e-9
        
        # Calculate velocity at the midpoint based on energy conservation.
        velocity = np.sqrt(2 * g * vertical_drop)

        # Time for this segment is dt = ds / v.
        segment_time = segment_length / velocity
        
        # Accumulate the total time.
        total_time += segment_time

    return total_time

def get_cycloid_path(axes: Axes, start_point: np.ndarray, end_point: np.ndarray) -> VMobject:
    """
    Generates the cycloid path (the true Brachistochrone curve).
    This function encapsulates the complex mathematics of the cycloid,
    adhering to the Axiomatic Design principle of functional independence.
    Its sole purpose is to create the correct geometric path.

    Args:
        axes (Axes): The Manim axes object for coordinate conversion.
        start_point (np.ndarray): The starting coordinates (x, y).
        end_point (np.ndarray): The ending coordinates (x, y).

    Returns:
        VMobject: A Manim path object for the cycloid.
    """
    x_start, y_start, _ = start_point
    x_end, y_end, _ = end_point

    # The cycloid must pass through (x_end - x_start, y_end - y_start).
    # We need to solve B_y / B_x = (1 - cos(theta)) / (theta - sin(theta))
    # for theta numerically to find the correct radius R.
    def f(theta: float) -> float:
        return (y_end - y_start) / (x_end - x_start) - (1 - np.cos(theta)) / (theta - np.sin(theta))

    # Use a numerical solver to find the root theta_end.
    # Start the search around pi, a reasonable guess.
    theta_end = newton(f, np.pi) 

    # Once theta_end is found, calculate the radius of the generating circle.
    R = (y_end - y_start) / (1 - np.cos(theta_end))

    # Parametric equations for the cycloid, shifted to the start point.
    cycloid_func = lambda t: axes.c2p(
        x_start + R * (t - np.sin(t)),
        y_start + R * (1 - np.cos(t))
    )
    
    # Create the path from theta=0 to theta=theta_end.
    return ParametricFunction(cycloid_func, t_range=[0, theta_end], color=CYCLOID_COLOR)


# --- Manim Scene ---

class BrachistochroneScene(Scene):
    """
    A Manim scene to visually demonstrate the Brachistochrone problem.
    The animation is structured as a narrative:
    1.  Problem setup.
    2.  Introduction of candidate paths.
    3.  A race between beads on each path.
    4.  Analysis of the result.
    """
    def construct(self):
        # 1. Setup the visual environment
        self.setup_problem()
        
        # 2. Define and draw the competing paths
        paths, path_labels = self.introduce_paths()
        
        # 3. Calculate times and run the race animation
        self.run_race(paths)

        # 4. Show the results and provide the core intuition
        self.show_results(paths, path_labels)
        
    def setup_problem(self):
        """Sets up the axes, points, and problem statement."""
        # Create axes with an inverted y-axis, which is more natural for gravity problems.
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            axis_config={"color": BLUE},
            x_length=10,
            y_length=6
        ).add_coordinates().invert_yaxis()
        
        self.play(Create(self.axes))
        
        # Define start and end points in graph coordinates.
        self.start_coord = np.array([1, 1])
        self.end_coord = np.array([9, 5])
        
        # Convert to screen coordinates.
        start_pos = self.axes.c2p(*self.start_coord)
        end_pos = self.axes.c2p(*self.end_coord)

        dot_a = Dot(start_pos, color=ACCENT_COLOR, radius=0.12).set_z_index(10)
        dot_b = Dot(end_pos, color=ACCENT_COLOR, radius=0.12).set_z_index(10)
        label_a = MathTex("A").next_to(dot_a, UL)
        label_b = MathTex("B").next_to(dot_b, DR)
        
        title = Text("最速降线问题 (The Brachistochrone Problem)", font_size=36).to_edge(UP)
        question = Text("找到从 A 到 B 在重力作用下最快的路径", font_size=24).next_to(title, DOWN)

        self.play(
            Write(title),
            FadeIn(dot_a, label_a, dot_b, label_b)
        )
        self.play(Write(question))
        self.wait(1)
        self.play(FadeOut(question))

    def introduce_paths(self) -> tuple[VGroup, VGroup]:
        """Creates and labels the three competing paths."""
        # Define paths using the axes coordinate system
        line_path = self.axes.plot_line(self.start_coord, self.end_coord, color=LINE_COLOR)
        
        # The circular arc is a segment of a circle passing through A and B
        arc_path = self.axes.plot(
            lambda x: 1 + np.sqrt(4**2 - (x-5)**2), 
            x_range=[1, 9], 
            color=ARC_COLOR
        )
        
        cycloid_path = get_cycloid_path(self.axes, self.start_coord, self.end_coord)
        
        # Create labels for each path
        line_label = Text("直线 (Line)", color=LINE_COLOR, font_size=24).next_to(line_path, DOWN)
        arc_label = Text("圆弧 (Arc)", color=ARC_COLOR, font_size=24).next_to(self.axes.c2p(5, 5.2), UP)
        cycloid_label = Text("摆线 (Cycloid)", color=CYCLOID_COLOR, font_size=24).next_to(cycloid_path, DOWN, buff=0.4)
        
        paths = VGroup(line_path, arc_path, cycloid_path)
        path_labels = VGroup(line_label, arc_label, cycloid_label)

        self.play(
            Create(line_path), Write(line_label),
            Create(arc_path), Write(arc_label),
            Create(cycloid_path), Write(cycloid_label)
        )
        self.wait(1)
        return paths, path_labels

    def run_race(self, paths: VGroup):
        """Simulates the race by moving beads along each path according to calculated time."""
        # Calculate the real-world travel time for each path
        times = [calculate_travel_time(p) for p in paths]
        
        # Create a bead for each path
        beads = VGroup(*[Dot(radius=0.1, color=p.get_color()).move_to(p.get_start()) for p in paths])
        beads.set_z_index(5)

        # Create the animations. The `run_time` of each animation is set
        # to the physically calculated travel time. This is the core of the simulation.
        animations = AnimationGroup(
            *[MoveAlongPath(beads[i], paths[i], run_time=times[i], rate_func=linear) for i in range(len(paths))]
        )
        
        race_title = Text("开始比赛!", font_size=32).to_edge(UP, buff=1.5)
        self.play(Write(race_title))
        self.play(FadeIn(beads))
        self.play(animations)
        self.wait(1)
        
        # Store results for the final scene
        self.times = times
        self.play(FadeOut(race_title), FadeOut(beads))

    def show_results(self, paths: VGroup, path_labels: VGroup):
        """Displays the calculated times and explains the result."""
        winner_index = np.argmin(self.times)
        
        # Display the times for each path
        time_texts = VGroup()
        for i, time in enumerate(self.times):
            text = Text(f"T = {time:.2f} s", color=paths[i].get_color(), font_size=28)
            text.next_to(path_labels[i], RIGHT, buff=0.5)
            time_texts.add(text)
        
        self.play(Write(time_texts))
        self.wait(1)

        # Highlight the winner
        winner_box = SurroundingRectangle(VGroup(paths[winner_index], path_labels[winner_index], time_texts[winner_index]), color=ACCENT_COLOR)
        winner_text = Text("Winner!", color=ACCENT_COLOR).next_to(winner_box, UP)
        self.play(Create(winner_box), Write(winner_text))
        self.wait(1)

        # Explain the intuition
        explanation = VGroup(
            Text("为什么摆线最快?", font_size=32),
            Text("它在开始时最陡峭，可以最快地将势能转化为动能。", font_size=24),
            Text("Builds speed earliest!", font_size=24, color=ACCENT_COLOR),
        ).arrange(DOWN).to_corner(UL, buff=0.5)
        
        # Highlight the steep initial part of the cycloid
        initial_segment = ParametricFunction(
            paths[winner_index].func,
            t_range=[0, 0.5], # Show just the first part
            color=YELLOW,
            stroke_width=10
        )
        
        self.play(Write(explanation))
        self.play(Create(initial_segment))
        self.wait(3)