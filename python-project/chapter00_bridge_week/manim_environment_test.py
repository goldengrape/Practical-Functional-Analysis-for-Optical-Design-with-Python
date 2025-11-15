"""
Chapter 0: Bridge Week - Manim Environment Test
Practice Project: Mathematical Animation for Optical Concepts
"""

# This script demonstrates Manim usage for creating mathematical animations
# Note: Manim needs to be installed separately and may require LaTeX

try:
    from manim import *
    MANIM_AVAILABLE = True
except ImportError:
    print("Manim not available. Install with: pip install manim")
    print("Also requires LaTeX installation for text rendering")
    MANIM_AVAILABLE = False


class OpticalManimDemo:
    """Demonstrate Manim usage for optical concepts"""
    
    def __init__(self):
        if not MANIM_AVAILABLE:
            print("Manim is not available. Creating static visualizations instead.")
            self.create_static_demo()
        else:
            print("Manim is available! You can run the animation scenes.")
            self.show_animation_examples()
    
    def create_static_demo(self):
        """Create static visualizations as fallback"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("\n=== Static Visualization Demo ===")
        
        # Create a simple optical diagram
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Light ray propagation
        ax1 = axes[0, 0]
        x = np.linspace(0, 10, 100)
        y1 = np.zeros_like(x)
        y2 = np.sin(0.5 * x)
        
        ax1.plot(x, y1, 'r-', linewidth=2, label='Light ray 1')
        ax1.plot(x, y2, 'b-', linewidth=2, label='Light ray 2')
        ax1.set_title('Light Ray Propagation')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Lens surface
        ax2 = axes[0, 1]
        x_lens = np.linspace(-2, 2, 100)
        y_lens = x_lens**2 / 4  # Parabolic lens
        
        ax2.plot(x_lens, y_lens, 'g-', linewidth=3, label='Lens surface')
        ax2.fill_between(x_lens, 0, y_lens, alpha=0.3, color='green')
        ax2.set_title('Lens Surface Profile')
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Sag')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Wavefront error
        ax3 = axes[1, 0]
        x_wave = np.linspace(-3, 3, 50)
        y_wave = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_wave, y_wave)
        wavefront = 0.5 * (X**2 + Y**2) + 0.2 * np.sin(X) * np.cos(Y)
        
        contour = ax3.contourf(X, Y, wavefront, levels=20, cmap='RdBu')
        ax3.set_title('Wavefront Error Map')
        ax3.set_xlabel('X position')
        ax3.set_ylabel('Y position')
        plt.colorbar(contour, ax=ax3)
        
        # 4. Optimization process
        ax4 = axes[1, 1]
        iterations = np.arange(1, 21)
        error = 10 * np.exp(-iterations/5) + 0.1 * np.random.randn(20)
        
        ax4.plot(iterations, error, 'mo-', linewidth=2, markersize=6)
        ax4.set_title('Optimization Convergence')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Error Metric')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print("Static visualizations created successfully!")
        print("These demonstrate the concepts that would be animated in Manim.")
    
    def show_animation_examples(self):
        """Show examples of Manim animations"""
        print("\n=== Manim Animation Examples ===")
        print("To run these animations, save them as separate Python files and execute:")
        print("manim -pql script_name.py SceneName")
        
        # Example 1: Light Ray Animation
        print("\n1. Light Ray Propagation Animation:")
        print("""
class LightRayAnimation(Scene):
    def construct(self):
        # Create light rays
        ray1 = Line(LEFT*3, RIGHT*3, color=YELLOW, stroke_width=3)
        ray2 = Line(LEFT*3 + UP*0.5, RIGHT*3 + UP*0.5, color=RED, stroke_width=3)
        
        # Create labels
        label1 = Text("Ray 1", font_size=24).next_to(ray1, DOWN)
        label2 = Text("Ray 2", font_size=24).next_to(ray2, UP)
        
        # Animate
        self.play(Create(ray1), Create(ray2))
        self.play(Write(label1), Write(label2))
        self.wait(2)
""")
        
        # Example 2: Lens Surface Creation
        print("\n2. Lens Surface Animation:")
        print("""
class LensSurfaceCreation(Scene):
    def construct(self):
        # Create coordinate axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": BLUE}
        )
        
        # Create lens surface (parabola)
        lens_func = lambda x: x**2 / 4
        lens_graph = axes.plot(lens_func, color=GREEN, stroke_width=4)
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("sag")
        title = Text("Lens Surface Profile", font_size=36).to_edge(UP)
        
        # Animate construction
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(lens_graph))
        self.play(Write(title))
        self.wait(2)
""")
        
        # Example 3: Wavefront Error Visualization
        print("\n3. Wavefront Error Animation:")
        print("""
class WavefrontErrorVisualization(Scene):
    def construct(self):
        # Create a grid of points
        grid = NumberPlane(x_range=[-3, 3, 0.5], y_range=[-3, 3, 0.5])
        
        # Create wavefront error visualization
        colors = [BLUE, GREEN, YELLOW, RED]
        wavefront_group = VGroup()
        
        for i in range(-3, 4):
            for j in range(-3, 4):
                x, y = i * 0.5, j * 0.5
                # Simulate wavefront error
                error = (x**2 + y**2) * 0.5 + 0.1 * np.sin(x * 4) * np.cos(y * 4)
                color_intensity = abs(error) * 10
                color_index = min(int(color_intensity), len(colors) - 1)
                
                dot = Dot(point=[x, y, 0], color=colors[color_index], radius=0.1)
                wavefront_group.add(dot)
        
        title = Text("Wavefront Error Map", font_size=36).to_edge(UP)
        
        self.play(Create(grid))
        self.play(Create(wavefront_group))
        self.play(Write(title))
        self.wait(2)
""")
        
        # Example 4: Optimization Process
        print("\n4. Optimization Convergence Animation:")
        print("""
class OptimizationConvergence(Scene):
    def construct(self):
        # Create axes for error plot
        axes = Axes(
            x_range=[0, 20, 5],
            y_range=[0, 10, 2],
            x_length=8,
            y_length=5,
            axis_config={"color": BLUE}
        )
        
        # Create error data
        iterations = np.arange(0, 21)
        error = 10 * np.exp(-iterations/5) + 0.1 * np.random.randn(21)
        
        # Create the plot
        error_plot = VMobject()
        points = [axes.c2p(x, y) for x, y in zip(iterations, error)]
        error_plot.set_points_as_corners(points)
        error_plot.set_color(RED)
        error_plot.set_stroke(width=3)
        
        # Add labels
        x_label = axes.get_x_axis_label("Iterations")
        y_label = axes.get_y_axis_label("Error")
        title = Text("Optimization Convergence", font_size=36).to_edge(UP)
        
        # Animate
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(error_plot))
        self.play(Write(title))
        self.wait(2)
""")


# If Manim is available, define actual scenes
if MANIM_AVAILABLE:
    class LightRayAnimation(Scene):
        def construct(self):
            # Create light rays
            ray1 = Line(LEFT*3, RIGHT*3, color=YELLOW, stroke_width=3)
            ray2 = Line(LEFT*3 + UP*0.5, RIGHT*3 + UP*0.5, color=RED, stroke_width=3)
            
            # Create labels
            label1 = Text("Ray 1", font_size=24).next_to(ray1, DOWN)
            label2 = Text("Ray 2", font_size=24).next_to(ray2, UP)
            
            # Animate
            self.play(Create(ray1), Create(ray2))
            self.play(Write(label1), Write(label2))
            self.wait(2)
    
    class LensSurfaceCreation(Scene):
        def construct(self):
            # Create coordinate axes
            axes = Axes(
                x_range=[-3, 3, 1],
                y_range=[-1, 3, 1],
                x_length=6,
                y_length=4,
                axis_config={"color": BLUE}
            )
            
            # Create lens surface (parabola)
            lens_func = lambda x: x**2 / 4
            lens_graph = axes.plot(lens_func, color=GREEN, stroke_width=4)
            
            # Add labels
            x_label = axes.get_x_axis_label("x")
            y_label = axes.get_y_axis_label("sag")
            title = Text("Lens Surface Profile", font_size=36).to_edge(UP)
            
            # Animate construction
            self.play(Create(axes), Write(x_label), Write(y_label))
            self.play(Create(lens_graph))
            self.play(Write(title))
            self.wait(2)


def main():
    """Main function to demonstrate Manim usage"""
    print("Chapter 0: Bridge Week - Manim Environment Test")
    print("=" * 50)
    
    # Initialize the demo
    demo = OpticalManimDemo()
    
    print("\n=== Manim Learning Path ===")
    print("1. Install Manim: pip install manim")
    print("2. Install LaTeX (for text rendering)")
    print("3. Test with simple scene: manim -pql script.py SceneName")
    print("4. Explore Manim documentation and examples")
    
    print("\n=== Key Animation Concepts ===")
    print("1. Scene: Container for animations")
    print("2. Mobjects: Mathematical objects to animate")
    print("3. Animations: Transformations applied to Mobjects")
    print("4. Coordinate systems: Define the animation space")
    
    print("\n=== Practice Exercises ===")
    print("1. Create animation of light refraction through lens")
    print("2. Animate the formation of a spherical wavefront")
    print("3. Create optimization process visualization")
    print("4. Combine multiple scenes into a complete optical system demo")


if __name__ == "__main__":
    main()