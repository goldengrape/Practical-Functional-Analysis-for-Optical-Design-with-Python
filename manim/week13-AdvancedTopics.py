from manim import *
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor

class AdvancedTopics(Scene):
    """
    Week 13: Advanced Topics - AI and Functional Analysis Fusion
    
    This animation demonstrates:
    - Neural networks approximating Sobolev spaces
    - AR/VR optical design challenges
    - Future AI-driven design workflows
    """
    
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#0a0a0a"
        
        # Title
        title = Text(
            "Advanced Topics: AI and Functional Analysis Fusion",
            font_size=36,
            color=WHITE
        ).to_edge(UP)
        
        subtitle = Text(
            "Week 13: Future of Optical Design",
            font_size=24,
            color=PURPLE
        ).next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.wait(2)
        
        # Part 1: Neural Networks and Sobolev Spaces
        self.show_neural_sobolev_fusion()
        
        # Part 2: AR/VR Optical Design Challenges
        self.show_ar_vr_challenges()
        
        # Part 3: Future AI-Driven Workflows
        self.show_future_workflows()
        
        # Conclusion
        self.show_final_conclusion()
    
    def show_neural_sobolev_fusion(self):
        """Demonstrate neural networks approximating Sobolev spaces"""
        self.clear_scene_except_title()
        
        section_title = Text(
            "Neural Networks in Sobolev Spaces",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create Sobolev space visualization
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        ).shift(LEFT * 3)
        
        # True function in Sobolev space
        def true_function(x):
            return np.sin(x) + 0.4 * np.sin(2*x) + 0.2 * np.sin(4*x)
        
        true_graph = axes.plot(true_function, color=BLUE, stroke_width=3)
        true_label = Text("f ∈ H¹(Ω)", font_size=16, color=BLUE).next_to(axes, DOWN)
        
        # Neural network approximation
        x_train = np.linspace(-3, 3, 200).reshape(-1, 1)
        y_train = true_function(x_train.flatten())
        
        # Create a more sophisticated neural network
        nn = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='tanh',
            max_iter=2000,
            random_state=42
        )
        nn.fit(x_train, y_train)
        
        x_pred = np.linspace(-3, 3, 300)
        y_pred = nn.predict(x_pred.reshape(-1, 1))
        
        nn_graph = axes.plot_line_graph(x_pred, y_pred, line_color=RED, stroke_width=2)
        nn_label = Text("f_NN ≈ f", font_size=16, color=RED).next_to(true_label, DOWN)
        
        self.play(
            Create(axes),
            Create(true_graph),
            Write(true_label)
        )
        self.wait(1)
        
        self.play(
            Create(nn_graph),
            Write(nn_label)
        )
        self.wait(2)
        
        # Show Sobolev norm comparison
        sobolev_text = Text(
            "Sobolev Norm Approximation:",
            font_size=20,
            color=GREEN
        ).shift(RIGHT * 3 + UP * 2)
        
        # Calculate H¹ norm
        h = 0.01
        x_deriv = np.linspace(-2.9, 2.9, 100)
        
        # True function derivatives
        true_deriv = (true_function(x_deriv + h) - true_function(x_deriv - h)) / (2 * h)
        
        # NN derivatives
        nn_deriv = (nn.predict((x_deriv + h).reshape(-1, 1)) - 
                   nn.predict((x_deriv - h).reshape(-1, 1))) / (2 * h)
        
        # H¹ norms
        true_h1 = np.sqrt(np.mean(true_function(x_deriv)**2) + np.mean(true_deriv**2))
        nn_h1 = np.sqrt(np.mean(nn.predict(x_deriv.reshape(-1, 1))**2) + np.mean(nn_deriv**2))
        
        # Show mathematical formulation
        math_form = MathTex(
            r"\|f\|_{H^1}^2 = \int_\Omega (|f|^2 + |",
            r"\nabla f|^2)",
            r"dx",
            font_size=18
        ).next_to(sobolev_text, DOWN)
        
        approx_form = MathTex(
            r"\approx \int_\Omega (|f_{NN}|^2 + |\partial_x f_{NN}|^2)",
            r"dx",
            font_size=18
        ).next_to(math_form, DOWN)
        
        # Results
        true_result = Text(
            f"True H¹: {true_h1:.4f}",
            font_size=14,
            color=BLUE
        ).next_to(approx_form, DOWN)
        
        nn_result = Text(
            f"NN H¹: {nn_h1:.4f}",
            font_size=14,
            color=RED
        ).next_to(true_result, DOWN)
        
        error_result = Text(
            f"Error: {abs(true_h1 - nn_h1):.4f}",
            font_size=14,
            color=YELLOW
        ).next_to(nn_result, DOWN)
        
        self.play(
            Write(sobolev_text),
            Write(math_form),
            Write(approx_form)
        )
        self.wait(1)
        
        self.play(
            Write(true_result),
            Write(nn_result),
            Write(error_result)
        )
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(axes), FadeOut(true_graph), FadeOut(nn_graph),
            FadeOut(true_label), FadeOut(nn_label),
            FadeOut(sobolev_text), FadeOut(math_form), FadeOut(approx_form),
            FadeOut(true_result), FadeOut(nn_result), FadeOut(error_result)
        )
    
    def show_ar_vr_challenges(self):
        """Demonstrate AR/VR optical design challenges"""
        section_title = Text(
            "AR/VR Optical Design Challenges",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create AR/VR headset visualization
        headset = self.create_ar_vr_headset()
        headset.shift(LEFT * 4)
        
        self.play(Create(headset))
        self.wait(1)
        
        # Show optical challenges
        challenges = [
            "Field of View",
            "Eye Relief",
            "Distortion",
            "Resolution",
            "Comfort"
        ]
        
        challenge_texts = VGroup(*[
            Text(challenge, font_size=18, color=WHITE)
            for challenge in challenges
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        challenge_texts.shift(RIGHT * 3 + UP * 1)
        
        # Add challenge descriptions
        descriptions = [
            "Wide FOV requires complex optics",
            "Large eye box for comfort",
            "Pincushion/barrel correction",
            "Pixel density vs. FOV tradeoff",
            "Weight and form factor"
        ]
        
        desc_texts = VGroup(*[
            Text(desc, font_size=14, color=GRAY)
            for desc in descriptions
        ])
        
        for i, desc_text in enumerate(desc_texts):
            desc_text.next_to(challenge_texts[i], RIGHT, buff=0.5)
        
        # Animate challenges appearing
        for challenge, desc in zip(challenge_texts, desc_texts):
            self.play(Write(challenge), Write(desc))
            self.wait(0.5)
        
        # Show mathematical approach
        approach_text = Text(
            "Functional Analysis Solutions:",
            font_size=20,
            color=GREEN
        ).next_to(challenge_texts, DOWN, buff=0.8)
        
        solutions = [
            "• Optimize over Sobolev spaces",
            "• Multi-objective functional minimization",
            "• Constraint optimization in H¹",
            "• Regularized inverse problems"
        ]
        
        solution_texts = VGroup(*[
            Text(solution, font_size=16, color=GREEN)
            for solution in solutions
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        solution_texts.next_to(approach_text, DOWN, buff=0.3)
        
        self.play(Write(approach_text))
        for solution_text in solution_texts:
            self.play(Write(solution_text))
            self.wait(0.3)
        
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(headset),
            FadeOut(challenge_texts), FadeOut(desc_texts),
            FadeOut(approach_text), FadeOut(solution_texts)
        )
    
    def show_future_workflows(self):
        """Demonstrate future AI-driven design workflows"""
        section_title = Text(
            "Future AI-Driven Design Workflows",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create workflow visualization
        workflow_steps = [
            "1. Problem Definition",
            "2. AI-Assisted Modeling",
            "3. Functional Optimization",
            "4. Uncertainty Quantification",
            "5. Validation & Testing",
            "6. Deployment"
        ]
        
        step_colors = [BLUE, GREEN, YELLOW, ORANGE, RED, PURPLE]
        
        # Create step boxes
        step_boxes = VGroup()
        step_texts = VGroup()
        
        for i, (step, color) in enumerate(zip(workflow_steps, step_colors)):
            box = Rectangle(
                width=3.5,
                height=0.8,
                color=color,
                fill_opacity=0.3,
                stroke_width=2
            )
            
            if i < 3:
                box.shift(UP * (2 - i * 1.5) + LEFT * 4)
            else:
                box.shift(DOWN * (1.5 * (i - 3)) + RIGHT * 4)
            
            text = Text(step, font_size=14, color=WHITE)
            text.move_to(box.get_center())
            
            step_boxes.add(box)
            step_texts.add(text)
        
        # Create connections
        connections = VGroup()
        
        # Connect steps
        for i in range(len(step_boxes) - 1):
            if i < 2:  # First three steps on left
                start = step_boxes[i].get_bottom()
                end = step_boxes[i + 1].get_top()
            elif i == 2:  # Connect left to right side
                start = step_boxes[i].get_right()
                end = step_boxes[i + 1].get_left()
            else:  # Last three steps on right
                start = step_boxes[i].get_bottom()
                end = step_boxes[i + 1].get_top()
            
            arrow = Arrow(start, end, color=WHITE, stroke_width=2)
            connections.add(arrow)
        
        self.play(
            Create(step_boxes),
            Write(step_texts),
            Create(connections),
            run_time=3
        )
        self.wait(2)
        
        # Show AI integration
        ai_text = Text(
            "AI Integration at Every Step:",
            font_size=20,
            color=GREEN
        ).to_edge(DOWN, buff=1.5)
        
        ai_features = [
            "• Natural language interfaces",
            "• Automated model selection",
            "• Intelligent optimization",
            "• Predictive uncertainty",
            "• Autonomous testing",
            "• Continuous learning"
        ]
        
        ai_feature_texts = VGroup(*[
            Text(feature, font_size=12, color=GREEN)
            for feature in ai_features
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        ai_feature_texts.next_to(ai_text, UP, buff=0.3)
        
        self.play(Write(ai_text))
        for feature_text in ai_feature_texts:
            self.play(Write(feature_text))
            self.wait(0.3)
        
        self.wait(3)
        
        # Clear for final section
        self.play(
            FadeOut(section_title),
            FadeOut(step_boxes), FadeOut(step_texts), FadeOut(connections),
            FadeOut(ai_text), FadeOut(ai_feature_texts)
        )
    
    def show_final_conclusion(self):
        """Show final conclusion and summary"""
        # Create comprehensive summary
        summary_title = Text(
            "The Future of Optical Design",
            font_size=32,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(summary_title))
        
        # Key insights
        insights = [
            "Functional Analysis + AI = Revolutionary Design",
            "Sobolev Spaces Enable Precise Optical Modeling",
            "Machine Learning Accelerates Optimization",
            "AR/VR Challenges Drive Innovation",
            "End-to-End AI Workflows are the Future"
        ]
        
        insight_texts = VGroup(*[
            Text(insight, font_size=18, color=WHITE)
            for insight in insights
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        insight_texts.shift(UP * 1)
        
        for insight_text in insight_texts:
            self.play(Write(insight_text))
            self.wait(0.5)
        
        # Final vision
        vision_text = Text(
            "Vision: Autonomous Optical Design Systems",
            font_size=24,
            color=GREEN
        ).next_to(insight_texts, DOWN, buff=1)
        
        vision_points = [
            "• AI understands design requirements",
            "• Functional analysis provides mathematical foundation",
            "• Automated optimization finds optimal solutions",
            "• Continuous learning improves performance",
            "• Human-AI collaboration enhances creativity"
        ]
        
        vision_points_texts = VGroup(*[
            Text(point, font_size=16, color=GREEN)
            for point in vision_points
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        vision_points_texts.next_to(vision_text, DOWN, buff=0.5)
        
        self.play(Write(vision_text))
        for vision_point_text in vision_points_texts:
            self.play(Write(vision_point_text))
            self.wait(0.3)
        
        # Final message
        final_message = Text(
            "The convergence of mathematics, physics, and AI\nwill revolutionize optical design",
            font_size=20,
            color=PURPLE
        ).to_edge(DOWN, buff=1)
        
        self.play(Write(final_message))
        self.wait(5)
    
    def clear_scene_except_title(self):
        """Clear all objects except the main title"""
        objects_to_remove = []
        for obj in self.mobjects:
            if isinstance(obj, (Text, MathTex)):
                if "Advanced Topics" not in obj.text and "Week 13" not in obj.text:
                    objects_to_remove.append(obj)
            else:
                objects_to_remove.append(obj)
        
        if objects_to_remove:
            self.play(*[FadeOut(obj) for obj in objects_to_remove])
    
    def create_ar_vr_headset(self):
        """Create a simple AR/VR headset visualization"""
        headset = VGroup()
        
        # Main body
        body = Rectangle(
            width=4,
            height=2,
            color=GRAY,
            fill_opacity=0.8,
            stroke_width=3
        )
        
        # Lenses
        left_lens = Circle(radius=0.6, color=BLUE, fill_opacity=0.3).shift(LEFT * 1)
        right_lens = Circle(radius=0.6, color=BLUE, fill_opacity=0.3).shift(RIGHT * 1)
        
        # Strap
        strap = Arc(
            radius=2.5,
            start_angle=PI/3,
            angle=PI/3,
            color=GRAY,
            stroke_width=8
        ).shift(UP * 0.5)
        
        headset.add(body, left_lens, right_lens, strap)
        return headset


if __name__ == "__main__":
    # Render the animations
    scene = AdvancedTopics()
    scene.render()