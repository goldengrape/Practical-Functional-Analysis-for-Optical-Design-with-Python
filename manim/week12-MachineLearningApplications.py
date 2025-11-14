from manim import *
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class MachineLearningApplications(Scene):
    """
    Week 12: Machine Learning Applications in Functional Analysis
    
    This animation demonstrates how machine learning techniques can be applied
    to functional analysis problems in optical design, including:
    - Neural networks approximating Sobolev norms
    - Bayesian uncertainty quantification
    - ML-assisted functional optimization
    """
    
    def construct(self):
        # Set up the scene
        self.camera.background_color = "#1a1a1a"
        
        # Title
        title = Text(
            "Machine Learning Applications in Functional Analysis",
            font_size=36,
            color=WHITE
        ).to_edge(UP)
        
        subtitle = Text(
            "Week 12: AI-Assisted Optical Design",
            font_size=24,
            color=BLUE
        ).next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.wait(2)
        
        # Part 1: Neural Networks Approximating Sobolev Norms
        self.show_neural_approximation()
        
        # Part 2: Bayesian Uncertainty Quantification
        self.show_bayesian_uncertainty()
        
        # Part 3: ML-Assisted Functional Optimization
        self.show_ml_optimization()
        
        # Conclusion
        self.show_conclusion()
    
    def show_neural_approximation(self):
        """Demonstrate neural networks approximating Sobolev norms"""
        # Clear previous content
        self.clear_scene_except_title()
        
        # Title for this section
        section_title = Text(
            "Neural Networks Approximating Sobolev Norms",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create a simple function space visualization
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        ).shift(LEFT * 3)
        
        # True function (target for approximation)
        def true_function(x):
            return np.sin(x) + 0.3 * np.sin(3*x)
        
        true_graph = axes.plot(true_function, color=BLUE, stroke_width=3)
        true_label = Text("True Function f(x)", font_size=16, color=BLUE).next_to(axes, DOWN)
        
        # Neural network approximation
        # Train a simple neural network
        x_train = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_train = true_function(x_train.flatten())
        
        nn = MLPRegressor(hidden_layer_sizes=(20, 15, 10), max_iter=1000, random_state=42)
        nn.fit(x_train, y_train)
        
        # Neural network prediction
        x_pred = np.linspace(-3, 3, 200)
        y_pred = nn.predict(x_pred.reshape(-1, 1))
        
        nn_graph = axes.plot_line_graph(x_pred, y_pred, line_color=RED, stroke_width=2)
        nn_label = Text("NN Approximation", font_size=16, color=RED).next_to(true_label, DOWN)
        
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
        
        # Show Sobolev norm approximation
        sobolev_text = Text(
            "Sobolev Norm Approximation:",
            font_size=20,
            color=GREEN
        ).shift(RIGHT * 3 + UP * 2)
        
        # Calculate Sobolev norm approximation
        h = 0.01
        x_deriv = np.linspace(-2.9, 2.9, 100)
        
        # True derivative
        true_deriv = np.cos(x_deriv) + 0.9 * np.cos(3*x_deriv)
        
        # NN derivative (numerical)
        nn_deriv = (nn.predict((x_deriv + h).reshape(-1, 1)) - 
                   nn.predict((x_deriv - h).reshape(-1, 1))) / (2 * h)
        
        # Sobolev norm (H1)
        true_sobolev = np.sqrt(np.mean(true_function(x_deriv)**2) + np.mean(true_deriv**2))
        nn_sobolev = np.sqrt(np.mean(nn.predict(x_deriv.reshape(-1, 1))**2) + np.mean(nn_deriv**2))
        
        sobolev_formula = MathTex(
            r"\|f\|_{H^1}^2 \approx \|f_{NN}\|_{H^1}^2 = \int (f_{NN}^2 + (f_{NN}')^2) dx",
            font_size=24
        ).next_to(sobolev_text, DOWN)
        
        true_norm_text = Text(
            f"True H¹ norm: {true_sobolev:.3f}",
            font_size=16,
            color=BLUE
        ).next_to(sobolev_formula, DOWN)
        
        nn_norm_text = Text(
            f"NN H¹ norm: {nn_sobolev:.3f}",
            font_size=16,
            color=RED
        ).next_to(true_norm_text, DOWN)
        
        error_text = Text(
            f"Error: {abs(true_sobolev - nn_sobolev):.3f}",
            font_size=16,
            color=YELLOW
        ).next_to(nn_norm_text, DOWN)
        
        self.play(
            Write(sobolev_text),
            Write(sobolev_formula)
        )
        self.wait(1)
        
        self.play(
            Write(true_norm_text),
            Write(nn_norm_text),
            Write(error_text)
        )
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(axes), FadeOut(true_graph), FadeOut(nn_graph),
            FadeOut(true_label), FadeOut(nn_label),
            FadeOut(sobolev_text), FadeOut(sobolev_formula),
            FadeOut(true_norm_text), FadeOut(nn_norm_text), FadeOut(error_text)
        )
    
    def show_bayesian_uncertainty(self):
        """Demonstrate Bayesian uncertainty quantification"""
        section_title = Text(
            "Bayesian Uncertainty Quantification",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create a simple Bayesian inference example
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        ).shift(DOWN * 0.5)
        
        # Prior distribution
        def prior(x):
            return multivariate_normal.pdf(x, mean=0, cov=0.5)
        
        prior_graph = axes.plot(prior, color=BLUE, stroke_width=3)
        prior_label = Text("Prior p(θ)", font_size=16, color=BLUE).next_to(axes, UP)
        
        # Likelihood (data)
        x_data = np.array([-0.8, -0.2, 0.3, 0.9])
        y_data = np.array([0.2, 0.8, 0.7, 0.1])
        
        # Plot data points
        data_points = VGroup()
        for x, y in zip(x_data, y_data):
            point = Dot(axes.c2p(x, y), color=YELLOW, radius=0.08)
            data_points.add(point)
        
        data_label = Text("Data", font_size=16, color=YELLOW).next_to(prior_label, RIGHT)
        
        self.play(
            Create(axes),
            Create(prior_graph),
            Write(prior_label)
        )
        self.wait(1)
        
        self.play(
            Create(data_points),
            Write(data_label)
        )
        self.wait(1)
        
        # Posterior distribution (approximate)
        # Simple Bayesian update for normal distribution
        posterior_mean = np.mean(x_data)
        posterior_var = 0.1  # Simplified
        
        def posterior(x):
            return multivariate_normal.pdf(x, mean=posterior_mean, cov=posterior_var)
        
        posterior_graph = axes.plot(posterior, color=GREEN, stroke_width=3)
        posterior_label = Text("Posterior p(θ|D)", font_size=16, color=GREEN).next_to(data_label, RIGHT)
        
        self.play(
            Create(posterior_graph),
            Write(posterior_label)
        )
        self.wait(2)
        
        # Show uncertainty bands
        uncertainty_text = Text(
            "Uncertainty Quantification:",
            font_size=20,
            color=WHITE
        ).shift(UP * 2.5 + LEFT * 3)
        
        # Show confidence intervals
        ci_lower = posterior_mean - 1.96 * np.sqrt(posterior_var)
        ci_upper = posterior_mean + 1.96 * np.sqrt(posterior_var)
        
        ci_line = Line(
            axes.c2p(ci_lower, 0.8),
            axes.c2p(ci_upper, 0.8),
            color=RED,
            stroke_width=8
        )
        
        ci_text = Text(
            "95% Confidence Interval",
            font_size=14,
            color=RED
        ).next_to(ci_line, UP)
        
        self.play(
            Write(uncertainty_text),
            Create(ci_line),
            Write(ci_text)
        )
        self.wait(2)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(axes), FadeOut(prior_graph), FadeOut(posterior_graph),
            FadeOut(prior_label), FadeOut(data_label), FadeOut(posterior_label),
            FadeOut(data_points), FadeOut(uncertainty_text),
            FadeOut(ci_line), FadeOut(ci_text)
        )
    
    def show_ml_optimization(self):
        """Demonstrate ML-assisted functional optimization"""
        section_title = Text(
            "ML-Assisted Functional Optimization",
            font_size=28,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        self.play(Write(section_title))
        
        # Create optimization landscape
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[0, 3, 0.5],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"color": WHITE},
            tips=False
        ).shift(LEFT * 2)
        
        # Complex optimization landscape
        def objective(x, y):
            return (x**2 + y**2 - 1)**2 + 0.5 * x * y + 0.1 * np.sin(3*x) * np.cos(3*y)
        
        # Create surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, objective(u, v)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=30,
            fill_opacity=0.7,
            fill_color=BLUE,
            stroke_color=BLUE_E,
            stroke_width=0.5
        )
        
        self.play(
            Create(axes),
            Create(surface)
        )
        self.wait(1)
        
        # Traditional optimization path
        def traditional_optimize():
            # Gradient descent
            x, y = 1.5, -1.5
            path = []
            learning_rate = 0.01
            
            for i in range(100):
                path.append([x, y, objective(x, y)])
                
                # Numerical gradient
                dx = (objective(x + 1e-6, y) - objective(x - 1e-6, y)) / 2e-6
                dy = (objective(x, y + 1e-6) - objective(x, y - 1e-6)) / 2e-6
                
                x -= learning_rate * dx
                y -= learning_rate * dy
                
                if np.sqrt(dx**2 + dy**2) < 1e-4:
                    break
            
            return np.array(path)
        
        traditional_path = traditional_optimize()
        
        # ML-assisted optimization
        def ml_optimize():
            # Use neural network to predict good starting points
            # Train on some sample points
            x_train = np.random.uniform(-2, 2, (50, 2))
            y_train = np.array([objective(x[0], x[1]) for x in x_train])
            
            # Find best point
            best_idx = np.argmin(y_train)
            best_x, best_y = x_train[best_idx]
            
            # Start gradient descent from best point
            x, y = best_x, best_y
            path = []
            learning_rate = 0.01
            
            for i in range(50):
                path.append([x, y, objective(x, y)])
                
                dx = (objective(x + 1e-6, y) - objective(x - 1e-6, y)) / 2e-6
                dy = (objective(x, y + 1e-6) - objective(x, y - 1e-6)) / 2e-6
                
                x -= learning_rate * dx
                y -= learning_rate * dy
                
                if np.sqrt(dx**2 + dy**2) < 1e-4:
                    break
            
            return np.array(path)
        
        ml_path = ml_optimize()
        
        # Create path visualizations
        traditional_dots = VGroup(*[
            Dot3D(axes.c2p(point[0], point[1], point[2]), color=RED, radius=0.05)
            for point in traditional_path[::5]
        ])
        
        ml_dots = VGroup(*[
            Dot3D(axes.c2p(point[0], point[1], point[2]), color=GREEN, radius=0.05)
            for point in ml_path[::3]
        ])
        
        # Labels
        traditional_label = Text(
            "Traditional GD\n(100 iterations)",
            font_size=14,
            color=RED
        ).shift(RIGHT * 4 + UP * 2)
        
        ml_label = Text(
            "ML-Assisted\n(50 iterations)",
            font_size=14,
            color=GREEN
        ).next_to(traditional_label, DOWN, buff=0.5)
        
        self.play(
            Create(traditional_dots),
            Write(traditional_label)
        )
        self.wait(1)
        
        self.play(
            Create(ml_dots),
            Write(ml_label)
        )
        self.wait(2)
        
        # Performance comparison
        traditional_final = traditional_path[-1]
        ml_final = ml_path[-1]
        
        comparison_text = Text(
            "Final Results:",
            font_size=18,
            color=WHITE
        ).shift(RIGHT * 4 + DOWN * 1)
        
        traditional_result = Text(
            f"Traditional: {traditional_final[2]:.4f}",
            font_size=14,
            color=RED
        ).next_to(comparison_text, DOWN, buff=0.3)
        
        ml_result = Text(
            f"ML-Assisted: {ml_final[2]:.4f}",
            font_size=14,
            color=GREEN
        ).next_to(traditional_result, DOWN, buff=0.2)
        
        speedup_text = Text(
            f"Speedup: {len(traditional_path)/len(ml_path):.1f}x",
            font_size=14,
            color=YELLOW
        ).next_to(ml_result, DOWN, buff=0.2)
        
        self.play(
            Write(comparison_text),
            Write(traditional_result),
            Write(ml_result),
            Write(speedup_text)
        )
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(axes), FadeOut(surface),
            FadeOut(traditional_dots), FadeOut(ml_dots),
            FadeOut(traditional_label), FadeOut(ml_label),
            FadeOut(comparison_text), FadeOut(traditional_result),
            FadeOut(ml_result), FadeOut(speedup_text)
        )
    
    def show_conclusion(self):
        """Show conclusion and summary"""
        # Create summary text
        summary_text = Text(
            "Machine Learning + Functional Analysis",
            font_size=32,
            color=YELLOW
        ).to_edge(UP, buff=1)
        
        # Key points
        points = [
            "• Neural networks can approximate Sobolev norms",
            "• Bayesian methods quantify uncertainty in functional spaces",
            "• ML-assisted optimization accelerates functional minimization",
            "• AI enables real-time optical design optimization",
            "• Future: End-to-end AI-driven design workflows"
        ]
        
        point_texts = VGroup(*[
            Text(point, font_size=18, color=WHITE)
            for point in points
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        point_texts.next_to(summary_text, DOWN, buff=1)
        
        self.play(Write(summary_text))
        self.wait(1)
        
        for point_text in point_texts:
            self.play(Write(point_text))
            self.wait(0.5)
        
        self.wait(3)
        
        # Final message
        final_message = Text(
            "AI + Functional Analysis = Next-Gen Optical Design",
            font_size=24,
            color=GREEN
        ).to_edge(DOWN, buff=1)
        
        self.play(Write(final_message))
        self.wait(4)
    
    def clear_scene_except_title(self):
        """Clear all objects except the main title"""
        objects_to_remove = []
        for obj in self.mobjects:
            if isinstance(obj, (Text, MathTex)):
                if "Machine Learning Applications" not in obj.text and "Week 12" not in obj.text:
                    objects_to_remove.append(obj)
            else:
                objects_to_remove.append(obj)
        
        if objects_to_remove:
            self.play(*[FadeOut(obj) for obj in objects_to_remove])


class NeuralNetworkArchitecture(Scene):
    """Detailed visualization of neural network architecture for functional approximation"""
    
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
        title = Text(
            "Neural Network Architecture for Functional Approximation",
            font_size=28,
            color=WHITE
        ).to_edge(UP)
        
        self.play(Write(title))
        
        # Create neural network visualization
        layers = [3, 8, 6, 4, 1]  # Input, hidden, output layers
        layer_colors = [BLUE, GREEN, YELLOW, RED, PURPLE]
        
        # Create neurons
        neurons = VGroup()
        positions = []
        
        for i, layer_size in enumerate(layers):
            layer_neurons = VGroup()
            layer_pos = []
            
            for j in range(layer_size):
                neuron = Circle(radius=0.2, color=layer_colors[i], fill_opacity=0.8)
                y_pos = (layer_size - 1) * 0.4 / 2 - j * 0.4
                neuron.move_to([i * 2 - 4, y_pos, 0])
                layer_neurons.add(neuron)
                layer_pos.append([i * 2 - 4, y_pos, 0])
            
            neurons.add(layer_neurons)
            positions.append(layer_pos)
        
        # Create connections
        connections = VGroup()
        for i in range(len(layers) - 1):
            for j in range(layers[i]):
                for k in range(layers[i + 1]):
                    line = Line(
                        positions[i][j],
                        positions[i + 1][k],
                        stroke_width=1,
                        color=GRAY,
                        opacity=0.6
                    )
                    connections.add(line)
        
        self.play(
            Create(connections),
            Create(neurons)
        )
        self.wait(2)
        
        # Add labels
        input_label = Text("Input Layer\n(Function values)", font_size=14, color=BLUE)
        input_label.next_to(neurons[0], LEFT)
        
        hidden1_label = Text("Hidden Layer 1\n(Feature extraction)", font_size=14, color=GREEN)
        hidden1_label.next_to(neurons[1], UP)
        
        output_label = Text("Output Layer\n(Sobolev norm)", font_size=14, color=PURPLE)
        output_label.next_to(neurons[-1], RIGHT)
        
        self.play(
            Write(input_label),
            Write(hidden1_label),
            Write(output_label)
        )
        self.wait(3)


class BayesianUpdateAnimation(Scene):
    """Animation showing Bayesian update process for functional parameters"""
    
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
        title = Text(
            "Bayesian Update in Functional Parameter Space",
            font_size=28,
            color=WHITE
        ).to_edge(UP)
        
        self.play(Write(title))
        
        # Create parameter space visualization
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 0.8, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        ).shift(DOWN * 0.5)
        
        # Initial prior
        def prior(x):
            return multivariate_normal.pdf(x, mean=0, cov=1.0)
        
        prior_graph = axes.plot(prior, color=BLUE, stroke_width=3)
        prior_label = Text("Prior p(θ)", font_size=16, color=BLUE).next_to(axes, UP)
        
        self.play(
            Create(axes),
            Create(prior_graph),
            Write(prior_label)
        )
        self.wait(1)
        
        # Simulate data collection
        data_points = []
        true_param = 0.5
        
        for i in range(5):
            # Generate noisy observation
            obs = true_param + np.random.normal(0, 0.3)
            data_points.append(obs)
            
            # Update posterior
            posterior_mean = np.mean(data_points)
            posterior_var = 0.3 / (i + 1)  # Simplified update
            
            def posterior(x):
                return multivariate_normal.pdf(x, mean=posterior_mean, cov=posterior_var)
            
            posterior_graph = axes.plot(posterior, color=GREEN, stroke_width=3)
            
            # Show data point
            data_dot = Dot(axes.c2p(obs, 0.1), color=YELLOW, radius=0.08)
            data_label = Text(f"Data {i+1}", font_size=12, color=YELLOW)
            data_label.next_to(data_dot, UP)
            
            self.play(
                Create(data_dot),
                Write(data_label),
                Transform(prior_graph, posterior_graph)
            )
            self.wait(0.5)
        
        # Final posterior
        final_posterior = Text(
            "Posterior: p(θ|D₁,D₂,D₃,D₄,D₅)",
            font_size=18,
            color=GREEN
        ).next_to(axes, DOWN)
        
        self.play(Write(final_posterior))
        self.wait(3)


if __name__ == "__main__":
    # Render the animations
    scenes = [
        MachineLearningApplications(),
        NeuralNetworkArchitecture(),
        BayesianUpdateAnimation()
    ]
    
    for scene in scenes:
        scene.render()