"""
Chapter 13: AI Integration in Optical Design
Practical Functional Analysis for Optical Design with Python

This module implements AI and machine learning methods for optical design,
including neural networks, Gaussian processes, reinforcement learning, and
hybrid AI-traditional optimization approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats, interpolate
from scipy.special import erf, gamma
import sympy as sp
from typing import Tuple, Callable, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class AIOpticalDesign:
    """AI and machine learning methods for optical design."""
    
    def __init__(self):
        self.rng = np.random.RandomState(42)
        
    def neural_network_lens_design(self, training_data: Optional[Dict] = None) -> Dict:
        """
        Neural network for lens design parameter prediction.
        
        Implements a simple feedforward network for optical design.
        """
        if training_data is None:
            # Generate synthetic training data
            n_samples = 1000
            
            # Input features: [focal_length_target, f_number, field_of_view, wavelength]
            # Output: [radius_1, radius_2, thickness, refractive_index]
            
            X = np.zeros((n_samples, 4))
            y = np.zeros((n_samples, 4))
            
            for i in range(n_samples):
                # Random design specifications
                focal_length = self.rng.uniform(10, 200)  # mm
                f_number = self.rng.uniform(2, 8)
                fov = self.rng.uniform(5, 30)  # degrees
                wavelength = self.rng.uniform(0.4, 0.8)  # microns
                
                X[i] = [focal_length, f_number, fov, wavelength]
                
                # Generate corresponding lens parameters (with some physics-based constraints)
                # This is a simplified model for demonstration
                r1 = focal_length * (n := self.rng.uniform(1.4, 1.8)) / (n - 1) * (0.8 + 0.4 * self.rng.random())
                r2 = -r1 * (0.6 + 0.8 * self.rng.random())
                thickness = focal_length * 0.05 * (0.5 + self.rng.random())
                
                y[i] = [r1, r2, thickness, n]
        else:
            X = training_data['X']
            y = training_data['y']
        
        # Simple neural network implementation
        class SimpleNeuralNetwork:
            def __init__(self, layer_sizes: List[int]):
                self.layer_sizes = layer_sizes
                self.weights = []
                self.biases = []
                
                # Initialize weights and biases
                for i in range(len(layer_sizes) - 1):
                    w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
                    b = np.zeros(layer_sizes[i+1])
                    self.weights.append(w)
                    self.biases.append(b)
            
            def sigmoid(self, x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
            def sigmoid_derivative(self, x):
                s = self.sigmoid(x)
                return s * (1 - s)
            
            def forward(self, X):
                self.activations = [X]
                self.z_values = []
                
                current_input = X
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    z = np.dot(current_input, w) + b
                    self.z_values.append(z)
                    
                    if i < len(self.weights) - 1:  # Hidden layers
                        current_input = self.sigmoid(z)
                    else:  # Output layer (linear)
                        current_input = z
                    
                    self.activations.append(current_input)
                
                return current_input
            
            def backward(self, X, y, learning_rate=0.01):
                m = X.shape[0]
                
                # Forward pass
                output = self.forward(X)
                
                # Compute loss gradient
                delta = output - y
                
                # Backpropagation
                deltas = [delta]
                
                for i in range(len(self.weights) - 1, 0, -1):
                    delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])
                    deltas.append(delta)
                
                deltas.reverse()
                
                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
                    self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0)
                
                # Return loss
                return np.mean(delta**2)
        
        # Create and train network
        nn = SimpleNeuralNetwork([4, 10, 8, 4])
        
        # Training loop
        losses = []
        for epoch in range(1000):
            loss = nn.backward(X, y, learning_rate=0.01)
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        # Test prediction
        test_spec = np.array([[50, 4, 10, 0.55]])  # focal_length=50mm, f/4, 10° FOV, 0.55μm
        predicted_params = nn.forward(test_spec)
        
        results = {
            'neural_network': nn,
            'training_losses': losses,
            'test_prediction': {
                'specifications': test_spec[0],
                'predicted_parameters': predicted_params[0],
                'parameter_names': ['radius_1', 'radius_2', 'thickness', 'refractive_index']
            },
            'training_data_stats': {
                'n_samples': X.shape[0],
                'input_features': ['focal_length', 'f_number', 'field_of_view', 'wavelength'],
                'output_parameters': ['radius_1', 'radius_2', 'thickness', 'refractive_index']
            }
        }
        
        return results
    
    def gaussian_process_optical_modeling(self, training_points: Optional[np.ndarray] = None) -> Dict:
        """
        Gaussian process regression for optical system modeling.
        
        Provides uncertainty estimates for optical performance predictions.
        """
        if training_points is None:
            # Generate training data for optical surface modeling
            n_train = 50
            
            # Sample points on optical surface
            x_train = np.linspace(-10, 10, n_train)
            y_train = np.linspace(-10, 10, n_train)
            X_train, Y_train = np.meshgrid(x_train, y_train)
            
            # True surface (aspheric lens)
            R = np.sqrt(X_train**2 + Y_train**2)
            k = -1.0  # conic constant
            A4 = 1e-6   # 4th order coefficient
            A6 = 1e-9   # 6th order coefficient
            
            Z_true = R**2 / (1 + np.sqrt(1 - (1 + k) * R**2)) + A4 * R**4 + A6 * R**6
            
            # Add measurement noise
            noise_std = 0.01
            Z_train = Z_true + np.random.normal(0, noise_std, Z_true.shape)
            
            # Flatten for GP
            X_flat = np.column_stack([X_train.ravel(), Y_train.ravel()])
            Z_flat = Z_train.ravel()
            
        else:
            X_flat = training_points[:, :2]
            Z_flat = training_points[:, 2]
        
        # Gaussian Process implementation
        class GaussianProcess:
            def __init__(self, length_scale=1.0, noise_level=1e-10):
                self.length_scale = length_scale
                self.noise_level = noise_level
                self.X_train = None
                self.y_train = None
                self.K_inv = None
                self.alpha = None
            
            def rbf_kernel(self, X1, X2):
                """Radial basis function kernel."""
                sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
                return np.exp(-sqdist / (2 * self.length_scale**2))
            
            def fit(self, X, y):
                """Fit the Gaussian process."""
                self.X_train = X
                self.y_train = y
                
                # Compute kernel matrix
                K = self.rbf_kernel(X, X)
                K += self.noise_level * np.eye(len(X))
                
                # Cholesky decomposition for numerical stability
                try:
                    L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
                except np.linalg.LinAlgError:
                    # Fallback to direct inversion
                    self.K_inv = np.linalg.inv(K)
                    self.alpha = np.dot(self.K_inv, y)
            
            def predict(self, X_test, return_std=True):
                """Make predictions with uncertainty estimates."""
                K_star = self.rbf_kernel(X_test, self.X_train)
                y_mean = np.dot(K_star, self.alpha)
                
                if return_std:
                    # Predictive variance
                    v = np.dot(self.K_inv, K_star.T)
                    y_var = self.rbf_kernel(X_test, X_test).diagonal() - np.sum(K_star * v.T, axis=1)
                    y_var = np.maximum(y_var, 0)  # Ensure non-negative
                    y_std = np.sqrt(y_var)
                    return y_mean, y_std
                
                return y_mean
        
        # Create and fit GP
        gp = GaussianProcess(length_scale=5.0, noise_level=noise_std**2)
        gp.fit(X_flat, Z_flat)
        
        # Test predictions
        x_test = np.linspace(-12, 12, 30)
        y_test = np.linspace(-12, 12, 30)
        X_test, Y_test = np.meshgrid(x_test, y_test)
        X_test_flat = np.column_stack([X_test.ravel(), Y_test.ravel()])
        
        Z_pred, Z_std = gp.predict(X_test_flat, return_std=True)
        Z_pred = Z_pred.reshape(X_test.shape)
        Z_std = Z_std.reshape(X_test.shape)
        
        # Calculate performance metrics
        Z_pred_train, _ = gp.predict(X_flat, return_std=True)
        mse = np.mean((Z_pred_train - Z_flat)**2)
        r2 = 1 - np.sum((Z_flat - Z_pred_train)**2) / np.sum((Z_flat - np.mean(Z_flat))**2)
        
        results = {
            'gaussian_process': gp,
            'predictions': {
                'mean': Z_pred,
                'std': Z_std,
                'test_grid': (X_test, Y_test)
            },
            'training_data': {
                'X': X_flat,
                'y': Z_flat,
                'original_grid': (X_train, Y_train, Z_train)
            },
            'performance_metrics': {
                'mse': mse,
                'r2': r2,
                'noise_estimate': np.sqrt(gp.noise_level)
            },
            'hyperparameters': {
                'length_scale': gp.length_scale,
                'noise_level': gp.noise_level
            }
        }
        
        return results
    
    def reinforcement_learning_optimization(self, environment_params: Optional[Dict] = None) -> Dict:
        """
        Reinforcement learning for optical system optimization.
        
        Uses Q-learning to optimize lens configurations.
        """
        if environment_params is None:
            environment_params = {
                'lens_types': ['convex', 'concave', 'plano_convex', 'plano_concave'],
                'materials': [1.5, 1.6, 1.7, 1.8],  # refractive indices
                'curvature_range': (-0.1, 0.1),  # 1/mm
                'thickness_range': (1, 10),  # mm
                'target_focal_length': 100.0,  # mm
                'wavelength': 0.55  # microns
            }
        
        # Simple optical environment
        class OpticalEnvironment:
            def __init__(self, params):
                self.params = params
                self.state = None
                self.reset()
            
            def reset(self):
                """Reset to initial random state."""
                self.state = {
                    'lens_type': np.random.choice(len(self.params['lens_types'])),
                    'material': np.random.choice(len(self.params['materials'])),
                    'curvature': np.random.uniform(*self.params['curvature_range']),
                    'thickness': np.random.uniform(*self.params['thickness_range'])
                }
                return self._state_to_array(self.state)
            
            def _state_to_array(self, state):
                """Convert state dict to array."""
                return np.array([
                    state['lens_type'] / len(self.params['lens_types']),
                    state['material'] / len(self.params['materials']),
                    (state['curvature'] - self.params['curvature_range'][0]) / 
                    (self.params['curvature_range'][1] - self.params['curvature_range'][0]),
                    (state['thickness'] - self.params['thickness_range'][0]) / 
                    (self.params['thickness_range'][1] - self.params['thickness_range'][0])
                ])
            
            def _calculate_focal_length(self, state):
                """Calculate focal length using simplified lensmaker's equation."""
                n = self.params['materials'][state['material']]
                R = 1.0 / state['curvature'] if state['curvature'] != 0 else np.inf
                t = state['thickness']
                
                # Simplified focal length calculation
                if np.isfinite(R):
                    f = R / (n - 1) * (0.8 + 0.4 * np.random.random())  # Add some noise
                else:
                    f = 1000.0  # Very long focal length for planar surface
                
                return f
            
            def step(self, action):
                """Take action and return new state, reward, done."""
                # Apply action to state
                if action == 0:  # Change lens type
                    self.state['lens_type'] = (self.state['lens_type'] + 1) % len(self.params['lens_types'])
                elif action == 1:  # Change material
                    self.state['material'] = (self.state['material'] + 1) % len(self.params['materials'])
                elif action == 2:  # Increase curvature
                    self.state['curvature'] = np.clip(
                        self.state['curvature'] + 0.01, 
                        self.params['curvature_range'][0], 
                        self.params['curvature_range'][1]
                    )
                elif action == 3:  # Decrease curvature
                    self.state['curvature'] = np.clip(
                        self.state['curvature'] - 0.01, 
                        self.params['curvature_range'][0], 
                        self.params['curvature_range'][1]
                    )
                elif action == 4:  # Increase thickness
                    self.state['thickness'] = np.clip(
                        self.state['thickness'] + 0.5, 
                        self.params['thickness_range'][0], 
                        self.params['thickness_range'][1]
                    )
                elif action == 5:  # Decrease thickness
                    self.state['thickness'] = np.clip(
                        self.state['thickness'] - 0.5, 
                        self.params['thickness_range'][0], 
                        self.params['thickness_range'][1]
                    )
                
                # Calculate reward based on focal length error
                current_focal_length = self._calculate_focal_length(self.state)
                focal_length_error = abs(current_focal_length - self.params['target_focal_length'])
                
                # Reward is negative error (want to minimize error)
                reward = -focal_length_error / self.params['target_focal_length']
                
                # Done if we're close enough to target
                done = focal_length_error < 5.0  # Within 5mm
                
                return self._state_to_array(self.state), reward, done
        
        # Q-Learning implementation
        class QLearningAgent:
            def __init__(self, n_states=4, n_actions=6, learning_rate=0.1, gamma=0.95, epsilon=0.1):
                self.n_states = n_states  # State dimension
                self.n_actions = n_actions
                self.learning_rate = learning_rate
                self.gamma = gamma
                self.epsilon = epsilon
                
                # Simple Q-table (discretized state space)
                self.n_bins = 10  # Discretization bins per dimension
                self.q_table = np.random.uniform(-1, 1, (self.n_bins**self.n_states, n_actions))
            
            def _discretize_state(self, state):
                """Convert continuous state to discrete index."""
                # Simple binning
                discrete_state = np.digitize(state, np.linspace(0, 1, self.n_bins-1))
                discrete_state = np.clip(discrete_state, 0, self.n_bins-1)
                
                # Convert to single index
                index = 0
                for i, s in enumerate(discrete_state):
                    index += s * (self.n_bins ** i)
                
                return index
            
            def choose_action(self, state):
                """Choose action using epsilon-greedy policy."""
                if np.random.random() < self.epsilon:
                    return np.random.choice(self.n_actions)
                
                discrete_state = self._discretize_state(state)
                return np.argmax(self.q_table[discrete_state])
            
            def learn(self, state, action, reward, next_state):
                """Update Q-values using Q-learning update rule."""
                discrete_state = self._discretize_state(state)
                discrete_next_state = self._discretize_state(next_state)
                
                current_q = self.q_table[discrete_state, action]
                max_next_q = np.max(self.q_table[discrete_next_state])
                
                # Q-learning update
                new_q = current_q + self.learning_rate * (
                    reward + self.gamma * max_next_q - current_q
                )
                
                self.q_table[discrete_state, action] = new_q
        
        # Training loop
        env = OpticalEnvironment(environment_params)
        agent = QLearningAgent()
        
        n_episodes = 1000
        rewards_history = []
        success_rate_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 50
            
            while steps < max_steps:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                agent.learn(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            rewards_history.append(total_reward)
            
            # Track success rate over last 100 episodes
            if episode >= 100:
                success_rate = np.mean([r > -0.1 for r in rewards_history[-100:]])
                success_rate_history.append(success_rate)
        
        # Test the trained agent
        test_episodes = 100
        test_rewards = []
        final_states = []
        
        for _ in range(test_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 30
            
            while steps < max_steps:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            final_states.append(env.state)
        
        results = {
            'agent': agent,
            'environment': env,
            'training_history': {
                'rewards': rewards_history,
                'success_rates': success_rate_history
            },
            'test_performance': {
                'mean_reward': np.mean(test_rewards),
                'std_reward': np.std(test_rewards),
                'success_rate': np.mean([r > -0.1 for r in test_rewards])
            },
            'final_solutions': final_states,
            'learning_parameters': {
                'learning_rate': agent.learning_rate,
                'gamma': agent.gamma,
                'epsilon': agent.epsilon,
                'episodes': n_episodes
            }
        }
        
        return results
    
    def hybrid_ai_traditional_optimization(self, initial_design: Optional[Dict] = None) -> Dict:
        """
        Hybrid AI-traditional optimization for optical systems.
        
        Combines machine learning predictions with traditional optimization.
        """
        if initial_design is None:
            initial_design = {
                'focal_length': 100.0,  # mm
                'f_number': 4.0,
                'field_of_view': 10.0,  # degrees
                'wavelength': 0.55  # microns
            }
        
        # Traditional optical merit function
        def merit_function(params, design_specs):
            """
            Traditional optical merit function.
            Combines multiple optical performance metrics.
            """
            r1, r2, thickness, n = params
            
            # Target specifications
            target_focal = design_specs['focal_length']
            target_f_num = design_specs['f_number']
            
            # Calculate actual focal length using lensmaker's equation
            try:
                focal_length = 1 / ((n - 1) * (1/r1 - 1/r2 + (n - 1) * thickness / (n * r1 * r2)))
            except (ZeroDivisionError, OverflowError):
                return 1e10  # Large penalty for invalid parameters
            
            # Calculate other optical properties
            f_number = focal_length / (2 * 12.5)  # Assuming 25mm aperture
            
            # Spherical aberration (simplified)
            if r1 != 0 and r2 != 0:
                sa_coeff = (n - 1) * (1/r1**3 - 1/r2**3)
                spherical_aberration = abs(sa_coeff)
            else:
                spherical_aberration = 1e6
            
            # Merit function components
            focal_error = (focal_length - target_focal)**2 / target_focal**2
            fnum_error = (f_number - target_f_num)**2 / target_f_num**2
            aberration_penalty = spherical_aberration * 1e6
            
            # Total merit (lower is better)
            total_merit = focal_error + fnum_error + aberration_penalty
            
            return total_merit
        
        # AI-based prediction model (simplified)
        class AIPredictor:
            def __init__(self):
                # Pre-trained weights (would normally come from actual training)
                self.weights = np.array([0.3, -0.2, 0.1, 0.4])  # Simplified model
                self.bias = np.array([50.0, -30.0, 5.0, 1.5])
            
            def predict(self, design_specs):
                """Predict initial lens parameters from design specifications."""
                features = np.array([
                    design_specs['focal_length'] / 200.0,  # Normalized
                    design_specs['f_number'] / 10.0,
                    design_specs['field_of_view'] / 30.0,
                    design_specs['wavelength'] / 1.0
                ])
                
                # Simple linear prediction
                params = self.weights * features + self.bias
                
                # Ensure reasonable ranges
                params = np.clip(params, [10, -100, 1, 1.4], [200, -10, 10, 1.8])
                
                return params
            
            def confidence_score(self, design_specs):
                """Estimate confidence in prediction based on similarity to training data."""
                # Simple confidence based on how close specs are to typical values
                focal_conf = np.exp(-abs(design_specs['focal_length'] - 100) / 100)
                fnum_conf = np.exp(-abs(design_specs['f_number'] - 4) / 4)
                fov_conf = np.exp(-abs(design_specs['field_of_view'] - 10) / 10)
                
                return (focal_conf + fnum_conf + fov_conf) / 3.0
        
        # Create AI predictor
        ai_predictor = AIPredictor()
        
        # Get AI prediction
        ai_prediction = ai_predictor.predict(initial_design)
        confidence = ai_predictor.confidence_score(initial_design)
        
        print(f"AI Prediction: {ai_prediction}")
        print(f"Confidence Score: {confidence:.3f}")
        
        # Traditional optimization starting from AI prediction
        bounds = [(10, 200), (-200, -10), (1, 10), (1.4, 1.8)]  # [r1, r2, thickness, n]
        
        # First optimization: from AI prediction
        result_ai_start = optimize.minimize(
            merit_function,
            ai_prediction,
            args=(initial_design,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Second optimization: from random start (for comparison)
        random_start = [
            np.random.uniform(20, 180),      # r1
            np.random.uniform(-180, -20),    # r2
            np.random.uniform(2, 8),         # thickness
            np.random.uniform(1.45, 1.75)    # refractive index
        ]
        
        result_random_start = optimize.minimize(
            merit_function,
            random_start,
            args=(initial_design,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Compare results
        def evaluate_solution(params, design_specs):
            """Evaluate a solution comprehensively."""
            merit = merit_function(params, design_specs)
            
            # Calculate additional metrics
            r1, r2, thickness, n = params
            
            try:
                focal_length = 1 / ((n - 1) * (1/r1 - 1/r2 + (n - 1) * thickness / (n * r1 * r2)))
                f_number = focal_length / 25.0  # Assuming 25mm aperture
            except:
                focal_length = np.inf
                f_number = np.inf
            
            return {
                'merit': merit,
                'focal_length': focal_length,
                'f_number': f_number,
                'parameters': params
            }
        
        ai_solution = evaluate_solution(result_ai_start.x, initial_design)
        random_solution = evaluate_solution(result_random_start.x, initial_design)
        
        results = {
            'ai_prediction': {
                'parameters': ai_prediction,
                'confidence': confidence
            },
            'optimization_results': {
                'ai_start': {
                    'success': result_ai_start.success,
                    'iterations': result_ai_start.nit,
                    'final_merit': result_ai_start.fun,
                    'solution': ai_solution
                },
                'random_start': {
                    'success': result_random_start.success,
                    'iterations': result_random_start.nit,
                    'final_merit': result_random_start.fun,
                    'solution': random_solution
                }
            },
            'comparison': {
                'merit_improvement': random_solution['merit'] - ai_solution['merit'],
                'iteration_savings': result_random_start.nit - result_ai_start.nit,
                'ai_better': ai_solution['merit'] < random_solution['merit']
            },
            'design_specifications': initial_design
        }
        
        return results
    
    def demonstrate_all_methods(self):
        """Demonstrate all AI integration methods."""
        print("=== Chapter 13: AI Integration in Optical Design ===\n")
        
        # 1. Neural Network Lens Design
        print("1. Neural Network for Lens Design Prediction")
        print("-" * 45)
        nn_results = self.neural_network_lens_design()
        
        print(f"Training completed in {len(nn_results['training_losses'])} epochs")
        print(f"Final training loss: {nn_results['training_losses'][-1]:.6f}")
        print("Test prediction for 50mm f/4 lens:")
        pred = nn_results['test_prediction']
        for name, value in zip(pred['parameter_names'], pred['predicted_parameters']):
            print(f"  {name}: {value:.3f}")
        print()
        
        # 2. Gaussian Process Optical Modeling
        print("2. Gaussian Process for Optical Surface Modeling")
        print("-" * 50)
        gp_results = self.gaussian_process_optical_modeling()
        
        print(f"GP Model Performance:")
        print(f"  Mean Squared Error: {gp_results['performance_metrics']['mse']:.6f}")
        print(f"  R² Score: {gp_results['performance_metrics']['r2']:.4f}")
        print(f"  Estimated noise level: {gp_results['performance_metrics']['noise_estimate']:.4f}")
        print(f"  Length scale: {gp_results['hyperparameters']['length_scale']:.2f}")
        print()
        
        # 3. Reinforcement Learning Optimization
        print("3. Reinforcement Learning for Optical System Optimization")
        print("-" * 55)
        rl_results = self.reinforcement_learning_optimization()
        
        print(f"Q-Learning Training Results:")
        print(f"  Episodes: {rl_results['learning_parameters']['episodes']}")
        print(f"  Final success rate: {rl_results['test_performance']['success_rate']:.1%}")
        print(f"  Mean test reward: {rl_results['test_performance']['mean_reward']:.3f} ± "
              f"{rl_results['test_performance']['std_reward']:.3f}")
        print()
        
        # 4. Hybrid AI-Traditional Optimization
        print("4. Hybrid AI-Traditional Optimization")
        print("-" * 40)
        hybrid_results = self.hybrid_ai_traditional_optimization()
        
        print(f"Hybrid Optimization Results:")
        ai_start = hybrid_results['optimization_results']['ai_start']
        random_start = hybrid_results['optimization_results']['random_start']
        
        print(f"AI-based initialization:")
        print(f"  Final merit: {ai_start['final_merit']:.6f}")
        print(f"  Iterations: {ai_start['iterations']}")
        print(f"  Success: {ai_start['success']}")
        
        print(f"Random initialization:")
        print(f"  Final merit: {random_start['final_merit']:.6f}")
        print(f"  Iterations: {random_start['iterations']}")
        print(f"  Success: {random_start['success']}")
        
        comparison = hybrid_results['comparison']
        print(f"AI advantage:")
        print(f"  Merit improvement: {comparison['merit_improvement']:.6f}")
        print(f"  Iteration savings: {comparison['iteration_savings']}")
        print(f"  AI performed better: {comparison['ai_better']}")
        
        print("\n" + "="*60)
        print("All AI integration methods demonstrated successfully!")


if __name__ == "__main__":
    # Create instance and run demonstrations
    ai_optics = AIOpticalDesign()
    
    # Run all demonstrations
    ai_optics.demonstrate_all_methods()
    
    print("\nKey Concepts Demonstrated:")
    print("- Neural networks for inverse design (specifications → parameters)")
    print("- Gaussian processes for uncertainty-aware optical modeling")
    print("- Reinforcement learning for sequential optimization decisions")
    print("- Hybrid AI-traditional optimization combining both approaches")
    print("- Machine learning integration throughout the optical design workflow")