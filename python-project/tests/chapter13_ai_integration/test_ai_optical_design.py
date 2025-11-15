"""
Test module for Chapter 13: AI Integration in Optical Design
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chapter13_ai_integration.ai_optical_design import AIOpticalDesign


class TestAIOpticalDesign:
    """Test class for AI integration methods in optical design."""
    
    @pytest.fixture
    def ai_optics(self):
        """Fixture for AIOpticalDesign instance."""
        return AIOpticalDesign()
    
    def test_initialization(self, ai_optics):
        """Test proper initialization of AIOpticalDesign."""
        assert hasattr(ai_optics, 'rng')
        assert isinstance(ai_optics.rng, np.random.RandomState)
        assert ai_optics.rng.get_state()[1][0] == 42  # Check seed
    
    def test_neural_network_lens_design_basic(self, ai_optics):
        """Test basic neural network lens design functionality."""
        results = ai_optics.neural_network_lens_design()
        
        # Check that all expected keys are present
        expected_keys = [
            'neural_network', 'training_losses', 'test_prediction', 'training_data_stats'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check neural network
        nn = results['neural_network']
        assert hasattr(nn, 'weights')
        assert hasattr(nn, 'biases')
        assert hasattr(nn, 'forward')
        assert hasattr(nn, 'backward')
        
        # Check training losses
        losses = results['training_losses']
        assert len(losses) == 1000  # Should have 1000 epochs
        assert all(isinstance(loss, (int, float)) for loss in losses)
        assert losses[-1] <= losses[0]  # Loss should generally decrease
        
        # Check test prediction
        test_pred = results['test_prediction']
        assert 'specifications' in test_pred
        assert 'predicted_parameters' in test_pred
        assert 'parameter_names' in test_pred
        assert len(test_pred['predicted_parameters']) == 4  # 4 output parameters
        assert len(test_pred['parameter_names']) == 4
        
        # Check training data stats
        training_stats = results['training_data_stats']
        assert training_stats['n_samples'] == 1000
        assert len(training_stats['input_features']) == 4
        assert len(training_stats['output_parameters']) == 4
    
    def test_neural_network_lens_design_with_custom_data(self, ai_optics):
        """Test neural network with custom training data."""
        # Create custom training data
        n_samples = 100
        X_custom = np.random.randn(n_samples, 4)
        y_custom = np.random.randn(n_samples, 4)
        
        training_data = {'X': X_custom, 'y': y_custom}
        
        results = ai_optics.neural_network_lens_design(training_data=training_data)
        
        # Check that custom data was used
        assert results['training_data_stats']['n_samples'] == n_samples
        
        # Test prediction with custom data
        test_spec = np.array([[1.0, 2.0, 3.0, 0.5]])
        nn = results['neural_network']
        prediction = nn.forward(test_spec)
        
        assert prediction.shape == (1, 4)  # Should have 4 output parameters
    
    def test_neural_network_forward_pass(self, ai_optics):
        """Test neural network forward pass functionality."""
        results = ai_optics.neural_network_lens_design()
        nn = results['neural_network']
        
        # Test forward pass with different inputs
        test_inputs = [
            np.array([[50, 4, 10, 0.55]]),  # Normal case
            np.array([[10, 2, 5, 0.4]]),   # Edge case
            np.array([[200, 8, 30, 0.8]])  # Edge case
        ]
        
        for test_input in test_inputs:
            output = nn.forward(test_input)
            
            # Check output shape
            assert output.shape == (1, 4)
            
            # Check that output values are reasonable
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
    
    def test_neural_network_sigmoid_functions(self, ai_optics):
        """Test sigmoid activation functions in neural network."""
        results = ai_optics.neural_network_lens_design()
        nn = results['neural_network']
        
        # Test sigmoid function
        test_values = np.array([-2, -1, 0, 1, 2])
        sigmoid_output = nn.sigmoid(test_values)
        
        # Check sigmoid properties
        assert np.all(sigmoid_output >= 0)
        assert np.all(sigmoid_output <= 1)
        assert np.isclose(nn.sigmoid(0), 0.5)
        assert nn.sigmoid(np.inf) == 1
        assert nn.sigmoid(-np.inf) == 0
        
        # Test sigmoid derivative
        sigmoid_deriv = nn.sigmoid_derivative(test_values)
        assert np.all(sigmoid_deriv >= 0)  # Should be non-negative
        assert np.all(sigmoid_deriv <= 0.25)  # Maximum derivative is 0.25 at x=0
    
    def test_gaussian_process_optical_modeling_basic(self, ai_optics):
        """Test basic Gaussian process optical modeling functionality."""
        results = ai_optics.gaussian_process_optical_modeling()
        
        # Check that all expected keys are present
        expected_keys = [
            'gaussian_process', 'predictions', 'training_data', 
            'performance_metrics', 'hyperparameters'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check Gaussian process
        gp = results['gaussian_process']
        assert hasattr(gp, 'length_scale')
        assert hasattr(gp, 'noise_level')
        assert hasattr(gp, 'fit')
        assert hasattr(gp, 'predict')
        
        # Check predictions
        predictions = results['predictions']
        assert 'mean' in predictions
        assert 'std' in predictions
        assert 'test_grid' in predictions
        
        pred_mean = predictions['mean']
        pred_std = predictions['std']
        
        assert pred_mean.shape == (30, 30)  # Should match test grid size
        assert pred_std.shape == (30, 30)
        assert np.all(pred_std >= 0)  # Standard deviation should be non-negative
        
        # Check performance metrics
        metrics = results['performance_metrics']
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert 'noise_estimate' in metrics
        
        assert metrics['mse'] >= 0  # MSE should be non-negative
        assert metrics['r2'] <= 1  # R² should be ≤ 1
        assert metrics['r2'] >= 0  # Should be non-negative for this case
        assert metrics['noise_estimate'] >= 0
    
    def test_gaussian_process_kernel_function(self, ai_optics):
        """Test Gaussian process RBF kernel function."""
        results = ai_optics.gaussian_process_optical_modeling()
        gp = results['gaussian_process']
        
        # Test kernel function
        X1 = np.array([[0, 0], [1, 1]])
        X2 = np.array([[0, 0], [2, 2]])
        
        kernel_matrix = gp.rbf_kernel(X1, X2)
        
        # Check kernel properties
        assert kernel_matrix.shape == (2, 2)
        assert np.all(kernel_matrix >= 0)  # Should be non-negative
        assert np.all(kernel_matrix <= 1)  # Should be ≤ 1
        
        # Diagonal should be 1 (self-similarity)
        kernel_self = gp.rbf_kernel(X1, X1)
        np.testing.assert_allclose(np.diag(kernel_self), 1.0, rtol=1e-10)
        
        # Should be symmetric
        assert np.allclose(kernel_self, kernel_self.T)
    
    def test_gaussian_process_with_custom_data(self, ai_optics):
        """Test Gaussian process with custom training data."""
        # Create custom training data
        n_train = 20
        x_train = np.linspace(-5, 5, n_train)
        y_train = np.linspace(-5, 5, n_train)
        X_train, Y_train = np.meshgrid(x_train, y_train)
        
        # Simple surface: paraboloid
        Z_train = X_train**2 + Y_train**2 + np.random.normal(0, 0.1, X_train.shape)
        
        training_points = np.column_stack([
            X_train.ravel(), Y_train.ravel(), Z_train.ravel()
        ])
        
        results = ai_optics.gaussian_process_optical_modeling(training_points=training_points)
        
        # Check that custom data was used
        training_data = results['training_data']
        assert training_data['X'].shape[0] == n_train**2
        assert training_data['y'].shape[0] == n_train**2
        
        # Test predictions
        predictions = results['predictions']
        pred_mean = predictions['mean']
        pred_std = predictions['std']
        
        # Should have reasonable prediction quality
        assert np.all(pred_std >= 0)
        assert pred_mean.shape == (30, 30)
    
    def test_reinforcement_learning_optimization_basic(self, ai_optics):
        """Test basic reinforcement learning optimization functionality."""
        results = ai_optics.reinforcement_learning_optimization()
        
        # Check that all expected keys are present
        expected_keys = [
            'agent', 'environment', 'training_history', 'test_performance', 
            'final_solutions', 'learning_parameters'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check agent
        agent = results['agent']
        assert hasattr(agent, 'q_table')
        assert hasattr(agent, 'choose_action')
        assert hasattr(agent, 'learn')
        assert hasattr(agent, 'learning_rate')
        assert hasattr(agent, 'gamma')
        assert hasattr(agent, 'epsilon')
        
        # Check environment
        env = results['environment']
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'params')
        
        # Check training history
        training_history = results['training_history']
        assert 'rewards' in training_history
        assert 'success_rates' in training_history
        
        rewards = training_history['rewards']
        success_rates = training_history['success_rates']
        
        assert len(rewards) == 1000  # 1000 episodes
        assert len(success_rates) == 900  # Should start after 100 episodes
        assert all(isinstance(r, (int, float)) for r in rewards)
        assert all(0 <= rate <= 1 for rate in success_rates)
        
        # Check test performance
        test_performance = results['test_performance']
        assert 'mean_reward' in test_performance
        assert 'std_reward' in test_performance
        assert 'success_rate' in test_performance
        
        assert -1 <= test_performance['mean_reward'] <= 0  # Should be negative (reward is -error)
        assert test_performance['std_reward'] >= 0
        assert 0 <= test_performance['success_rate'] <= 1
    
    def test_reinforcement_learning_agent_choose_action(self, ai_optics):
        """Test RL agent action selection."""
        results = ai_optics.reinforcement_learning_optimization()
        agent = results['agent']
        
        # Test with different states
        test_states = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.9, 0.8, 0.7, 0.6])
        ]
        
        for state in test_states:
            action = agent.choose_action(state)
            
            # Check that action is valid
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < 6  # Should be in [0, 5] for 6 actions
    
    def test_reinforcement_learning_environment_step(self, ai_optics):
        """Test RL environment step functionality."""
        results = ai_optics.reinforcement_learning_optimization()
        env = results['environment']
        
        # Reset environment
        initial_state = env.reset()
        assert isinstance(initial_state, np.ndarray)
        assert len(initial_state) == 4  # Should have 4 state dimensions
        assert np.all(initial_state >= 0)  # Should be normalized
        assert np.all(initial_state <= 1)
        
        # Test different actions
        for action in range(6):
            next_state, reward, done = env.step(action)
            
            # Check next state
            assert isinstance(next_state, np.ndarray)
            assert len(next_state) == 4
            assert np.all(next_state >= 0)
            assert np.all(next_state <= 1)
            
            # Check reward
            assert isinstance(reward, (int, float))
            assert reward <= 0  # Reward should be negative (minimize error)
            assert reward >= -1  # Should be bounded
            
            # Check done flag
            assert isinstance(done, bool)
    
    def test_reinforcement_learning_q_learning_update(self, ai_optics):
        """Test Q-learning update rule."""
        results = ai_optics.reinforcement_learning_optimization()
        agent = results['agent']
        
        # Test Q-learning update
        state = np.array([0.5, 0.5, 0.5, 0.5])
        action = 2
        reward = -0.1
        next_state = np.array([0.6, 0.6, 0.6, 0.6])
        
        # Get initial Q-value
        initial_q = agent.q_table[agent._discretize_state(state), action]
        
        # Apply learning update
        agent.learn(state, action, reward, next_state)
        
        # Get updated Q-value
        updated_q = agent.q_table[agent._discretize_state(state), action]
        
        # Q-value should have changed
        assert updated_q != initial_q
    
    def test_hybrid_ai_traditional_optimization_basic(self, ai_optics):
        """Test basic hybrid AI-traditional optimization functionality."""
        results = ai_optics.hybrid_ai_traditional_optimization()
        
        # Check that all expected keys are present
        expected_keys = [
            'ai_prediction', 'optimization_results', 'comparison', 'design_specifications'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check AI prediction
        ai_prediction = results['ai_prediction']
        assert 'parameters' in ai_prediction
        assert 'confidence' in ai_prediction
        
        predicted_params = ai_prediction['parameters']
        assert len(predicted_params) == 4  # 4 parameters
        assert all(isinstance(p, (int, float)) for p in predicted_params)
        
        confidence = ai_prediction['confidence']
        assert 0 <= confidence <= 1  # Should be a probability
        
        # Check optimization results
        opt_results = results['optimization_results']
        assert 'ai_start' in opt_results
        assert 'random_start' in opt_results
        
        for start_type in ['ai_start', 'random_start']:
            start_results = opt_results[start_type]
            assert 'success' in start_results
            assert 'iterations' in start_results
            assert 'final_merit' in start_results
            assert 'solution' in start_results
            
            assert isinstance(start_results['success'], bool)
            assert start_results['iterations'] > 0
            assert start_results['final_merit'] >= 0  # Merit should be non-negative
        
        # Check comparison
        comparison = results['comparison']
        assert 'merit_improvement' in comparison
        assert 'iteration_savings' in comparison
        assert 'ai_better' in comparison
        
        assert isinstance(comparison['merit_improvement'], (int, float))
        assert isinstance(comparison['iteration_savings'], (int, float))
        assert isinstance(comparison['ai_better'], bool)
    
    def test_hybrid_ai_predictor_confidence(self, ai_optics):
        """Test AI predictor confidence scoring."""
        results = ai_optics.hybrid_ai_traditional_optimization()
        
        # Test different design specifications
        test_specs = [
            {'focal_length': 100.0, 'f_number': 4.0, 'field_of_view': 10.0, 'wavelength': 0.55},
            {'focal_length': 50.0, 'f_number': 2.0, 'field_of_view': 5.0, 'wavelength': 0.4},
            {'focal_length': 200.0, 'f_number': 8.0, 'field_of_view': 30.0, 'wavelength': 0.8}
        ]
        
        # Create a new AI predictor to test confidence
        from chapter13_ai_integration.ai_optical_design import AIOpticalDesign
        test_ai = AIOpticalDesign()
        hybrid_results = test_ai.hybrid_ai_traditional_optimization(test_specs[0])
        
        confidence = hybrid_results['ai_prediction']['confidence']
        assert 0 <= confidence <= 1
        
        # Confidence should be reasonable for typical specifications
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_hybrid_optimization_merit_function(self, ai_optics):
        """Test the merit function used in hybrid optimization."""
        results = ai_optics.hybrid_ai_traditional_optimization()
        
        # Test merit function with different parameters
        test_params = [
            [50.0, -40.0, 5.0, 1.5],   # Reasonable parameters
            [20.0, -15.0, 2.0, 1.4],   # Edge case parameters
            [100.0, -80.0, 8.0, 1.7]   # Different parameters
        ]
        
        design_specs = {'focal_length': 100.0, 'f_number': 4.0, 'field_of_view': 10.0, 'wavelength': 0.55}
        
        # Import the merit function from the module
        from chapter13_ai_integration.ai_optical_design import AIOpticalDesign
        test_ai = AIOpticalDesign()
        
        # Get the merit function by running optimization
        hybrid_results = test_ai.hybrid_ai_traditional_optimization(design_specs)
        
        # Check that optimization succeeded
        ai_start_results = hybrid_results['optimization_results']['ai_start']
        random_start_results = hybrid_results['optimization_results']['random_start']
        
        assert ai_start_results['success']  # AI start should succeed
        assert random_start_results['success']  # Random start should succeed
        
        # Check that AI start performed better or comparably
        assert hybrid_results['comparison']['ai_better'] or hybrid_results['comparison']['merit_improvement'] < 0.1
    
    def test_reproducibility(self, ai_optics):
        """Test that results are reproducible with the same random seed."""
        # First run
        results1 = ai_optics.neural_network_lens_design()
        
        # Create new instance with same seed
        from chapter13_ai_integration.ai_optical_design import AIOpticalDesign
        ai_optics2 = AIOpticalDesign()
        
        # Second run
        results2 = ai_optics2.neural_network_lens_design()
        
        # Check that key results are reproducible
        np.testing.assert_allclose(
            results1['test_prediction']['predicted_parameters'],
            results2['test_prediction']['predicted_parameters'],
            rtol=1e-10
        )
        
        np.testing.assert_allclose(
            results1['training_losses'],
            results2['training_losses'],
            rtol=1e-10
        )
    
    def test_edge_cases_empty_data(self, ai_optics):
        """Test edge cases with empty or minimal data."""
        # Test neural network with very small dataset
        small_data = {'X': np.random.randn(5, 4), 'y': np.random.randn(5, 4)}
        results = ai_optics.neural_network_lens_design(training_data=small_data)
        
        # Should still produce valid results
        assert 'neural_network' in results
        assert 'training_losses' in results
        assert len(results['training_losses']) == 1000
    
    def test_edge_cases_extreme_parameters(self, ai_optics):
        """Test edge cases with extreme parameters."""
        # Test with extreme design specifications
        extreme_specs = {
            'focal_length': 500.0,  # Very long focal length
            'f_number': 1.0,        # Very fast lens
            'field_of_view': 60.0,  # Very wide field
            'wavelength': 0.3       # Short wavelength
        }
        
        results = ai_optics.hybrid_ai_traditional_optimization(extreme_specs)
        
        # Should still produce valid results
        assert 'ai_prediction' in results
        assert 'optimization_results' in results
        assert results['optimization_results']['ai_start']['success']
    
    def test_mathematical_consistency(self, ai_optics):
        """Test mathematical consistency of AI methods."""
        # Test neural network mathematical properties
        results = ai_optics.neural_network_lens_design()
        nn = results['neural_network']
        
        # Test that sigmoid derivative is correct
        test_values = np.array([-1, 0, 1])
        sigmoid_vals = nn.sigmoid(test_values)
        sigmoid_deriv_vals = nn.sigmoid_derivative(test_values)
        
        # Manual calculation of derivative
        expected_deriv = sigmoid_vals * (1 - sigmoid_vals)
        np.testing.assert_allclose(sigmoid_deriv_vals, expected_deriv, rtol=1e-10)
        
        # Test Gaussian process kernel properties
        gp_results = ai_optics.gaussian_process_optical_modeling()
        gp = gp_results['gaussian_process']
        
        # Kernel matrix should be positive semi-definite
        X_test = np.random.randn(10, 2)
        K = gp.rbf_kernel(X_test, X_test)
        
        # Check positive semi-definite (all eigenvalues should be >= 0)
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals.real >= -1e-10)  # Allow small numerical errors
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_demonstrate_all_methods(self, mock_figure, mock_show, ai_optics):
        """Test the demonstration method."""
        # This should run without errors
        ai_optics.demonstrate_all_methods()
        
        # Check that matplotlib functions were called
        assert mock_figure.called or mock_show.called