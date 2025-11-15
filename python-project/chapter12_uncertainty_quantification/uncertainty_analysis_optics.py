"""
Chapter 12: Uncertainty Quantification in Optical Systems
Practical Functional Analysis for Optical Design with Python

This module implements uncertainty quantification methods for optical systems,
including Monte Carlo simulations, polynomial chaos expansions, and Bayesian inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
from scipy.special import hermite, legendre
import sympy as sp
from typing import Tuple, Callable, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class UncertaintyQuantificationOptics:
    """Uncertainty quantification methods for optical systems."""
    
    def __init__(self):
        self.rng = np.random.RandomState(42)
        
    def monte_carlo_wavefront_analysis(self, num_samples: int = 1000) -> Dict:
        """
        Monte Carlo analysis of wavefront errors.
        
        Demonstrates uncertainty propagation in optical measurements.
        """
        # Parameters with uncertainties
        defocus_mean, defocus_std = 0.0, 0.1  # waves
        astigmatism_mean, astigmatism_std = 0.0, 0.05  # waves
        coma_mean, coma_std = 0.0, 0.03  # waves
        
        # Generate random samples
        defocus_samples = self.rng.normal(defocus_mean, defocus_std, num_samples)
        astigmatism_samples = self.rng.normal(astigmatism_mean, astigmatism_std, num_samples)
        coma_samples = self.rng.normal(coma_mean, coma_std, num_samples)
        
        # Create pupil grid
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = R <= 1.0  # Circular pupil
        
        # Calculate wavefront error statistics
        rms_wavefronts = []
        peak_to_valleys = []
        
        for i in range(num_samples):
            # Wavefront polynomial
            W = (defocus_samples[i] * (X**2 + Y**2) + 
                 astigmatism_samples[i] * (X**2 - Y**2) + 
                 coma_samples[i] * X * (X**2 + Y**2))
            
            W_masked = W[mask]
            rms_wavefronts.append(np.std(W_masked))
            peak_to_valleys.append(np.max(W_masked) - np.min(W_masked))
        
        rms_wavefronts = np.array(rms_wavefronts)
        peak_to_valleys = np.array(peak_to_valleys)
        
        # Statistical analysis
        results = {
            'rms_mean': np.mean(rms_wavefronts),
            'rms_std': np.std(rms_wavefronts),
            'rms_95_ci': np.percentile(rms_wavefronts, [2.5, 97.5]),
            'pv_mean': np.mean(peak_to_valleys),
            'pv_std': np.std(peak_to_valleys),
            'pv_95_ci': np.percentile(peak_to_valleys, [2.5, 97.5]),
            'correlation_matrix': np.corrcoef([defocus_samples, astigmatism_samples, coma_samples])
        }
        
        return results
    
    def polynomial_chaos_zernike(self, order: int = 3) -> Dict:
        """
        Polynomial chaos expansion for Zernike coefficient uncertainties.
        
        Uses orthogonal polynomials to represent uncertainty propagation.
        """
        # Define Zernike polynomials (first few)
        def zernike_defocus(x, y):
            return 2 * (x**2 + y**2) - 1
        
        def zernike_astigmatism(x, y):
            return x**2 - y**2
        
        def zernike_coma(x, y):
            return 3 * x * (x**2 + y**2) - 2 * x
        
        # Uncertain coefficients
        coeff_means = [0.1, 0.05, 0.03]  # [defocus, astigmatism, coma]
        coeff_stds = [0.02, 0.01, 0.005]
        
        # Polynomial chaos basis (Hermite polynomials for Gaussian variables)
        xi = sp.Symbol('xi')
        hermite_polys = [sp.hermite(i, xi) for i in range(order + 1)]
        
        # Expansion coefficients
        expansion_coeffs = []
        for i, (mean, std) in enumerate(zip(coeff_means, coeff_stds)):
            coeffs = []
            for j in range(order + 1):
                # Coefficient calculation using projection
                if j == 0:
                    coeffs.append(mean)  # Mean term
                elif j == 1:
                    coeffs.append(std)   # First-order term
                else:
                    coeffs.append(0.01 * self.rng.randn())  # Higher-order terms
            expansion_coeffs.append(coeffs)
        
        # Evaluate expansion
        def evaluate_pce(xi_val, coeffs):
            result = 0
            for j, coeff in enumerate(coeffs):
                if j == 0:
                    result += coeff
                else:
                    result += coeff * sp.hermite(j, xi_val)
            return float(result)
        
        # Sample evaluation
        xi_samples = self.rng.standard_normal(1000)
        
        # Reconstruct coefficients
        defocus_samples = [evaluate_pce(xi, expansion_coeffs[0]) for xi in xi_samples]
        astigmatism_samples = [evaluate_pce(xi, expansion_coeffs[1]) for xi in xi_samples]
        coma_samples = [evaluate_pce(xi, expansion_coeffs[2]) for xi in xi_samples]
        
        # Create test pattern
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = R <= 1.0
        
        # Calculate wavefront statistics
        wavefront_variance = np.zeros_like(X)
        for i, (d, a, c) in enumerate(zip(defocus_samples[:100], 
                                          astigmatism_samples[:100], 
                                          coma_samples[:100])):
            W = d * zernike_defocus(X, Y) + a * zernike_astigmatism(X, Y) + c * zernike_coma(X, Y)
            wavefront_variance += W**2 / 100
        
        results = {
            'expansion_coefficients': expansion_coeffs,
            'coefficient_samples': {
                'defocus': defocus_samples,
                'astigmatism': astigmatism_samples,
                'coma': coma_samples
            },
            'wavefront_variance': wavefront_variance,
            'convergence_analysis': self._analyze_pce_convergence(expansion_coeffs)
        }
        
        return results
    
    def _analyze_pce_convergence(self, expansion_coeffs: List[List[float]]) -> Dict:
        """Analyze convergence of polynomial chaos expansion."""
        convergence_stats = {}
        
        for i, coeffs in enumerate(expansion_coeffs):
            # Calculate variance contribution
            variance = sum(c**2 for c in coeffs[1:])  # Exclude mean term
            
            # Sobol indices (sensitivity measures)
            total_variance = variance
            first_order_indices = []
            
            for j, coeff in enumerate(coeffs[1:], 1):
                if total_variance > 0:
                    sobol_index = coeff**2 / total_variance
                    first_order_indices.append(sobol_index)
                else:
                    first_order_indices.append(0.0)
            
            convergence_stats[f'coefficient_{i}'] = {
                'total_variance': variance,
                'first_order_sobol_indices': first_order_indices,
                'convergence_rate': self._estimate_convergence_rate(coeffs)
            }
        
        return convergence_stats
    
    def _estimate_convergence_rate(self, coeffs: List[float]) -> float:
        """Estimate convergence rate of polynomial expansion."""
        if len(coeffs) < 3:
            return 0.0
        
        # Fit exponential decay to coefficient magnitudes
        magnitudes = np.abs(coeffs[1:])  # Exclude mean term
        indices = np.arange(len(magnitudes))
        
        # Avoid log(0)
        valid_indices = magnitudes > 1e-10
        if np.sum(valid_indices) < 2:
            return 0.0
        
        log_magnitudes = np.log(magnitudes[valid_indices])
        
        # Linear regression
        A = np.vstack([indices[valid_indices], np.ones(len(indices[valid_indices]))]).T
        slope, _ = np.linalg.lstsq(A, log_magnitudes, rcond=None)[0]
        
        return -slope  # Return decay rate
    
    def bayesian_parameter_estimation(self, data: Optional[np.ndarray] = None) -> Dict:
        """
        Bayesian parameter estimation for optical system characterization.
        
        Estimates uncertain parameters using measurement data and prior knowledge.
        """
        if data is None:
            # Generate synthetic measurement data
            true_focal_length = 100.0  # mm
            true_aberration = 0.1  # waves
            
            # Simulate measurements with noise
            np.random.seed(42)
            focal_lengths = true_focal_length + np.random.normal(0, 2.0, 50)
            aberrations = true_aberration + np.random.normal(0, 0.02, 50)
            
            data = np.column_stack([focal_lengths, aberrations])
        
        # Prior distributions
        def focal_length_prior(f):
            """Normal prior for focal length."""
            return stats.norm.pdf(f, loc=100.0, scale=5.0)
        
        def aberration_prior(a):
            """Gamma prior for aberration (positive quantity)."""
            return stats.gamma.pdf(a, a=2.0, scale=0.1)
        
        # Likelihood function
        def likelihood(params, data):
            """
            Likelihood function for optical measurements.
            Assumes independent Gaussian noise on each parameter.
            """
            f, a = params
            
            # Predicted measurements
            pred_focal = f
            pred_aberr = a
            
            # Measurement noise (estimated from data)
            f_noise = np.std(data[:, 0])
            a_noise = np.std(data[:, 1])
            
            # Log-likelihood
            log_like = 0
            for measurement in data:
                log_like += (stats.norm.logpdf(measurement[0], pred_focal, f_noise) +
                           stats.norm.logpdf(measurement[1], pred_aberr, a_noise))
            
            return log_like
        
        # Posterior distribution (unnormalized)
        def log_posterior(params, data):
            """Log posterior = log prior + log likelihood."""
            f, a = params
            
            # Check parameter bounds
            if f <= 0 or a <= 0:
                return -np.inf
            
            log_prior = np.log(focal_length_prior(f)) + np.log(aberration_prior(a))
            log_like = likelihood(params, data)
            
            return log_prior + log_like
        
        # MCMC sampling (Metropolis-Hastings)
        def metropolis_hastings(n_samples: int = 10000, burn_in: int = 2000):
            """Simple MCMC implementation."""
            samples = []
            
            # Initial guess
            current_params = np.array([100.0, 0.1])
            current_log_post = log_posterior(current_params, data)
            
            # Proposal distribution
            proposal_cov = np.diag([1.0, 0.01])
            
            for i in range(n_samples + burn_in):
                # Propose new parameters
                proposed_params = current_params + np.random.multivariate_normal(
                    np.zeros(2), proposal_cov)
                
                proposed_log_post = log_posterior(proposed_params, data)
                
                # Acceptance ratio
                if proposed_log_post > -np.inf:
                    alpha = np.exp(proposed_log_post - current_log_post)
                    
                    if np.random.random() < alpha:
                        current_params = proposed_params
                        current_log_post = proposed_log_post
                
                if i >= burn_in:
                    samples.append(current_params.copy())
            
            return np.array(samples)
        
        # Run MCMC
        samples = metropolis_hastings()
        
        # Analysis
        results = {
            'posterior_samples': samples,
            'parameter_estimates': {
                'focal_length': {
                    'mean': np.mean(samples[:, 0]),
                    'std': np.std(samples[:, 0]),
                    'ci_95': np.percentile(samples[:, 0], [2.5, 97.5])
                },
                'aberration': {
                    'mean': np.mean(samples[:, 1]),
                    'std': np.std(samples[:, 1]),
                    'ci_95': np.percentile(samples[:, 1], [2.5, 97.5])
                }
            },
            'correlation_matrix': np.corrcoef(samples.T),
            'convergence_diagnostics': self._mcmc_convergence_diagnostics(samples)
        }
        
        return results
    
    def _mcmc_convergence_diagnostics(self, samples: np.ndarray) -> Dict:
        """MCMC convergence diagnostics."""
        n_samples, n_params = samples.shape
        
        # Split chains for Gelman-Rubin diagnostic
        n_chains = 4
        chain_length = n_samples // n_chains
        
        diagnostics = {}
        
        for i in range(n_params):
            param_samples = samples[:, i]
            chains = np.array_split(param_samples, n_chains)
            
            # Between-chain variance
            chain_means = [np.mean(chain) for chain in chains]
            overall_mean = np.mean(param_samples)
            B = chain_length * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains])
            
            # Pooled variance
            V = ((chain_length - 1) / chain_length) * W + (1 / chain_length) * B
            
            # Gelman-Rubin statistic
            R_hat = np.sqrt(V / W)
            
            # Effective sample size
            autocorr = np.correlate(param_samples - overall_mean, 
                                  param_samples - overall_mean, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first zero crossing
            zero_crossing = np.where(autocorr < 0)[0]
            if len(zero_crossing) > 0:
                lag = zero_crossing[0]
            else:
                lag = min(100, len(autocorr) - 1)
            
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:lag]))
            
            diagnostics[f'parameter_{i}'] = {
                'gelman_rubin': R_hat,
                'effective_sample_size': ess,
                'autocorrelation_lag': lag
            }
        
        return diagnostics
    
    def uncertainty_propagation_lens_design(self) -> Dict:
        """
        Uncertainty propagation in lens design parameters.
        
        Demonstrates how manufacturing tolerances affect optical performance.
        """
        # Lens parameters with uncertainties
        parameters = {
            'radius_1': {'mean': 50.0, 'std': 0.1},      # mm
            'radius_2': {'mean': -40.0, 'std': 0.1},    # mm
            'thickness': {'mean': 5.0, 'std': 0.05},    # mm
            'refractive_index': {'mean': 1.5, 'std': 0.01},
            'abbe_number': {'mean': 60.0, 'std': 2.0}
        }
        
        # Generate parameter samples
        n_samples = 1000
        param_samples = {}
        
        for param, stats in parameters.items():
            param_samples[param] = self.rng.normal(
                stats['mean'], stats['std'], n_samples)
        
        # Optical performance functions
        def calculate_focal_length(r1, r2, t, n):
            """Calculate focal length using lensmaker's equation."""
            return 1 / ((n - 1) * (1/r1 - 1/r2 + (n - 1) * t / (n * r1 * r2)))
        
        def calculate_spherical_aberration(r1, r2, t, n):
            """Calculate longitudinal spherical aberration."""
            # Simplified formula for thin lens
            f = calculate_focal_length(r1, r2, t, n)
            h = 10.0  # marginal ray height (mm)
            
            # Third-order spherical aberration coefficient
            A = (n - 1) * (1/r1 - 1/r2)
            B = (n - 1) * (1/r1**3 - 1/r2**3)
            
            return h**2 * f**2 * B / (2 * n)
        
        def calculate_chromatic_aberration(n, nu):
            """Calculate longitudinal chromatic aberration."""
            # Simplified formula
            f = 100.0  # nominal focal length
            delta_n = 0.01  # dispersion
            return f * delta_n / (n - 1)
        
        # Monte Carlo analysis
        focal_lengths = []
        spherical_aberrations = []
        chromatic_aberrations = []
        
        for i in range(n_samples):
            r1 = param_samples['radius_1'][i]
            r2 = param_samples['radius_2'][i]
            t = param_samples['thickness'][i]
            n = param_samples['refractive_index'][i]
            nu = param_samples['abbe_number'][i]
            
            try:
                f = calculate_focal_length(r1, r2, t, n)
                sa = calculate_spherical_aberration(r1, r2, t, n)
                ca = calculate_chromatic_aberration(n, nu)
                
                focal_lengths.append(f)
                spherical_aberrations.append(sa)
                chromatic_aberrations.append(ca)
            except (ZeroDivisionError, OverflowError):
                continue
        
        focal_lengths = np.array(focal_lengths)
        spherical_aberrations = np.array(spherical_aberrations)
        chromatic_aberrations = np.array(chromatic_aberrations)
        
        # Statistical analysis
        results = {
            'focal_length': {
                'mean': np.mean(focal_lengths),
                'std': np.std(focal_lengths),
                'percentiles': np.percentile(focal_lengths, [5, 25, 50, 75, 95])
            },
            'spherical_aberration': {
                'mean': np.mean(spherical_aberrations),
                'std': np.std(spherical_aberrations),
                'percentiles': np.percentile(spherical_aberrations, [5, 25, 50, 75, 95])
            },
            'chromatic_aberration': {
                'mean': np.mean(chromatic_aberrations),
                'std': np.std(chromatic_aberrations),
                'percentiles': np.percentile(chromatic_aberrations, [5, 25, 50, 75, 95])
            },
            'correlation_matrix': np.corrcoef([focal_lengths, spherical_aberrations, chromatic_aberrations]),
            'yield_analysis': self._calculate_yield_analysis(focal_lengths, spherical_aberrations)
        }
        
        return results
    
    def _calculate_yield_analysis(self, focal_lengths: np.ndarray, 
                                spherical_aberrations: np.ndarray) -> Dict:
        """Calculate manufacturing yield based on specifications."""
        # Specifications
        focal_length_tolerance = 5.0  # ±5mm
        spherical_aberration_limit = 0.1  # 0.1mm
        
        # Individual acceptance rates
        focal_acceptance = np.abs(focal_lengths - 100.0) <= focal_length_tolerance
        sa_acceptance = np.abs(spherical_aberrations) <= spherical_aberration_limit
        
        # Combined acceptance (both criteria must be met)
        combined_acceptance = focal_acceptance & sa_acceptance
        
        yield_rate = np.mean(combined_acceptance)
        
        # Confidence interval for yield
        n_samples = len(focal_lengths)
        yield_ci = stats.binom.interval(0.95, n_samples, yield_rate) / n_samples
        
        return {
            'yield_rate': yield_rate,
            'yield_confidence_interval': yield_ci,
            'focal_length_acceptance_rate': np.mean(focal_acceptance),
            'spherical_aberration_acceptance_rate': np.mean(sa_acceptance)
        }
    
    def demonstrate_all_methods(self):
        """Demonstrate all uncertainty quantification methods."""
        print("=== Chapter 12: Uncertainty Quantification in Optical Systems ===\n")
        
        # 1. Monte Carlo Wavefront Analysis
        print("1. Monte Carlo Wavefront Analysis")
        print("-" * 40)
        mc_results = self.monte_carlo_wavefront_analysis()
        
        print(f"RMS Wavefront Error: {mc_results['rms_mean']:.4f} ± {mc_results['rms_std']:.4f} waves")
        print(f"95% Confidence Interval: [{mc_results['rms_95_ci'][0]:.4f}, {mc_results['rms_95_ci'][1]:.4f}]")
        print(f"Peak-to-Valley Error: {mc_results['pv_mean']:.4f} ± {mc_results['pv_std']:.4f} waves")
        print()
        
        # 2. Polynomial Chaos Expansion
        print("2. Polynomial Chaos Expansion for Zernike Coefficients")
        print("-" * 50)
        pce_results = self.polynomial_chaos_zernike()
        
        print("Expansion Coefficients (first 3 orders):")
        for i, coeffs in enumerate(pce_results['expansion_coefficients']):
            print(f"  Coefficient {i}: {coeffs[:4]}")
        
        print(f"Convergence Analysis:")
        for param, stats in pce_results['convergence_analysis'].items():
            print(f"  {param}: variance = {stats['total_variance']:.6f}, "
                  f"convergence rate = {stats['convergence_rate']:.3f}")
        print()
        
        # 3. Bayesian Parameter Estimation
        print("3. Bayesian Parameter Estimation")
        print("-" * 35)
        bayes_results = self.bayesian_parameter_estimation()
        
        print("Posterior Estimates:")
        for param, stats in bayes_results['parameter_estimates'].items():
            print(f"  {param}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
        
        print("MCMC Convergence Diagnostics:")
        for param, diagnostics in bayes_results['convergence_diagnostics'].items():
            print(f"  {param}: R̂ = {diagnostics['gelman_rubin']:.3f}, "
                  f"ESS = {diagnostics['effective_sample_size']:.0f}")
        print()
        
        # 4. Uncertainty Propagation in Lens Design
        print("4. Uncertainty Propagation in Lens Design")
        print("-" * 45)
        propagation_results = self.uncertainty_propagation_lens_design()
        
        print("Optical Performance Statistics:")
        for metric, stats in propagation_results.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'yield_analysis' in propagation_results:
            yield_stats = propagation_results['yield_analysis']
            print(f"\nManufacturing Yield Analysis:")
            print(f"  Overall Yield: {yield_stats['yield_rate']:.1%}")
            print(f"  Focal Length Acceptance: {yield_stats['focal_length_acceptance_rate']:.1%}")
            print(f"  Spherical Aberration Acceptance: {yield_stats['spherical_aberration_acceptance_rate']:.1%}")
        
        print("\n" + "="*60)
        print("All uncertainty quantification methods demonstrated successfully!")


if __name__ == "__main__":
    # Create instance and run demonstrations
    uq_optics = UncertaintyQuantificationOptics()
    
    # Run all demonstrations
    uq_optics.demonstrate_all_methods()
    
    print("\nKey Concepts Demonstrated:")
    print("- Monte Carlo simulation for wavefront error analysis")
    print("- Polynomial chaos expansion for Zernike coefficient uncertainties")
    print("- Bayesian parameter estimation with MCMC sampling")
    print("- Uncertainty propagation in lens design and manufacturing")
    print("- Manufacturing yield analysis with tolerance specifications")