# portfolio_c_interface.py
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os

class PortfolioEngineC:
    """Python interface to the C portfolio engine"""
    
    def __init__(self, library_path="./portfolio_engine.so"):
        """Initialize the C library interface"""
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"C library not found at {library_path}")
        
        self.lib = ctypes.CDLL(library_path)
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Define C function signatures for proper interface"""
        
        # run_portfolio_simulation function
        self.lib.run_portfolio_simulation.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # returns_data
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # weights
            ctypes.c_int,    # n_assets
            ctypes.c_int,    # n_periods
            ctypes.c_double, # risk_free_rate
            ctypes.c_int,    # simulation_years
            ctypes.c_int,    # n_simulations
            ctypes.c_double  # initial_value
        ]
        self.lib.run_portfolio_simulation.restype = ctypes.c_void_p
        
        # get_simulation_results function
        self.lib.get_simulation_results.argtypes = [
            ctypes.c_void_p,  # results pointer
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # output_simulations
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # output_finals
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # output_percentiles
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")   # output_metrics
        ]
        
        # cleanup function
        self.lib.free_simulation_results.argtypes = [ctypes.c_void_p]
    
    def run_monte_carlo_simulation(self, returns_df, weights, risk_free_rate=0.045, 
                                 simulation_years=5, n_simulations=1000, 
                                 initial_value=100000.0):
        """
        Run Monte Carlo simulation using C engine
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            Historical returns data with assets as columns
        weights : numpy.array
            Portfolio weights (must sum to 1)
        risk_free_rate : float
            Annual risk-free rate
        simulation_years : int
            Number of years to simulate
        n_simulations : int
            Number of Monte Carlo paths
        initial_value : float
            Initial portfolio value
            
        Returns:
        --------
        dict : Simulation results and metrics
        """
        
        # Prepare data for C function
        returns_data = returns_df.values.astype(np.float64, order='C')
        weights_array = np.array(weights, dtype=np.float64, order='C')
        
        n_assets = returns_df.shape[1]
        n_periods = returns_df.shape[0]
        
        # Call C function
        results_ptr = self.lib.run_portfolio_simulation(
            returns_data, weights_array, n_assets, n_periods,
            risk_free_rate, simulation_years, n_simulations, initial_value
        )
        
        if not results_ptr:
            raise RuntimeError("C simulation failed")
        
        # Prepare output arrays
        n_days = simulation_years * 252
        simulations = np.zeros((n_simulations, n_days), dtype=np.float64, order='C')
        final_values = np.zeros(n_simulations, dtype=np.float64, order='C')
        percentiles = np.zeros(3, dtype=np.float64, order='C')
        metrics = np.zeros(10, dtype=np.float64, order='C')  # 10 different metrics
        
        # Get results from C
        self.lib.get_simulation_results(
            results_ptr, simulations.flatten(), final_values, percentiles, metrics
        )
        
        # Clean up C memory
        self.lib.free_simulation_results(results_ptr)
        
        # Format results
        results = {
            'simulations': simulations,
            'final_values': final_values,
            'percentiles': {
                '5th': percentiles[0],
                '50th': percentiles[1],
                '95th': percentiles[2]
            },
            'metrics': {
                'annual_return': metrics[0],
                'annual_volatility': metrics[1],
                'sharpe_ratio': metrics[2],
                'max_drawdown': metrics[3],
                'var_95': metrics[4],
                'cvar_95': metrics[5],
                'var_99': metrics[6],
                'cvar_99': metrics[7],
                'skewness': metrics[8],
                'kurtosis': metrics[9]
            }
        }
        
        return results


# Modified Streamlit functions to use C engine
def calculate_advanced_metrics_c(returns, weights, c_engine, simulation_years=5, 
                               n_simulations=1000, initial_investment=100000):
    """Calculate advanced metrics using C engine"""
    
    # Run C simulation
    results = c_engine.run_monte_carlo_simulation(
        returns, weights, simulation_years=simulation_years,
        n_simulations=n_simulations, initial_value=initial_investment
    )
    
    # Format for Streamlit display
    metrics = results['metrics']
    final_values = results['final_values']
    
    # Calculate additional metrics
    prob_profit = (final_values > initial_investment).mean()
    
    formatted_metrics = {
        'Annual Return': f"{metrics['annual_return']:.2%}",
        'Annual Volatility': f"{metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
        'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
        'VaR (95%)': f"{metrics['var_95']:.2%}",
        'CVaR (95%)': f"{metrics['cvar_95']:.2%}",
        'VaR (99%)': f"{metrics['var_99']:.2%}",
        'CVaR (99%)': f"{metrics['cvar_99']:.2%}",
        'Skewness': f"{metrics['skewness']:.3f}",
        'Kurtosis': f"{metrics['kurtosis']:.3f}",
        'Probability of Profit': f"{prob_profit:.1%}",
        'Median Final Value': f"${results['percentiles']['50th']:,.0f}",
        '5th Percentile': f"${results['percentiles']['5th']:,.0f}",
        '95th Percentile': f"${results['percentiles']['95th']:,.0f}"
    }
    
    return formatted_metrics, results['simulations']

def monte_carlo_simulation_c(returns, weights, years, n_sims, initial_value, c_engine):
    """Wrapper for C Monte Carlo simulation"""
    results = c_engine.run_monte_carlo_simulation(
        returns, weights, simulation_years=years,
        n_simulations=n_sims, initial_value=initial_value
    )
    return results['simulations']

# Integration example for your main app
def integrate_c_engine():
    """Example of how to integrate C engine into your Streamlit app"""
    
    # Initialize C engine (do this once at app startup)
    try:
        c_engine = PortfolioEngineC("./portfolio_engine.so")
        return c_engine
    except FileNotFoundError:
        print("C library not found. Falling back to Python implementation.")
        return None

# Modified display_statistics function to use C engine
def display_statistics_c(returns, selected_weights, simulation_years, num_simulations, 
                        initial_investment, c_engine):
    """Modified display function using C engine"""
    
    if c_engine is not None:
        # Use C engine
        metrics, simulations = calculate_advanced_metrics_c(
            returns, selected_weights, c_engine, simulation_years, 
            num_simulations, initial_investment
        )
    else:
        # Fallback to Python
        simulations = monte_carlo_simulation(returns, selected_weights, 
                                           simulation_years, num_simulations, 
                                           initial_investment)
        metrics = calculate_advanced_metrics(returns, selected_weights, simulations)
    
    # Rest of your display logic remains the same
    # ... existing plotting and display code ...