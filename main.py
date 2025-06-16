import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

st.title("Portfolio Monte Carlo Simulator")
st.write("Quantitative Finance Tool for Portfolio Future Value Simulation")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Portfolio inputs
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=100000, step=1000)
time_horizon = st.sidebar.slider("Time Horizon (Years)", min_value=1, max_value=30, value=10)
num_simulations = st.sidebar.slider("Number of Monte Carlo Simulations", min_value=100, max_value=10000, value=1000, step=100)

# Asset allocation
st.sidebar.subheader("Asset Allocation")
stock_allocation = st.sidebar.slider("Stock Allocation (%)", min_value=0, max_value=100, value=60)
bond_allocation = st.sidebar.slider("Bond Allocation (%)", min_value=0, max_value=100-stock_allocation, value=30)
cash_allocation = 100 - stock_allocation - bond_allocation
st.sidebar.write(f"Cash Allocation: {cash_allocation}%")

# Market assumptions
st.sidebar.subheader("Market Assumptions")
stock_return = st.sidebar.number_input("Expected Stock Return (%/year)", min_value=0.0, max_value=20.0, value=7.0, step=0.1) / 100
stock_volatility = st.sidebar.number_input("Stock Volatility (%/year)", min_value=5.0, max_value=50.0, value=15.0, step=0.5) / 100
bond_return = st.sidebar.number_input("Expected Bond Return (%/year)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
bond_volatility = st.sidebar.number_input("Bond Volatility (%/year)", min_value=1.0, max_value=20.0, value=5.0, step=0.1) / 100
cash_return = st.sidebar.number_input("Cash Return (%/year)", min_value=0.0, max_value=5.0, value=1.5, step=0.1) / 100
correlation = st.sidebar.slider("Stock-Bond Correlation", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)

def calculate_portfolio_stats(stock_alloc, bond_alloc, cash_alloc, stock_ret, bond_ret, cash_ret, stock_vol, bond_vol, corr):
    """Calculate portfolio expected return and volatility"""
    weights = np.array([stock_alloc/100, bond_alloc/100, cash_alloc/100])
    returns = np.array([stock_ret, bond_ret, cash_ret])
    
    # Portfolio expected return
    portfolio_return = np.sum(weights * returns)
    
    # Covariance matrix
    cov_matrix = np.array([
        [stock_vol**2, corr * stock_vol * bond_vol, 0],
        [corr * stock_vol * bond_vol, bond_vol**2, 0],
        [0, 0, 0]  # Cash has no volatility correlation
    ])
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    return portfolio_return, portfolio_volatility

def monte_carlo_simulation(initial_value, expected_return, volatility, years, n_sims):
    """Run Monte Carlo simulation for portfolio"""
    dt = 1/252  # Daily time step (252 trading days per year)
    n_steps = int(years * 252)
    
    # Generate random returns using geometric Brownian motion
    random_returns = np.random.normal(
        (expected_return - 0.5 * volatility**2) * dt,
        volatility * np.sqrt(dt),
        (n_sims, n_steps)
    )
    
    # Calculate cumulative returns
    price_paths = initial_value * np.exp(np.cumsum(random_returns, axis=1))
    
    # Add initial value column
    price_paths = np.column_stack([np.full(n_sims, initial_value), price_paths])
    
    return price_paths

def calculate_risk_metrics(final_values, initial_value):
    """Calculate various risk metrics"""
    returns = (final_values - initial_value) / initial_value
    
    metrics = {
        'Expected Final Value': np.mean(final_values),
        'Median Final Value': np.median(final_values),
        'Standard Deviation': np.std(final_values),
        'Value at Risk (5%)': np.percentile(final_values, 5),
        'Value at Risk (1%)': np.percentile(final_values, 1),
        'Probability of Loss': np.mean(final_values < initial_value) * 100,
        'Expected Return (%)': np.mean(returns) * 100,
        'Volatility (%)': np.std(returns) * 100,
        'Sharpe Ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        'Skewness': stats.skew(returns),
        'Kurtosis': stats.kurtosis(returns)
    }
    
    return metrics

# Calculate portfolio statistics
portfolio_return, portfolio_volatility = calculate_portfolio_stats(
    stock_allocation, bond_allocation, cash_allocation,
    stock_return, bond_return, cash_return,
    stock_volatility, bond_volatility, correlation
)

# Display portfolio summary
st.header("Portfolio Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Expected Annual Return", f"{portfolio_return:.2%}")
with col2:
    st.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
with col3:
    st.metric("Sharpe Ratio", f"{portfolio_return/portfolio_volatility:.2f}")

# Run Monte Carlo simulation
if st.button("Run Monte Carlo Simulation"):
    with st.spinner("Running simulation..."):
        # Run simulation
        price_paths = monte_carlo_simulation(
            initial_investment, portfolio_return, portfolio_volatility, 
            time_horizon, num_simulations
        )
        
        # Calculate metrics
        final_values = price_paths[:, -1]
        risk_metrics = calculate_risk_metrics(final_values, initial_investment)
        
        # Display results
        st.header("Simulation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Value", f"${risk_metrics['Expected Final Value']:,.0f}")
        with col2:
            st.metric("Median Value", f"${risk_metrics['Median Final Value']:,.0f}")
        with col3:
            st.metric("5% VaR", f"${risk_metrics['Value at Risk (5%)']:,.0f}")
        with col4:
            st.metric("Probability of Loss", f"{risk_metrics['Probability of Loss']:.1f}%")
        
        # Distribution plot
        st.subheader("Final Portfolio Value Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(initial_investment, color='red', linestyle='--', label='Initial Investment')
        ax1.axvline(risk_metrics['Expected Final Value'], color='green', linestyle='--', label='Expected Value')
        ax1.axvline(risk_metrics['Value at Risk (5%)'], color='orange', linestyle='--', label='5% VaR')
        ax1.set_xlabel('Final Portfolio Value ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        stats.probplot((final_values - np.mean(final_values))/np.std(final_values), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Sample path evolution
        st.subheader("Sample Portfolio Evolution Paths")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sample paths
        sample_paths = price_paths[:min(100, num_simulations)]
        time_axis = np.linspace(0, time_horizon, price_paths.shape[1])
        
        for i in range(len(sample_paths)):
            ax.plot(time_axis, sample_paths[i], alpha=0.1, color='blue')
        
        # Plot percentiles
        percentiles = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=0)
        ax.plot(time_axis, percentiles[2], color='red', linewidth=2, label='Median')
        ax.fill_between(time_axis, percentiles[0], percentiles[4], alpha=0.2, color='gray', label='5th-95th Percentile')
        ax.fill_between(time_axis, percentiles[1], percentiles[3], alpha=0.3, color='gray', label='25th-75th Percentile')
        
        ax.set_xlabel('Years')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Detailed risk metrics table
        st.subheader("Detailed Risk Metrics")
        metrics_df = pd.DataFrame(list(risk_metrics.items()), columns=['Metric', 'Value'])
        
        # Format values
        currency_metrics = ['Expected Final Value', 'Median Final Value', 'Standard Deviation', 
                          'Value at Risk (5%)', 'Value at Risk (1%)']
        percentage_metrics = ['Expected Return (%)', 'Volatility (%)', 'Probability of Loss']
        
        for idx, row in metrics_df.iterrows():
            if row['Metric'] in currency_metrics:
                metrics_df.loc[idx, 'Value'] = f"${row['Value']:,.0f}"
            elif row['Metric'] in percentage_metrics:
                if 'Probability' in row['Metric']:
                    metrics_df.loc[idx, 'Value'] = f"{row['Value']:.1f}%"
                else:
                    metrics_df.loc[idx, 'Value'] = f"{row['Value']:.2f}%"
            elif isinstance(row['Value'], float):
                metrics_df.loc[idx, 'Value'] = f"{row['Value']:.3f}"
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Statistical Analysis")
        st.write(f"""
        **Interpretation of Results:**
        
        - **Expected Portfolio Value**: ${risk_metrics['Expected Final Value']:,.0f} after {time_horizon} years
        - **Risk Assessment**: {risk_metrics['Probability of Loss']:.1f}% chance of losing money
        - **Volatility**: Annual portfolio volatility of {portfolio_volatility:.1%}
        - **Distribution Shape**: Skewness of {risk_metrics['Skewness']:.3f} (0 = normal, >0 = right tail, <0 = left tail)
        - **Tail Risk**: Kurtosis of {risk_metrics['Kurtosis']:.3f} (0 = normal, >0 = fat tails)
        
        **5% Value at Risk**: There's a 5% chance your portfolio will be worth less than ${risk_metrics['Value at Risk (5%)']:,.0f}
        """)

st.header("About This Tool")
st.write("""
This Monte Carlo portfolio simulator demonstrates key quantitative finance concepts:

- **Geometric Brownian Motion**: Models asset price evolution with drift and volatility
- **Portfolio Theory**: Combines multiple assets with correlation effects
- **Monte Carlo Methods**: Simulates thousands of possible future scenarios
- **Risk Metrics**: VaR, Expected Shortfall, and distribution analysis
- **Statistical Analysis**: Tests for normality and measures tail risks

The simulation assumes log-normal returns and constant parameters, which are simplifications of real market behavior.
""")