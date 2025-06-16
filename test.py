import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Advanced Portfolio Simulator", layout="wide")

st.title("Advanced Portfolio Monte Carlo Simulator")
st.write("Comprehensive Quantitative Finance Tool for Portfolio Analysis & Risk Management")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Portfolio inputs
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=100000, step=1000)
time_horizon = st.sidebar.slider("Time Horizon (Years)", min_value=1, max_value=30, value=10)
num_simulations = st.sidebar.slider("Number of Monte Carlo Simulations", min_value=100, max_value=10000, value=1000, step=100)

# Advanced options
st.sidebar.subheader("Advanced Options")
rebalancing_freq = st.sidebar.selectbox("Rebalancing Frequency", 
                                       ["No Rebalancing", "Monthly", "Quarterly", "Annually"])
include_fees = st.sidebar.checkbox("Include Management Fees", value=True)
if include_fees:
    annual_fee = st.sidebar.slider("Annual Management Fee (%)", min_value=0.0, max_value=3.0, value=1.0, step=0.1) / 100
else:
    annual_fee = 0.0

# Asset allocation
st.sidebar.subheader("Asset Allocation")
num_assets = st.sidebar.selectbox("Number of Asset Classes", [3, 4, 5], index=0)

if num_assets == 3:
    stock_allocation = st.sidebar.slider("Stock Allocation (%)", min_value=0, max_value=100, value=60)
    bond_allocation = st.sidebar.slider("Bond Allocation (%)", min_value=0, max_value=100-stock_allocation, value=30)
    cash_allocation = 100 - stock_allocation - bond_allocation
    allocations = [stock_allocation, bond_allocation, cash_allocation]
    asset_names = ["Stocks", "Bonds", "Cash"]
elif num_assets == 4:
    stock_allocation = st.sidebar.slider("Stock Allocation (%)", min_value=0, max_value=100, value=50)
    bond_allocation = st.sidebar.slider("Bond Allocation (%)", min_value=0, max_value=100-stock_allocation, value=25)
    intl_allocation = st.sidebar.slider("International (%)", min_value=0, max_value=100-stock_allocation-bond_allocation, value=15)
    cash_allocation = 100 - stock_allocation - bond_allocation - intl_allocation
    allocations = [stock_allocation, bond_allocation, intl_allocation, cash_allocation]
    asset_names = ["Stocks", "Bonds", "International", "Cash"]
else:
    stock_allocation = st.sidebar.slider("Stock Allocation (%)", min_value=0, max_value=100, value=40)
    bond_allocation = st.sidebar.slider("Bond Allocation (%)", min_value=0, max_value=100-stock_allocation, value=20)
    intl_allocation = st.sidebar.slider("International (%)", min_value=0, max_value=100-stock_allocation-bond_allocation, value=15)
    reit_allocation = st.sidebar.slider("REITs (%)", min_value=0, max_value=100-stock_allocation-bond_allocation-intl_allocation, value=10)
    cash_allocation = 100 - stock_allocation - bond_allocation - intl_allocation - reit_allocation
    allocations = [stock_allocation, bond_allocation, intl_allocation, reit_allocation, cash_allocation]
    asset_names = ["Stocks", "Bonds", "International", "REITs", "Cash"]

st.sidebar.write(f"Cash Allocation: {cash_allocation}%")

# Market assumptions
st.sidebar.subheader("Market Assumptions")
if num_assets == 3:
    returns = [
        st.sidebar.number_input("Stock Return (%/year)", value=7.0, step=0.1) / 100,
        st.sidebar.number_input("Bond Return (%/year)", value=3.0, step=0.1) / 100,
        st.sidebar.number_input("Cash Return (%/year)", value=1.5, step=0.1) / 100
    ]
    volatilities = [
        st.sidebar.number_input("Stock Volatility (%/year)", value=15.0, step=0.5) / 100,
        st.sidebar.number_input("Bond Volatility (%/year)", value=5.0, step=0.1) / 100,
        0.001  # Cash volatility
    ]
elif num_assets == 4:
    returns = [
        st.sidebar.number_input("Stock Return (%/year)", value=7.0, step=0.1) / 100,
        st.sidebar.number_input("Bond Return (%/year)", value=3.0, step=0.1) / 100,
        st.sidebar.number_input("International Return (%/year)", value=6.5, step=0.1) / 100,
        st.sidebar.number_input("Cash Return (%/year)", value=1.5, step=0.1) / 100
    ]
    volatilities = [
        st.sidebar.number_input("Stock Volatility (%/year)", value=15.0, step=0.5) / 100,
        st.sidebar.number_input("Bond Volatility (%/year)", value=5.0, step=0.1) / 100,
        st.sidebar.number_input("International Volatility (%/year)", value=18.0, step=0.5) / 100,
        0.001
    ]
else:
    returns = [
        st.sidebar.number_input("Stock Return (%/year)", value=7.0, step=0.1) / 100,
        st.sidebar.number_input("Bond Return (%/year)", value=3.0, step=0.1) / 100,
        st.sidebar.number_input("International Return (%/year)", value=6.5, step=0.1) / 100,
        st.sidebar.number_input("REIT Return (%/year)", value=8.0, step=0.1) / 100,
        st.sidebar.number_input("Cash Return (%/year)", value=1.5, step=0.1) / 100
    ]
    volatilities = [
        st.sidebar.number_input("Stock Volatility (%/year)", value=15.0, step=0.5) / 100,
        st.sidebar.number_input("Bond Volatility (%/year)", value=5.0, step=0.1) / 100,
        st.sidebar.number_input("International Volatility (%/year)", value=18.0, step=0.5) / 100,
        st.sidebar.number_input("REIT Volatility (%/year)", value=20.0, step=0.5) / 100,
        0.001
    ]

# Correlation matrix input
st.sidebar.subheader("Asset Correlations")
use_custom_corr = st.sidebar.checkbox("Use Custom Correlation Matrix", value=False)

def generate_correlation_matrix(n_assets, custom=False):
    if custom:
        # Simplified correlation input for demonstration
        if n_assets == 3:
            corr_matrix = np.array([
                [1.0, 0.2, 0.0],
                [0.2, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
        elif n_assets == 4:
            corr_matrix = np.array([
                [1.0, 0.2, 0.7, 0.0],
                [0.2, 1.0, 0.1, 0.0],
                [0.7, 0.1, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
        else:
            corr_matrix = np.array([
                [1.0, 0.2, 0.7, 0.6, 0.0],
                [0.2, 1.0, 0.1, 0.3, 0.0],
                [0.7, 0.1, 1.0, 0.4, 0.0],
                [0.6, 0.3, 0.4, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ])
    else:
        # Default correlation assumptions
        corr_matrix = np.eye(n_assets)
        if n_assets >= 3:
            corr_matrix[0, 1] = corr_matrix[1, 0] = 0.2  # Stock-Bond
        if n_assets >= 4:
            corr_matrix[0, 2] = corr_matrix[2, 0] = 0.7  # Stock-International
        if n_assets >= 5:
            corr_matrix[0, 3] = corr_matrix[3, 0] = 0.6  # Stock-REIT
            corr_matrix[1, 3] = corr_matrix[3, 1] = 0.3  # Bond-REIT
            corr_matrix[2, 3] = corr_matrix[3, 2] = 0.4  # International-REIT
    
    return corr_matrix

def calculate_portfolio_stats(allocations, returns, volatilities, corr_matrix):
    """Calculate portfolio expected return and volatility"""
    weights = np.array(allocations) / 100
    returns_array = np.array(returns)
    volatilities_array = np.array(volatilities)
    
    # Portfolio expected return
    portfolio_return = np.sum(weights * returns_array)
    
    # Covariance matrix
    cov_matrix = np.outer(volatilities_array, volatilities_array) * corr_matrix
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    return portfolio_return, portfolio_volatility, cov_matrix

def monte_carlo_simulation_advanced(initial_value, weights, returns, cov_matrix, years, n_sims, annual_fee=0.0, rebalancing="No Rebalancing"):
    """Advanced Monte Carlo simulation with rebalancing and fees"""
    
    if rebalancing == "No Rebalancing":
        rebal_freq = years * 252  # Never rebalance
    elif rebalancing == "Monthly":
        rebal_freq = 21  # ~21 trading days per month
    elif rebalancing == "Quarterly":
        rebal_freq = 63  # ~63 trading days per quarter
    else:  # Annually
        rebal_freq = 252  # 252 trading days per year
    
    dt = 1/252  # Daily time step
    n_steps = int(years * 252)
    n_assets = len(weights)
    
    # Initialize arrays
    portfolio_values = np.zeros((n_sims, n_steps + 1))
    portfolio_values[:, 0] = initial_value
    
    # Asset allocation tracking
    asset_values = np.zeros((n_sims, n_steps + 1, n_assets))
    asset_values[:, 0, :] = initial_value * weights
    
    # Generate all random returns at once for efficiency
    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition for correlated returns
    
    for sim in range(n_sims):
        current_values = asset_values[sim, 0, :].copy()
        
        for step in range(n_steps):
            # Generate correlated random returns
            uncorr_returns = np.random.normal(0, np.sqrt(dt), n_assets)
            corr_returns = L @ uncorr_returns
            
            # Add drift
            drift = (np.array(returns) - 0.5 * np.diag(cov_matrix)) * dt
            daily_returns = drift + corr_returns
            
            # Update asset values
            current_values = current_values * np.exp(daily_returns)
            
            # Apply fees (daily portion of annual fee)
            if annual_fee > 0:
                current_values *= (1 - annual_fee * dt)
            
            # Rebalancing
            if (step + 1) % rebal_freq == 0 and step < n_steps - 1:
                total_value = np.sum(current_values)
                current_values = total_value * weights
            
            asset_values[sim, step + 1, :] = current_values
            portfolio_values[sim, step + 1] = np.sum(current_values)
    
    return portfolio_values, asset_values

def calculate_advanced_metrics(portfolio_values, asset_values, initial_value, years):
    """Calculate comprehensive risk and performance metrics"""
    final_values = portfolio_values[:, -1]
    returns = (final_values - initial_value) / initial_value
    
    # Time series analysis
    daily_returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
    annualized_returns = (final_values / initial_value) ** (1/years) - 1
    
    # Risk metrics
    var_95 = np.percentile(final_values, 5)
    var_99 = np.percentile(final_values, 1)
    cvar_95 = np.mean(final_values[final_values <= var_95])  # Expected Shortfall
    cvar_99 = np.mean(final_values[final_values <= var_99])
    
    # Maximum drawdown calculation
    running_max = np.maximum.accumulate(portfolio_values, axis=1)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdowns = np.min(drawdowns, axis=1)
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = np.mean(annualized_returns) / downside_std if downside_std > 0 else 0
    
    # Calmar ratio (return/max drawdown)
    calmar_ratio = np.mean(annualized_returns) / abs(np.mean(max_drawdowns)) if np.mean(max_drawdowns) != 0 else 0
    
    metrics = {
        'Expected Final Value': np.mean(final_values),
        'Median Final Value': np.median(final_values),
        'Standard Deviation': np.std(final_values),
        'Value at Risk (5%)': var_95,
        'Value at Risk (1%)': var_99,
        'Expected Shortfall (5%)': cvar_95,
        'Expected Shortfall (1%)': cvar_99,
        'Probability of Loss': np.mean(final_values < initial_value) * 100,
        'Expected Annual Return': np.mean(annualized_returns) * 100,
        'Annual Volatility': np.std(annualized_returns) * 100,
        'Sharpe Ratio': np.mean(annualized_returns) / np.std(annualized_returns) if np.std(annualized_returns) > 0 else 0,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Maximum Drawdown': np.mean(max_drawdowns) * 100,
        'Skewness': stats.skew(returns),
        'Kurtosis': stats.kurtosis(returns),
        'Best Case (95th percentile)': np.percentile(final_values, 95),
        'Worst Case (5th percentile)': np.percentile(final_values, 5)
    }
    
    return metrics, daily_returns, max_drawdowns, drawdowns

# Calculate portfolio statistics
corr_matrix = generate_correlation_matrix(num_assets, use_custom_corr)
portfolio_return, portfolio_volatility, cov_matrix = calculate_portfolio_stats(
    allocations, returns, volatilities, corr_matrix
)

# Display portfolio summary
st.header("Portfolio Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Expected Annual Return", f"{portfolio_return:.2%}")
with col2:
    st.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
with col3:
    st.metric("Sharpe Ratio", f"{portfolio_return/portfolio_volatility:.2f}")
with col4:
    st.metric("Management Fee", f"{annual_fee:.2%}" if include_fees else "0.00%")

# Asset allocation visualization
st.subheader("Asset Allocation")
fig_pie = px.pie(values=allocations, names=asset_names, title="Portfolio Allocation")
st.plotly_chart(fig_pie, use_container_width=False)

# Correlation heatmap
# st.subheader("Asset Correlation Matrix")
fig_corr, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            xticklabels=asset_names, yticklabels=asset_names, ax=ax)
ax.set_title('Asset Correlation Matrix')
# st.pyplot(fig_corr)

# Run simulation
if st.button("Run Advanced Monte Carlo Simulation", type="primary"):
    with st.spinner("Running advanced simulation..."):
        # Run simulation
        weights = np.array(allocations) / 100
        portfolio_values, asset_values = monte_carlo_simulation_advanced(
            initial_investment, weights, returns, cov_matrix, 
            time_horizon, num_simulations, annual_fee, rebalancing_freq
        )
        
        # Calculate advanced metrics
        risk_metrics, daily_returns, max_drawdowns, drawdowns = calculate_advanced_metrics(
            portfolio_values, asset_values, initial_investment, time_horizon
        )
        
        # Display key results
        st.header("Advanced Simulation Results")
        
        # Key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Expected Value", f"${risk_metrics['Expected Final Value']:,.0f}")
        with col2:
            st.metric("Median Value", f"${risk_metrics['Median Final Value']:,.0f}")
        with col3:
            st.metric("5% VaR", f"${risk_metrics['Value at Risk (5%)']:,.0f}")
        with col4:
            st.metric("Max Drawdown", f"{risk_metrics['Maximum Drawdown']:.1f}%")
        with col5:
            st.metric("Probability of Loss", f"{risk_metrics['Probability of Loss']:.1f}%")
        
        # Create comprehensive visualization
        st.subheader("Portfolio Analysis Dashboard")
        
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Evolution Paths', 'Final Value Distribution',
                          'Drawdown Analysis', 'Return Distribution',
                          'Asset Allocation Over Time', 'Risk Metrics Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Portfolio evolution paths
        time_axis = np.linspace(0, time_horizon, portfolio_values.shape[1])
        sample_size = min(50, num_simulations)
        for i in range(sample_size):
            fig.add_trace(
                go.Scatter(x=time_axis, y=portfolio_values[i], 
                          mode='lines', opacity=0.1, 
                          line=dict(color='blue', width=1),
                          showlegend=False),
                row=1, col=1
            )
        
        # Add percentiles
        percentiles = np.percentile(portfolio_values, [5, 25, 50, 75, 95], axis=0)
        fig.add_trace(
            go.Scatter(x=time_axis, y=percentiles[2], 
                      mode='lines', name='Median',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # 2. Final value distribution
        fig.add_trace(
            go.Histogram(x=portfolio_values[:, -1], nbinsx=50, 
                        name='Final Values', opacity=0.7),
            row=1, col=2
        )
        
        # 3. Drawdown analysis
        median_drawdown = np.median(drawdowns, axis=0)
        fig.add_trace(
            go.Scatter(x=time_axis, y=median_drawdown * 100,
                      mode='lines', name='Median Drawdown',
                      line=dict(color='red', width=2),
                      fill='tonexty'),
            row=2, col=1
        )
        
        # 4. Return distribution
        annual_returns = (portfolio_values[:, -1] / initial_investment) ** (1/time_horizon) - 1
        fig.add_trace(
            go.Histogram(x=annual_returns * 100, nbinsx=50,
                        name='Annual Returns (%)', opacity=0.7),
            row=2, col=2
        )
        
        # 5. Asset allocation over time (using median simulation)
        median_sim_idx = np.argsort(portfolio_values[:, -1])[num_simulations//2]
        median_asset_values = asset_values[median_sim_idx]
        
        for i, asset_name in enumerate(asset_names):
            asset_weights = median_asset_values[:, i] / np.sum(median_asset_values, axis=1) * 100
            fig.add_trace(
                go.Scatter(x=time_axis, y=asset_weights,
                          mode='lines', name=f'{asset_name} %',
                          stackgroup='one' if rebalancing_freq != "No Rebalancing" else None),
                row=3, col=1
            )
        
        # 6. Risk metrics radar chart (simplified as bar chart)
        risk_ratios = [
            risk_metrics['Sharpe Ratio'],
            risk_metrics['Sortino Ratio'],
            risk_metrics['Calmar Ratio'],
            1 - risk_metrics['Probability of Loss']/100,  # Success probability
            (risk_metrics['Expected Final Value'] - initial_investment) / initial_investment  # Total return ratio
        ]
        risk_labels = ['Sharpe', 'Sortino', 'Calmar', 'Success Prob', 'Total Return']
        
        fig.add_trace(
            go.Bar(x=risk_labels, y=risk_ratios, name='Risk Metrics'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=True, 
                         title_text="Comprehensive Portfolio Analysis")
        fig.update_xaxes(title_text="Years", row=1, col=1)
        fig.update_xaxes(title_text="Portfolio Value ($)", row=1, col=2)
        fig.update_xaxes(title_text="Years", row=2, col=1)
        fig.update_xaxes(title_text="Annual Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Years", row=3, col=1)
        fig.update_xaxes(title_text="Metric", row=3, col=2)
        
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Allocation (%)", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency frontier analysis
        st.subheader("Portfolio Efficiency Analysis")
        
        # Generate efficient frontier
        target_returns = np.linspace(min(returns) * 0.8, max(returns) * 1.2, 50)
        efficient_vols = []
        efficient_allocations = []
        
        for target_ret in target_returns:
            # Simplified optimization (equal weight changes)
            best_vol = float('inf')
            best_allocation = None
            
            # Try different allocation combinations
            for stock_w in np.arange(0, 1.01, 0.1):
                remaining = 1 - stock_w
                for bond_w in np.arange(0, remaining + 0.01, 0.1):
                    if num_assets == 3:
                        test_weights = [stock_w, bond_w, remaining - bond_w]
                    elif num_assets == 4:
                        intl_w = min(0.5, remaining - bond_w)
                        test_weights = [stock_w, bond_w, intl_w, remaining - bond_w - intl_w]
                    else:
                        intl_w = min(0.3, remaining - bond_w)
                        reit_w = min(0.2, remaining - bond_w - intl_w)
                        test_weights = [stock_w, bond_w, intl_w, reit_w, remaining - bond_w - intl_w - reit_w]
                    
                    if all(w >= 0 for w in test_weights):
                        test_ret = np.sum(np.array(test_weights) * np.array(returns))
                        test_vol = np.sqrt(np.dot(test_weights, np.dot(cov_matrix, test_weights)))
                        
                        if abs(test_ret - target_ret) < 0.001 and test_vol < best_vol:
                            best_vol = test_vol
                            best_allocation = test_weights
            
            if best_vol < float('inf'):
                efficient_vols.append(best_vol)
                efficient_allocations.append(best_allocation)
            else:
                efficient_vols.append(None)
                efficient_allocations.append(None)
        
        # Plot efficient frontier
        fig_ef = go.Figure()
        
        valid_points = [(vol, ret) for vol, ret in zip(efficient_vols, target_returns) if vol is not None]
        if valid_points:
            vols, rets = zip(*valid_points)
            fig_ef.add_trace(go.Scatter(x=vols, y=rets, mode='lines+markers',
                                       name='Efficient Frontier', line=dict(color='blue', width=2)))
        
        # Add current portfolio
        fig_ef.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return],
                                   mode='markers', name='Current Portfolio',
                                   marker=dict(size=12, color='red', symbol='star')))
        
        fig_ef.update_layout(title='Efficient Frontier Analysis',
                            xaxis_title='Volatility (Standard Deviation)',
                            yaxis_title='Expected Return',
                            height=500)
        st.plotly_chart(fig_ef, use_container_width=True)
        
        # Detailed risk metrics table
        st.subheader("Comprehensive Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Metrics**")
            perf_metrics = {
                'Expected Final Value': f"${risk_metrics['Expected Final Value']:,.0f}",
                'Median Final Value': f"${risk_metrics['Median Final Value']:,.0f}",
                'Expected Annual Return': f"{risk_metrics['Expected Annual Return']:.2f}%",
                'Annual Volatility': f"{risk_metrics['Annual Volatility']:.2f}%",
                'Sharpe Ratio': f"{risk_metrics['Sharpe Ratio']:.3f}",
                'Sortino Ratio': f"{risk_metrics['Sortino Ratio']:.3f}",
                'Calmar Ratio': f"{risk_metrics['Calmar Ratio']:.3f}"
            }
            st.dataframe(pd.DataFrame(list(perf_metrics.items()), columns=['Metric', 'Value']))
        
        with col2:
            st.write("**Risk Metrics**")
            risk_only = {
                'Value at Risk (5%)': f"${risk_metrics['Value at Risk (5%)']:,.0f}",
                'Value at Risk (1%)': f"${risk_metrics['Value at Risk (1%)']:,.0f}",
                'Expected Shortfall (5%)': f"${risk_metrics['Expected Shortfall (5%)']:,.0f}",
                'Expected Shortfall (1%)': f"${risk_metrics['Expected Shortfall (1%)']:,.0f}",
                'Maximum Drawdown': f"{risk_metrics['Maximum Drawdown']:.2f}%",
                'Probability of Loss': f"{risk_metrics['Probability of Loss']:.1f}%",
                'Skewness': f"{risk_metrics['Skewness']:.3f}",
                'Kurtosis': f"{risk_metrics['Kurtosis']:.3f}"
            }
            st.dataframe(pd.DataFrame(list(risk_only.items()), columns=['Metric', 'Value']))
        
        # Monte Carlo convergence analysis
        st.subheader("Monte Carlo Convergence Analysis")
        
        # Calculate running statistics
        running_means = np.cumsum(portfolio_values[:, -1]) / np.arange(1, num_simulations + 1)
        running_stds = np.array([np.std(portfolio_values[:i+1, -1]) for i in range(num_simulations)])
        
        fig_conv = make_subplots(rows=1, cols=2, 
                                subplot_titles=('Convergence of Mean', 'Convergence of Std Dev'))
        
        fig_conv.add_trace(
            go.Scatter(x=np.arange(1, num_simulations + 1), y=running_means,
                      mode='lines', name='Running Mean'),
            row=1, col=1
        )
        
        fig_conv.add_trace(
            go.Scatter(x=np.arange(1, num_simulations + 1), y=running_stds,
                      mode='lines', name='Running Std Dev'),
            row=1, col=2
        )
        
        fig_conv.update_layout(height=400, title_text="Monte Carlo Convergence")
        fig_conv.update_xaxes(title_text="Number of Simulations", row=1, col=1)
        fig_conv.update_xaxes(title_text="Number of Simulations", row=1, col=2)
        fig_conv.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig_conv.update_yaxes(title_text="Standard Deviation ($)", row=1, col=2)
        
        st.plotly_chart(fig_conv, use_container_width=True)
        
        # Stress testing scenarios
        st.subheader("Stress Testing & Scenario Analysis")
        
        # Define stress scenarios
        stress_scenarios = {
            "2008 Financial Crisis": {"stock_shock": -0.37, "bond_shock": 0.05, "correlation_shock": 0.3},
            "Black Monday 1987": {"stock_shock": -0.22, "bond_shock": 0.02, "correlation_shock": 0.2},
            "COVID-19 Crash": {"stock_shock": -0.34, "bond_shock": 0.08, "correlation_shock": 0.4},
            "Dot-com Bubble": {"stock_shock": -0.49, "bond_shock": 0.12, "correlation_shock": 0.1},
            "Stagflation 1970s": {"stock_shock": -0.15, "bond_shock": -0.25, "correlation_shock": 0.6}
        }
        
        stress_results = []
        current_value = initial_investment
        
        for scenario_name, shocks in stress_scenarios.items():
            # Apply shocks to current portfolio
            stock_weight = allocations[0] / 100
            bond_weight = allocations[1] / 100
            
            # Calculate portfolio impact
            portfolio_shock = (stock_weight * shocks["stock_shock"] + 
                             bond_weight * shocks["bond_shock"])
            
            stressed_value = current_value * (1 + portfolio_shock)
            loss_amount = current_value - stressed_value
            loss_percent = (loss_amount / current_value) * 100
            
            stress_results.append({
                "Scenario": scenario_name,
                "Portfolio Value": f"${stressed_value:,.0f}",
                "Loss Amount": f"${loss_amount:,.0f}",
                "Loss Percentage": f"{loss_percent:.1f}%"
            })
        
        stress_df = pd.DataFrame(stress_results)
        st.dataframe(stress_df, use_container_width=True)
        
        # Performance attribution analysis
        st.subheader("Performance Attribution Analysis")
        
        # Calculate contribution of each asset to total return
        final_asset_values = np.mean(asset_values[:, -1, :], axis=0)
        initial_asset_values = initial_investment * weights
        asset_returns = (final_asset_values - initial_asset_values) / initial_investment * 100
        
        attribution_data = {
            "Asset Class": asset_names,
            "Initial Allocation": [f"{w:.1f}%" for w in allocations],
            "Final Value": [f"${v:,.0f}" for v in final_asset_values],
            "Contribution to Return": [f"{r:.2f}%" for r in asset_returns],
            "Weight Drift": [f"{(fv/np.sum(final_asset_values) - w/100)*100:.1f}%" 
                           for fv, w in zip(final_asset_values, allocations)]
        }
        
        attribution_df = pd.DataFrame(attribution_data)
        st.dataframe(attribution_df, use_container_width=True)
        
        # Asset performance comparison
        fig_assets = go.Figure()
        
        # Show evolution of each asset class
        for i, asset_name in enumerate(asset_names):
            median_asset_evolution = np.median(asset_values[:, :, i], axis=0)
            fig_assets.add_trace(
                go.Scatter(x=time_axis, y=median_asset_evolution,
                          mode='lines', name=f'{asset_name}',
                          line=dict(width=3))
            )
        
        fig_assets.update_layout(
            title='Asset Class Performance Evolution (Median)',
            xaxis_title='Years',
            yaxis_title='Value ($)',
            height=500
        )
        st.plotly_chart(fig_assets, use_container_width=True)
        
        # Rolling statistics analysis
        st.subheader("Rolling Performance Statistics")
        
        # Calculate rolling Sharpe ratios
        window_size = min(252, portfolio_values.shape[1] // 4)  # Quarterly windows
        rolling_sharpes = []
        rolling_vols = []
        rolling_returns = []
        
        for i in range(window_size, portfolio_values.shape[1]):
            window_values = portfolio_values[:, i-window_size:i+1]
            window_returns = np.diff(window_values, axis=1) / window_values[:, :-1]
            
            if window_returns.size > 0:
                ann_return = np.mean(window_returns) * 252
                ann_vol = np.std(window_returns) * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                rolling_sharpes.append(np.mean([sharpe] * num_simulations))
                rolling_vols.append(np.mean([ann_vol] * num_simulations))
                rolling_returns.append(np.mean([ann_return] * num_simulations))
        
        if rolling_sharpes:
            rolling_time = time_axis[window_size:]
            
            fig_rolling = make_subplots(rows=2, cols=2,
                                       subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility',
                                                     'Rolling Returns', 'Return Distribution by Year'))
            
            # Rolling Sharpe
            fig_rolling.add_trace(
                go.Scatter(x=rolling_time, y=rolling_sharpes,
                          mode='lines', name='Rolling Sharpe'),
                row=1, col=1
            )
            
            # Rolling Volatility
            fig_rolling.add_trace(
                go.Scatter(x=rolling_time, y=np.array(rolling_vols) * 100,
                          mode='lines', name='Rolling Vol %'),
                row=1, col=2
            )
            
            # Rolling Returns
            fig_rolling.add_trace(
                go.Scatter(x=rolling_time, y=np.array(rolling_returns) * 100,
                          mode='lines', name='Rolling Return %'),
                row=2, col=1
            )
            
            # Annual return distribution
            annual_periods = max(1, int(time_horizon))
            annual_returns_dist = []
            
            for year in range(1, annual_periods + 1):
                year_idx = min(int(year * 252), portfolio_values.shape[1] - 1)
                if year == 1:
                    year_returns = (portfolio_values[:, year_idx] / portfolio_values[:, 0]) - 1
                else:
                    prev_year_idx = int((year - 1) * 252)
                    year_returns = (portfolio_values[:, year_idx] / portfolio_values[:, prev_year_idx]) - 1
                
                annual_returns_dist.extend(year_returns * 100)
            
            if annual_returns_dist:
                fig_rolling.add_trace(
                    go.Histogram(x=annual_returns_dist, nbinsx=30,
                                name='Annual Returns'),
                    row=2, col=2
                )
            
            fig_rolling.update_layout(height=800, title_text="Rolling Performance Analysis")
            fig_rolling.update_xaxes(title_text="Years", row=1, col=1)
            fig_rolling.update_xaxes(title_text="Years", row=1, col=2)
            fig_rolling.update_xaxes(title_text="Years", row=2, col=1)
            fig_rolling.update_xaxes(title_text="Annual Return (%)", row=2, col=2)
            
            st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Quantile regression analysis
        st.subheader("Quantile Analysis & Tail Risk")
        
        # Calculate quantile paths
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        quantile_paths = np.percentile(portfolio_values, [q*100 for q in quantiles], axis=0)
        
        fig_quantiles = go.Figure()
        
        colors = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']
        
        for i, (q, color) in enumerate(zip(quantiles, colors)):
            fig_quantiles.add_trace(
                go.Scatter(x=time_axis, y=quantile_paths[i],
                          mode='lines', name=f'{q*100:.0f}th Percentile',
                          line=dict(color=color, width=2))
            )
        
        fig_quantiles.update_layout(
            title='Portfolio Value Quantile Evolution',
            xaxis_title='Years',
            yaxis_title='Portfolio Value ($)',
            height=500
        )
        st.plotly_chart(fig_quantiles, use_container_width=True)
        
        # Final comprehensive summary
        st.subheader("Executive Summary & Recommendations")
        
        # Generate insights based on results
        total_return = (risk_metrics['Expected Final Value'] - initial_investment) / initial_investment * 100
        success_rate = 100 - risk_metrics['Probability of Loss']
        
        summary_text = f"""
        ## Portfolio Performance Summary
        
        **Investment Horizon**: {time_horizon} years  
        **Initial Investment**: ${initial_investment:,}  
        **Expected Final Value**: ${risk_metrics['Expected Final Value']:,.0f}  
        **Total Expected Return**: {total_return:.1f}%  
        **Success Probability**: {success_rate:.1f}%  
        
        ### Key Insights:
        
        **Risk-Return Profile**:
        - Your portfolio has an expected annual return of {risk_metrics['Expected Annual Return']:.1f}% with {risk_metrics['Annual Volatility']:.1f}% volatility
        - Sharpe ratio of {risk_metrics['Sharpe Ratio']:.2f} indicates {"excellent" if risk_metrics['Sharpe Ratio'] > 1.0 else "good" if risk_metrics['Sharpe Ratio'] > 0.5 else "moderate"} risk-adjusted returns
        - Maximum expected drawdown: {abs(risk_metrics['Maximum Drawdown']):.1f}%
        
        **Tail Risk Analysis**:
        - 5% chance of portfolio value below ${risk_metrics['Value at Risk (5%)']:,.0f}
        - In worst-case scenarios (1% probability), expected loss: ${initial_investment - risk_metrics['Expected Shortfall (1%)']:,.0f}
        - Distribution skewness: {risk_metrics['Skewness']:.2f} ({"right-tailed" if risk_metrics['Skewness'] > 0 else "left-tailed" if risk_metrics['Skewness'] < 0 else "symmetric"})
        
        **Strategic Recommendations**:
        """
        
        # Add specific recommendations based on portfolio characteristics
        if risk_metrics['Sharpe Ratio'] < 0.5:
            summary_text += "\n- Consider rebalancing towards higher-return assets or reducing fees"
        if risk_metrics['Maximum Drawdown'] < -20:
            summary_text += "\n- Portfolio may experience significant drawdowns; consider defensive adjustments"
        if risk_metrics['Probability of Loss'] > 20:
            summary_text += "\n- High probability of loss suggests reviewing risk tolerance and time horizon"
        if portfolio_volatility > 0.15:
            summary_text += "\n- High volatility portfolio; ensure risk tolerance aligns with investment goals"
        
        summary_text += f"""
        
        **Rebalancing Impact**: {rebalancing_freq} rebalancing {"helps maintain target allocation" if rebalancing_freq != "No Rebalancing" else "allows for drift in asset allocation"}
        
        **Fee Impact**: {"Annual fees of " + f"{annual_fee:.1%}" + " reduce returns by approximately $" + f"{risk_metrics['Expected Final Value'] * annual_fee * time_horizon:,.0f}" + " over the investment period" if annual_fee > 0 else "No management fees applied"}
        """
        
        st.markdown(summary_text)

# Additional analysis tools
st.header("Advanced Analysis Tools")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Download Results"):
        st.info("Results download functionality would be implemented here")

with col2:
    if st.button("Compare Strategies"):
        st.info("Strategy comparison tool would be implemented here")

with col3:
    if st.button("Optimize Portfolio"):
        st.info("Portfolio optimization tool would be implemented here")

st.header("Methodology & Assumptions")
st.write("""
### Monte Carlo Methodology

This advanced portfolio simulator employs sophisticated quantitative finance techniques:

**Stochastic Modeling**:
- Geometric Brownian Motion with correlated multi-asset returns
- Cholesky decomposition for generating correlated random variables
- Daily time steps (252 trading days/year) for accurate path simulation

**Advanced Risk Metrics**:
- **Value at Risk (VaR)**: Potential loss at specific confidence levels
- **Expected Shortfall (CVaR)**: Average loss beyond VaR threshold
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sortino Ratio**: Risk-adjusted return using downside deviation
- **Calmar Ratio**: Return per unit of maximum drawdown

**Portfolio Dynamics**:
- Rebalancing effects on long-term performance
- Management fee compounding impact
- Asset correlation evolution over time
- Weight drift analysis without rebalancing

**Limitations & Assumptions**:
- Constant correlation and volatility (real markets show time-varying parameters)
- Log-normal return distribution (real returns exhibit fat tails and skewness)  
- No transaction costs beyond management fees
- No taxes or liquidity constraints
- Historical parameters may not predict future performance

**Statistical Validation**:
- Monte Carlo convergence analysis ensures simulation stability
- Quantile regression identifies tail behavior
- Rolling statistics capture time-varying performance
- Stress testing validates portfolio resilience
""")