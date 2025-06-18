import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Portfolio Optimizer & Simulator",
    page_icon="📈",
    layout="wide"
)


if 'optimal_weights_sharpe' not in st.session_state:
    st.session_state.optimal_weights_sharpe = None
if 'optimal_weights_minvol' not in st.session_state:
    st.session_state.optimal_weights_minvol = None
if 'equal_weights' not in st.session_state:
    st.session_state.equal_weights = None

@st.cache_data
def load_ticker_data(csv_file):
    """Load ticker data from CSV file with caching for better performance"""
    try:
        df = pd.read_csv(csv_file)
        # Create a formatted display option: "TICKER - Company Name"
        df['display_option'] = df['TICKER'] + ' - ' + df['NAME']
        return df.dropna()
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file}' not found. Please make sure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

csv_file = "data/combined_stocks.csv" 
df = load_ticker_data(csv_file)

# Title and description
st.title("Portfolio Optimization & Monte Carlo Simulation")
st.markdown("*Quantitative finance tool for portfolio analysis and risk management*")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Default stock symbols
default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

# Stock selection
stocks_input = st.sidebar.text_input(
    "Enter stock symbols (comma-separated):",
    value=','.join(default_stocks[:5]),
    help="Enter stock symbols separated by commas"
)

stocks = [stock.strip().upper() for stock in stocks_input.split(',') if stock.strip()]


# Multi-select widget
selected_options = st.sidebar.multiselect(
    "Choose your tickers:",
    options=df['display_option'].tolist(),
    default=['NVDA - NVIDIA Corporation Common Stock', 'JPM - JP Morgan Chase & Co. Common Stock', 'V - Visa Inc.', 'WMT - Walmart Inc. Common Stock'],
    help="You can select multiple stocks (Ticker - Company Name)."
)

select_tickers = [item.split(' - ')[0] for item in selected_options]


st.caption(selected_options)
st.caption(select_tickers)
stocks = select_tickers

# Time period for historical data
lookback_period = st.sidebar.selectbox(
    "Historical data period:",
    ['1y', '2y', '3y', '5y'],
    index=2
)

# Risk-free rate
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%):",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1
) / 100

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
simulation_years = st.sidebar.slider("Simulation years:", 1, 20, 5)
num_simulations = st.sidebar.slider("Number of simulations:", 100, 10000, 1000, step=100)
initial_investment = st.sidebar.number_input(
    "Initial investment ($):",
    min_value=1000,
    max_value=1000000,
    value=100000,
    step=1000
)

@st.cache_data
def fetch_stock_data(stocks, period):
    """Fetch historical stock data"""
    try:
        data = yf.download(stocks, period=period, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio return, volatility, and Sharpe ratio"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def optimize_portfolio(returns, optimization_type='sharpe'):
    """Optimize portfolio weights"""
    n_assets = len(returns.columns)
    
    def objective(weights):
        port_return, port_vol, sharpe = calculate_portfolio_metrics(returns, weights)
        if optimization_type == 'sharpe':
            return -sharpe  # Negative because we minimize
        elif optimization_type == 'min_vol':
            return port_vol
        elif optimization_type == 'max_return':
            return -port_return
    
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Equal weight starting point
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x

def monte_carlo_simulation(returns, weights, years, n_sims, initial_value):
    """Run Monte Carlo simulation"""
    n_days = years * 252
    
    # Calculate portfolio statistics
    mean_return = np.sum(returns.mean() * weights)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    
    # Generate random returns
    simulations = np.zeros((n_sims, n_days))
    
    for i in range(n_sims):
        daily_returns = np.random.normal(mean_return, portfolio_vol, n_days)
        cumulative_returns = np.cumprod(1 + daily_returns)
        simulations[i] = initial_value * cumulative_returns
    
    return simulations


@st.cache_data
def cached_monte_carlo_simulation(returns_hash, weights_hash, simulation_years, num_simulations, initial_investment):
    """Cached version of monte carlo simulation"""
    # Convert hashes back to actual data (you'll need to implement this)
    return monte_carlo_simulation(returns_hash, weights_hash, simulation_years, num_simulations, initial_investment)

def calculate_var_cvar(returns, confidence_level=0.05):
    """Calculate Value at Risk and Conditional VaR"""
    sorted_returns = np.sort(returns)
    index = int(confidence_level * len(sorted_returns))
    var = sorted_returns[index]
    cvar = sorted_returns[:index].mean()
    return var, cvar

def calculate_advanced_metrics(returns, weights, simulations):
    """Calculate advanced risk and performance metrics"""
    portfolio_returns = returns.dot(weights)
    
    # Basic metrics
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
    
    # Drawdown analysis
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # VaR and CVaR
    var_95, cvar_95 = calculate_var_cvar(portfolio_returns, 0.05)
    var_99, cvar_99 = calculate_var_cvar(portfolio_returns, 0.01)
    
    # Skewness and Kurtosis
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    
    # Simulation statistics
    final_values = simulations[:, -1]
    prob_profit = (final_values > initial_investment).mean()
    median_return = np.median(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)
    
    return {
        'Annual Return': f"{annual_return:.2%}",
        'Annual Volatility': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'VaR (95%)': f"{var_95:.2%}",
        'CVaR (95%)': f"{cvar_95:.2%}",
        'VaR (99%)': f"{var_99:.2%}",
        'CVaR (99%)': f"{cvar_99:.2%}",
        'Skewness': f"{skewness:.3f}",
        'Kurtosis': f"{kurtosis:.3f}",
        'Probability of Profit': f"{prob_profit:.1%}",
        'Median Final Value': f"${median_return:,.0f}",
        '5th Percentile': f"${percentile_5:,.0f}",
        '95th Percentile': f"${percentile_95:,.0f}"
    }

@st.cache_data
def cached_calculate_advanced_metrics(returns, weights, simulations):
    """Cached version of monte carlo simulation"""
    # Convert hashes back to actual data (you'll need to implement this)
    return calculate_advanced_metrics(returns, weights, simulations)


def display_statistics(returns, selected_weights, simulation_years, num_simulations, initial_investment):
    simulations = cached_monte_carlo_simulation(returns, selected_weights, 
                                                    simulation_years, num_simulations, 
                                                    initial_investment)
                
    # Calculate advanced metrics
    metrics = cached_calculate_advanced_metrics(returns, selected_weights, simulations)

    # Display simulation results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Monte Carlo Simulation Results")
        
        # Plot simulation paths
        fig_sim = go.Figure()
        
        # Plot sample paths
        time_axis = np.arange(0, simulation_years * 252) / 252
        for i in range(min(50, num_simulations)):  # Show max 50 paths
            fig_sim.add_trace(go.Scatter(
                x=time_axis, y=simulations[i],
                mode='lines', line=dict(width=0.5, color='lightblue'),
                showlegend=False, hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
            ))
        
        # Add percentiles
        percentiles = np.percentile(simulations, [5, 50, 95], axis=0)
        
        fig_sim.add_trace(go.Scatter(
            x=time_axis, y=percentiles[1],
            mode='lines', line=dict(width=3, color='red'),
            name='Median', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=time_axis, y=percentiles[2],
            mode='lines', line=dict(width=2, color='green'),
            name='95th Percentile', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=time_axis, y=percentiles[0],
            mode='lines', line=dict(width=2, color='orange'),
            name='5th Percentile', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
        ))
        
        fig_sim.update_layout(
            title=f"Portfolio Value Simulation ({num_simulations:,} paths)",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
    
    with col2:
        st.subheader("📊 Final Value Distribution")
        
        final_values = simulations[:, -1]
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name="Final Values",
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add vertical lines for key statistics
        fig_hist.add_vline(x=np.median(final_values), line_dash="dash", 
                            line_color="red", annotation_text="Median")
        fig_hist.add_vline(x=initial_investment, line_dash="dash", 
                            line_color="black", annotation_text="Initial")
        
        fig_hist.update_layout(
            title=f"Distribution of Final Portfolio Values (Year {simulation_years})",
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Advanced metrics table
    st.subheader("🔍 Advanced Risk & Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_list = list(metrics.items())
    for i, (metric, value) in enumerate(metrics_list):
        col = [col1, col2, col3, col4][i % 4]
        col.metric(metric, value)
    
    # Correlation matrix
    st.subheader("🔗 Asset Correlation Matrix")
    corr_matrix = returns.corr()
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig_corr.update_layout(
        title="Asset Correlation Matrix",
        width=600,
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
        

sim_ran = False   

# Main app logic
if st.button("Run Analysis", type="primary"):
    if len(stocks) < 2:
        st.error("Please enter at least 2 stock symbols")
    else:
        with st.spinner("Fetching data and optimizing portfolio..."):
            # Fetch data
            price_data = fetch_stock_data(stocks, lookback_period)
            
            if price_data is not None and not price_data.empty:
                # Calculate returns
                returns = price_data.pct_change().dropna()
                
                # Optimization options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Maximum Sharpe Ratio")
                    optimal_weights_sharpe = optimize_portfolio(returns, 'sharpe')
                    port_return_sharpe, port_vol_sharpe, sharpe_sharpe = calculate_portfolio_metrics(returns, optimal_weights_sharpe)
                    
                    # Display weights
                    weights_df_sharpe = pd.DataFrame({
                        'Stock': stocks,
                        'Weight': optimal_weights_sharpe
                    }).sort_values('Weight', ascending=False)
                    
                    fig_pie_sharpe = px.pie(weights_df_sharpe, values='Weight', names='Stock', 
                                          title="Optimal Weights (Max Sharpe)")
                    st.plotly_chart(fig_pie_sharpe, use_container_width=True)
                    
                    st.metric("Expected Return", f"{port_return_sharpe:.2%}")
                    st.metric("Volatility", f"{port_vol_sharpe:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe_sharpe:.3f}")
                
                with col2:
                    st.subheader("🛡️ Minimum Volatility")
                    optimal_weights_minvol = optimize_portfolio(returns, 'min_vol')
                    port_return_minvol, port_vol_minvol, sharpe_minvol = calculate_portfolio_metrics(returns, optimal_weights_minvol)
                    
                    weights_df_minvol = pd.DataFrame({
                        'Stock': stocks,
                        'Weight': optimal_weights_minvol
                    }).sort_values('Weight', ascending=False)
                    
                    fig_pie_minvol = px.pie(weights_df_minvol, values='Weight', names='Stock', 
                                          title="Optimal Weights (Min Vol)")
                    st.plotly_chart(fig_pie_minvol, use_container_width=True)
                    
                    st.metric("Expected Return", f"{port_return_minvol:.2%}")
                    st.metric("Volatility", f"{port_vol_minvol:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe_minvol:.3f}")
                
                with col3:
                    st.subheader("⚖️ Equal Weight")
                    equal_weights = np.array([1/len(stocks)] * len(stocks))
                    port_return_equal, port_vol_equal, sharpe_equal = calculate_portfolio_metrics(returns, equal_weights)
                    
                    weights_df_equal = pd.DataFrame({
                        'Stock': stocks,
                        'Weight': equal_weights
                    })
                    
                    fig_pie_equal = px.pie(weights_df_equal, values='Weight', names='Stock', 
                                         title="Equal Weights")
                    st.plotly_chart(fig_pie_equal, use_container_width=True)
                    
                    st.metric("Expected Return", f"{port_return_equal:.2%}")
                    st.metric("Volatility", f"{port_vol_equal:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe_equal:.3f}")
            else:
                st.error("Failed to fetch stock data. Please check the symbols and try again.")
                
        # Portfolio selection for simulation
        st.subheader("🎯 Select Portfolio for Simulation")
        portfolio_choice = st.selectbox(
            "Choose portfolio:",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weight"]
        )
        
        sim_ran = True
        
        if sim_ran:
            if portfolio_choice == "Maximum Sharpe Ratio":
                selected_weights = optimal_weights_sharpe
                st.session_state.optimal_weights_sharpe = optimal_weights_sharpe
                display_statistics(returns, st.session_state.optimal_weights_sharpe, simulation_years, num_simulations, initial_investment)
            elif portfolio_choice == "Minimum Volatility":
                selected_weights = optimal_weights_minvol
                st.session_state.optimal_weights_minvol = optimal_weights_minvol  
                display_statistics(returns, st.session_state.optimal_weights_minvol, simulation_years, num_simulations, initial_investment)
            else:
                selected_weights = equal_weights
                st.session_state.equal_weights = equal_weights
                display_statistics(returns, st.session_state.equal_weights, simulation_years, num_simulations, initial_investment)

        sim_ran = False
                # Run Monte Carlo simulation
                # with st.spinner("Running Monte Carlo simulation..."):
                #     simulations = monte_carlo_simulation(returns, selected_weights, 
                #                                        simulation_years, num_simulations, 
                #                                        initial_investment)
                    
                #     # Calculate advanced metrics
                #     metrics = calculate_advanced_metrics(returns, selected_weights, simulations)
                
            #     simulations = cached_monte_carlo_simulation(returns, selected_weights, 
            #                                         simulation_years, num_simulations, 
            #                                         initial_investment)
                
            #     # Calculate advanced metrics
            #     metrics = cached_calculate_advanced_metrics(returns, selected_weights, simulations)
            
            #     # Display simulation results
            #     col1, col2 = st.columns(2)
                
            #     with col1:
            #         st.subheader("📈 Monte Carlo Simulation Results")
                    
            #         # Plot simulation paths
            #         fig_sim = go.Figure()
                    
            #         # Plot sample paths
            #         time_axis = np.arange(0, simulation_years * 252) / 252
            #         for i in range(min(50, num_simulations)):  # Show max 50 paths
            #             fig_sim.add_trace(go.Scatter(
            #                 x=time_axis, y=simulations[i],
            #                 mode='lines', line=dict(width=0.5, color='lightblue'),
            #                 showlegend=False, hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
            #             ))
                    
            #         # Add percentiles
            #         percentiles = np.percentile(simulations, [5, 50, 95], axis=0)
                    
            #         fig_sim.add_trace(go.Scatter(
            #             x=time_axis, y=percentiles[1],
            #             mode='lines', line=dict(width=3, color='red'),
            #             name='Median', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
            #         ))
                    
            #         fig_sim.add_trace(go.Scatter(
            #             x=time_axis, y=percentiles[2],
            #             mode='lines', line=dict(width=2, color='green'),
            #             name='95th Percentile', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
            #         ))
                    
            #         fig_sim.add_trace(go.Scatter(
            #             x=time_axis, y=percentiles[0],
            #             mode='lines', line=dict(width=2, color='orange'),
            #             name='5th Percentile', hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'
            #         ))
                    
            #         fig_sim.update_layout(
            #             title=f"Portfolio Value Simulation ({num_simulations:,} paths)",
            #             xaxis_title="Years",
            #             yaxis_title="Portfolio Value ($)",
            #             hovermode='x unified'
            #         )
                    
            #         st.plotly_chart(fig_sim, use_container_width=True)
                
            #     with col2:
            #         st.subheader("📊 Final Value Distribution")
                    
            #         final_values = simulations[:, -1]
                    
            #         fig_hist = go.Figure()
            #         fig_hist.add_trace(go.Histogram(
            #             x=final_values,
            #             nbinsx=50,
            #             name="Final Values",
            #             marker_color='lightblue',
            #             opacity=0.7
            #         ))
                    
            #         # Add vertical lines for key statistics
            #         fig_hist.add_vline(x=np.median(final_values), line_dash="dash", 
            #                          line_color="red", annotation_text="Median")
            #         fig_hist.add_vline(x=initial_investment, line_dash="dash", 
            #                          line_color="black", annotation_text="Initial")
                    
            #         fig_hist.update_layout(
            #             title=f"Distribution of Final Portfolio Values (Year {simulation_years})",
            #             xaxis_title="Final Portfolio Value ($)",
            #             yaxis_title="Frequency",
            #             bargap=0.1
            #         )
                    
            #         st.plotly_chart(fig_hist, use_container_width=True)
                
            #     # Advanced metrics table
            #     st.subheader("🔍 Advanced Risk & Performance Metrics")
                
            #     col1, col2, col3, col4 = st.columns(4)
                
            #     metrics_list = list(metrics.items())
            #     for i, (metric, value) in enumerate(metrics_list):
            #         col = [col1, col2, col3, col4][i % 4]
            #         col.metric(metric, value)
                
            #     # Correlation matrix
            #     st.subheader("🔗 Asset Correlation Matrix")
            #     corr_matrix = returns.corr()
                
            #     fig_corr = go.Figure()
            #     fig_corr.add_trace(go.Heatmap(
            #         z=corr_matrix.values,
            #         x=corr_matrix.columns,
            #         y=corr_matrix.columns,
            #         colorscale='RdBu',
            #         zmid=0,
            #         text=np.round(corr_matrix.values, 2),
            #         texttemplate="%{text}",
            #         textfont={"size": 10},
            #         hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            #     ))
                
            #     fig_corr.update_layout(
            #         title="Asset Correlation Matrix",
            #         width=600,
            #         height=500
            #     )
                
            #     st.plotly_chart(fig_corr, use_container_width=True)
                
            

# Footer
st.markdown("---")
st.markdown("*Built with Python, Scipy, Yahoo Finance, and Streamlit, powered by quantitative finance principles and statistics*")
st.markdown("**Features:** Modern Portfolio Theory • Monte Carlo Simulation • Risk Metrics • VaR/CVaR Analysis")