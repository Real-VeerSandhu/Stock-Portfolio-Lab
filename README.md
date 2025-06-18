# Portfolio Lab 📈

A sophisticated quantitative finance application for portfolio optimization and risk management using Modern Portfolio Theory and Monte Carlo simulations.

## Overview

Portfolio Lab is a comprehensive financial analysis tool that enables users to optimize investment portfolios through advanced statistical methods. The application combines traditional portfolio optimization techniques with Monte Carlo simulations to provide detailed risk assessment and performance forecasting.

## Features

### 🎯 Portfolio Optimization
- **Maximum Sharpe Ratio**: Optimize for the best risk-adjusted returns
- **Minimum Volatility**: Minimize portfolio risk while maintaining diversification
- **Equal Weight**: Benchmark against naive diversification strategy

### 📊 Advanced Analytics
- **Monte Carlo Simulation**: Generate thousands of potential portfolio paths
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown analysis
- **Performance Statistics**: Sharpe ratio, Skewness, Kurtosis
- **Correlation Analysis**: Asset correlation matrix visualization

### 📈 Interactive Visualizations
- Portfolio composition pie charts
- Monte Carlo simulation paths
- Final value distribution histograms
- Asset correlation heatmaps

## Technical Implementation

### Mathematical Foundation

#### Modern Portfolio Theory (MPT)
The application implements Harry Markowitz's Modern Portfolio Theory to find optimal asset allocations:

```
Portfolio Return: μₚ = Σ(wᵢ × μᵢ)
Portfolio Variance: σₚ² = Σ(wᵢ² × σᵢ²) + ΣΣ(wᵢ × wⱼ × σᵢⱼ)
Sharpe Ratio: Sₚ = (μₚ - rₓ) / σₚ
```

Where:
- `wᵢ` = weight of asset i
- `μᵢ` = expected return of asset i
- `σᵢ` = standard deviation of asset i
- `σᵢⱼ` = covariance between assets i and j
- `rₓ` = risk-free rate

#### Optimization Constraints
```
Subject to:
- Σwᵢ = 1 (weights sum to 100%)
- wᵢ ≥ 0 (no short selling)
- 0 ≤ wᵢ ≤ 1 (individual weight bounds)
```

#### Monte Carlo Simulation
Portfolio value simulation using geometric Brownian motion:

```
Sₜ = S₀ × exp((μ - σ²/2) × t + σ × √t × Z)
```

Where:
- `Sₜ` = portfolio value at time t
- `S₀` = initial portfolio value
- `μ` = expected daily return
- `σ` = daily volatility
- `Z` = random normal variable N(0,1)

### Risk Metrics

#### Value at Risk (VaR)
Quantile-based risk measure:
```
VaR₍α₎ = -Quantile(returns, α)
```

#### Conditional Value at Risk (CVaR)
Expected loss beyond VaR threshold:
```
CVaR₍α₎ = E[loss | loss > VaR₍α₎]
```

#### Maximum Drawdown
Peak-to-trough decline:
```
DD(t) = (Peak_value - Current_value) / Peak_value
Max_DD = min(DD(t)) for all t
```

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Dependencies
```python
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
scipy>=1.10.0
yfinance>=0.2.0
```

### Data Requirements
Create a `data/` directory with `combined_stocks.csv` containing:
```csv
TICKER,NAME
AAPL,Apple Inc. Common Stock
MSFT,Microsoft Corporation Common Stock
GOOGL,Alphabet Inc. Class A Common Stock
...
```

### Running the Application
```bash
streamlit run app.py
```

## Usage Guide

### 1. Portfolio Configuration
- Select 2+ stocks from the dropdown (supports search)
- Choose historical data period (1-5 years)
- Set risk-free rate (default: 4.5%)

### 2. Simulation Parameters
- **Simulation Years**: Forecast horizon (1-20 years)
- **Number of Simulations**: Monte Carlo iterations (10-10,000)
- **Initial Investment**: Starting portfolio value

### 3. Analysis Execution
Click "Run Analysis" to:
- Fetch historical price data via Yahoo Finance API
- Calculate daily returns and covariance matrix
- Solve optimization problems using Sequential Least Squares Programming (SLSQP)
- Generate portfolio composition visualizations

### 4. Portfolio Selection
Choose from three optimized portfolios:
- **Max Sharpe**: Highest risk-adjusted returns
- **Min Volatility**: Lowest portfolio risk
- **Equal Weight**: Benchmark strategy

### 5. Results Interpretation

#### Performance Metrics
- **Annual Return**: Annualized expected return (252 trading days)
- **Annual Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Worst peak-to-trough decline

#### Risk Metrics
- **VaR (95%/99%)**: Potential loss at confidence levels
- **CVaR**: Expected loss beyond VaR threshold
- **Skewness**: Return distribution asymmetry
- **Kurtosis**: Tail risk measurement

#### Simulation Results
- **Probability of Profit**: Chance of positive returns
- **Percentile Analysis**: 5th, 50th, 95th percentile outcomes
- **Path Visualization**: Sample simulation trajectories

## Technical Architecture

### Caching Strategy
The application implements Streamlit's `@st.cache_data` decorator for:
- Historical data fetching
- Monte Carlo simulations
- Advanced metrics calculations

This reduces computation time and improves user experience.

### Session State Management
Key variables stored in `st.session_state`:
- Optimized portfolio weights
- Historical returns data
- Simulation parameters
- Analysis completion status

### Optimization Algorithm
Uses SciPy's `minimize()` with SLSQP method:
- **Objective Functions**: Negative Sharpe ratio or portfolio volatility
- **Constraints**: Weight sum equals 1
- **Bounds**: Individual weights between 0 and 1
- **Initial Guess**: Equal-weight allocation

## Statistical Validation

### Assumptions
1. **Log-normal returns**: Asset returns follow normal distribution
2. **Constant parameters**: Mean and covariance remain stable
3. **Efficient markets**: No transaction costs or taxes
4. **Liquidity**: Instant buying/selling at market prices

### Limitations
- Historical performance doesn't guarantee future results
- Assumes returns are independently and identically distributed
- Black swan events not captured in historical data
- Model risk from parameter estimation uncertainty

## Contributing

### Development Setup
```bash
git clone https://github.com/username/portfolio-lab.git
cd portfolio-lab
pip install -r requirements.txt
```

### Code Structure
```
portfolio-lab/
├── app.py                  # Main application
├── data/
│   └── combined_stocks.csv # Stock ticker database
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

### Testing
Run unit tests for optimization functions:
```bash
python -m pytest tests/
```

## Performance Considerations

### Computational Complexity
- **Portfolio Optimization**: O(n³) for n assets (covariance matrix inversion)
- **Monte Carlo Simulation**: O(m × t) for m simulations and t time steps
- **Risk Calculations**: O(m log m) for sorting simulation results

### Memory Usage
- Simulation matrix: ~8MB for 10,000 simulations × 1,260 days (5 years)
- Caching reduces redundant calculations
- Efficient NumPy operations for matrix computations

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **SciPy**: Numerical optimization
- **Yahoo Finance**: Market data API
- **NumPy/Pandas**: Numerical computing

## Author

**Veer Sandhu** - 2025

---

*Portfolio Lab leverages fundamental statistical principles and modern computational methods to democratize quantitative portfolio management.*

## Disclaimer

This application is for educational and research purposes only. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.