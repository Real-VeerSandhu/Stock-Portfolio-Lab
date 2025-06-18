# Portfolio Lab

A quantitative finance application for portfolio optimization and risk management using Modern Portfolio Theory and Monte Carlo simulations, with an optimized C computational engine.

## Overview

Portfolio Lab is a comprehensive financial analysis tool that enables users to optimize investment portfolios through statistical methods. The application combines traditional portfolio optimization techniques with Monte Carlo simulations to provide detailed risk assessment and performance forecasting. **Key differentiator**: Core computational functions are implemented in C for maximum performance, with a Python interface for ease of use.

## Architecture

### Hybrid Python-C Design
Portfolio Lab employs a multi-layer architecture optimized for both performance and usability

```
├── app.py                      # Main Streamlit application
├── portfolio_engine.c          # High-performance C computation engine
├── portfolio_interface.py      # Python-C bridge interface
├── portfolio_engine.so         # Compiled C library (generated)
├── data/
│   └── combined_stocks.csv     # Stock ticker dataset
├── requirements.txt            # Python dependencies
├── Makefile                    # Build automation (optional)
└── README.md                   # Documentation
```

### Performance Layer (C Engine)
The computational core (`portfolio_engine.c`) handles:
- **Monte Carlo Simulations**: Optimized random number generation with Box-Muller transform
- **Matrix Operations**: Efficient covariance matrix calculations
- **Risk Metrics**: VaR, CVaR, and drawdown computations
- **Statistical Analysis**: Moments calculation (skewness, kurtosis)
- **Memory Management**: Optimized allocation and cleanup

### Interface Layer (Python Bridge)
The `portfolio_interface.py` module provides:
- **C Library Loading**: Dynamic linking with compiled C engine
- **Data Marshaling**: Seamless numpy array ↔ C array conversion  
- **Type Safety**: Robust ctypes interface with error handling
- **Memory Safety**: Automatic cleanup of C-allocated resources

## Features

### Portfolio Optimization
- **Maximum Sharpe Ratio**: Optimize for the best risk-adjusted returns
- **Minimum Volatility**: Minimize portfolio risk while maintaining diversification
- **Equal Weight**: Benchmark against naive diversification strategy

### Advanced Analytics
- **High-Speed Monte Carlo**: C-optimized simulation engine with OpenMP parallelization
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown analysis
- **Performance Statistics**: Sharpe ratio, Skewness, Kurtosis
- **Correlation Analysis**: Asset correlation matrix visualization

### Interactive Visualizations
- Portfolio composition pie charts
- Monte Carlo simulation paths
- Final value distribution histograms
- Asset correlation heatmaps

## Models

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
GCC Compiler (for C engine compilation)
```

### System Dependencies
**Linux/Mac:**
```bash
sudo apt-get install gcc libomp-dev  # Ubuntu/Debian
brew install gcc libomp              # macOS
```

**Windows:**
```bash
# Install MinGW or Visual Studio Build Tools
# OpenMP support included with most modern compilers
```

### Required Python Dependencies
```python
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
scipy>=1.10.0
yfinance>=0.2.0
ctypes            
```

### Compilation & Setup
```bash
# Clone repository
git clone https://github.com/real-veersandhu/portfolio-lab.git
cd portfolio-lab

# Install Python dependencies
pip install -r requirements.txt

# Compile C engine with OpenMP support
gcc -O3 -march=native -fopenmp -fPIC -shared -o portfolio_engine.so portfolio_engine.c

# Alternative: Compile without OpenMP
gcc -O3 -march=native -fPIC -shared -o portfolio_engine.so portfolio_engine.c

# Run application
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

### C Engine Implementation

#### High-Performance Computing Features
- **Custom RNG**: Fast linear congruential generator with Box-Muller transform
- **SIMD Optimization**: Compiler vectorization with `-march=native`
- **OpenMP Parallelization**: Multi-threaded Monte Carlo simulations
- **Memory Efficiency**: Contiguous memory allocation and cache-friendly algorithms

#### Core C Functions
```c
SimulationResults* monte_carlo_simulation(Portfolio *portfolio, 
                                        int simulation_years, 
                                        int n_simulations, 
                                        double initial_value);

void calculate_covariance(double *retur