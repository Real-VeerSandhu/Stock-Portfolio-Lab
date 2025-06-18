// portfolio_engine.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Structure to hold portfolio data
typedef struct {
    double *returns;
    double *weights;
    int n_assets;
    int n_periods;
    double risk_free_rate;
} Portfolio;

// Structure for simulation results
typedef struct {
    double *simulations;
    int n_sims;
    int n_periods;
    double *final_values;
    double *percentiles;
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double max_drawdown;
    double sharpe_ratio;
    double annual_return;
    double annual_vol;
    double skewness;
    double kurtosis;
} SimulationResults;

// Fast random number generator 
static unsigned long next = 1;

int fast_rand(void) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % 32768;
}

void fast_srand(unsigned int seed) {
    next = seed;
}

// Box-Muller transform for normal random numbers
double normal_random() {
    static int has_spare = 0;
    static double spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    has_spare = 1;
    static double u, v, mag;
    
    do {
        u = ((double)fast_rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)fast_rand() / RAND_MAX) * 2.0 - 1.0;
        mag = u * u + v * v;
    } while (mag >= 1.0 || mag == 0.0);
    
    mag = sqrt(-2.0 * log(mag) / mag);
    spare = v * mag;
    return u * mag;
}

// Matrix operations
void matrix_multiply(double *A, double *B, double *C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < m; k++) {
                C[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

// Calculate covariance matrix
void calculate_covariance(double *returns, double *cov_matrix, int n_assets, int n_periods) {
    // Calculate means
    double *means = (double*)calloc(n_assets, sizeof(double));
    
    for (int i = 0; i < n_assets; i++) {
        for (int t = 0; t < n_periods; t++) {
            means[i] += returns[t * n_assets + i];
        }
        means[i] /= n_periods;
    }
    
    // Calculate covariance
    for (int i = 0; i < n_assets; i++) {
        for (int j = 0; j < n_assets; j++) {
            double covar = 0.0;
            for (int t = 0; t < n_periods; t++) {
                covar += (returns[t * n_assets + i] - means[i]) * 
                        (returns[t * n_assets + j] - means[j]);
            }
            cov_matrix[i * n_assets + j] = covar / (n_periods - 1);
        }
    }
    
    free(means);
}

// Portfolio metrics calculation
void calculate_portfolio_metrics(Portfolio *portfolio, double *annual_return, 
                                double *annual_vol, double *sharpe_ratio) {
    int n = portfolio->n_assets;
    
    // Calculate portfolio return
    double *asset_returns = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < portfolio->n_periods; t++) {
            asset_returns[i] += portfolio->returns[t * n + i];
        }
        asset_returns[i] /= portfolio->n_periods;
    }
    
    double port_return = 0.0;
    for (int i = 0; i < n; i++) {
        port_return += portfolio->weights[i] * asset_returns[i];
    }
    *annual_return = port_return * 252.0;
    
    // Calculate covariance matrix
    double *cov_matrix = (double*)malloc(n * n * sizeof(double));
    calculate_covariance(portfolio->returns, cov_matrix, n, portfolio->n_periods);
    
    // Calculate portfolio variance
    double port_var = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            port_var += portfolio->weights[i] * portfolio->weights[j] * 
                       cov_matrix[i * n + j];
        }
    }
    *annual_vol = sqrt(port_var * 252.0);
    *sharpe_ratio = (*annual_return - portfolio->risk_free_rate) / (*annual_vol);
    
    free(asset_returns);
    free(cov_matrix);
}

// Fast Monte Carlo simulation
SimulationResults* monte_carlo_simulation(Portfolio *portfolio, int simulation_years, 
                                        int n_simulations, double initial_value) {
    int n_days = simulation_years * 252;
    
    SimulationResults *results = (SimulationResults*)malloc(sizeof(SimulationResults));
    results->simulations = (double*)malloc(n_simulations * n_days * sizeof(double));
    results->final_values = (double*)malloc(n_simulations * sizeof(double));
    results->percentiles = (double*)malloc(3 * sizeof(double)); // 5th, 50th, 95th
    results->n_sims = n_simulations;
    results->n_periods = n_days;
    
    // Calculate portfolio statistics
    double annual_return, annual_vol, sharpe_ratio;
    calculate_portfolio_metrics(portfolio, &annual_return, &annual_vol, &sharpe_ratio);
    
    double daily_return = annual_return / 252.0;
    double daily_vol = annual_vol / sqrt(252.0);
    
    // Seed random number generator
    fast_srand((unsigned int)time(NULL));
    
    // Run simulations
    #pragma omp parallel for // OpenMP parallelization
    for (int sim = 0; sim < n_simulations; sim++) {
        double current_value = initial_value;
        
        for (int day = 0; day < n_days; day++) {
            double random_return = normal_random() * daily_vol + daily_return;
            current_value *= (1.0 + random_return);
            results->simulations[sim * n_days + day] = current_value;
        }
        
        results->final_values[sim] = current_value;
    }
    
    // Calculate statistics
    calculate_simulation_statistics(results, portfolio, initial_value);
    
    return results;
}

// Calculate VaR and CVaR
void calculate_var_cvar(double *returns, int n_returns, double confidence, 
                       double *var, double *cvar) {
    // Sort returns (simple bubble sort - use qsort for better performance)
    double *sorted_returns = (double*)malloc(n_returns * sizeof(double));
    memcpy(sorted_returns, returns, n_returns * sizeof(double));
    
    for (int i = 0; i < n_returns - 1; i++) {
        for (int j = 0; j < n_returns - i - 1; j++) {
            if (sorted_returns[j] > sorted_returns[j + 1]) {
                double temp = sorted_returns[j];
                sorted_returns[j] = sorted_returns[j + 1];
                sorted_returns[j + 1] = temp;
            }
        }
    }
    
    int index = (int)(confidence * n_returns);
    *var = -sorted_returns[index];
    
    double sum = 0.0;
    for (int i = 0; i <= index; i++) {
        sum += sorted_returns[i];
    }
    *cvar = -sum / (index + 1);
    
    free(sorted_returns);
}

// Calculate drawdown
double calculate_max_drawdown(double *values, int n_values) {
    double max_drawdown = 0.0;
    double peak = values[0];
    
    for (int i = 1; i < n_values; i++) {
        if (values[i] > peak) {
            peak = values[i];
        }
        
        double drawdown = (peak - values[i]) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    return max_drawdown;
}

// Calculate skewness and kurtosis
void calculate_moments(double *values, int n_values, double *skewness, double *kurtosis) {
    double mean = 0.0;
    for (int i = 0; i < n_values; i++) {
        mean += values[i];
    }
    mean /= n_values;
    
    double variance = 0.0;
    double third_moment = 0.0;
    double fourth_moment = 0.0;
    
    for (int i = 0; i < n_values; i++) {
        double diff = values[i] - mean;
        variance += diff * diff;
        third_moment += diff * diff * diff;
        fourth_moment += diff * diff * diff * diff;
    }
    
    variance /= (n_values - 1);
    third_moment /= n_values;
    fourth_moment /= n_values;
    
    double std_dev = sqrt(variance);
    *skewness = third_moment / (std_dev * std_dev * std_dev);
    *kurtosis = (fourth_moment / (variance * variance)) - 3.0;
}

// Calculate all simulation statistics
void calculate_simulation_statistics(SimulationResults *results, Portfolio *portfolio, 
                                   double initial_value) {
    // Calculate percentiles
    double *sorted_finals = (double*)malloc(results->n_sims * sizeof(double));
    memcpy(sorted_finals, results->final_values, results->n_sims * sizeof(double));
    
    // Sort final values
    for (int i = 0; i < results->n_sims - 1; i++) {
        for (int j = 0; j < results->n_sims - i - 1; j++) {
            if (sorted_finals[j] > sorted_finals[j + 1]) {
                double temp = sorted_finals[j];
                sorted_finals[j] = sorted_finals[j + 1];
                sorted_finals[j + 1] = temp;
            }
        }
    }
    
    results->percentiles[0] = sorted_finals[(int)(0.05 * results->n_sims)]; // 5th
    results->percentiles[1] = sorted_finals[(int)(0.5 * results->n_sims)];  // 50th
    results->percentiles[2] = sorted_finals[(int)(0.95 * results->n_sims)]; // 95th
    
    // Calculate returns for VaR/CVaR
    double *returns = (double*)malloc(results->n_sims * sizeof(double));
    for (int i = 0; i < results->n_sims; i++) {
        returns[i] = (results->final_values[i] - initial_value) / initial_value;
    }
    
    calculate_var_cvar(returns, results->n_sims, 0.05, &results->var_95, &results->cvar_95);
    calculate_var_cvar(returns, results->n_sims, 0.01, &results->var_99, &results->cvar_99);
    
    // Calculate portfolio metrics
    calculate_portfolio_metrics(portfolio, &results->annual_return, 
                               &results->annual_vol, &results->sharpe_ratio);
    
    // Calculate max drawdown from median path
    int median_sim = results->n_sims / 2;
    results->max_drawdown = calculate_max_drawdown(
        &results->simulations[median_sim * results->n_periods], results->n_periods);
    
    // Calculate moments
    calculate_moments(returns, results->n_sims, &results->skewness, &results->kurtosis);
    
    free(sorted_finals);
    free(returns);
}

// Cleanup function
void free_simulation_results(SimulationResults *results) {
    if (results) {
        free(results->simulations);
        free(results->final_values);
        free(results->percentiles);
        free(results);
    }
}

// Python interface functions
extern "C" {
    SimulationResults* run_portfolio_simulation(double *returns_data, double *weights, 
                                              int n_assets, int n_periods, 
                                              double risk_free_rate, int simulation_years, 
                                              int n_simulations, double initial_value) {
        Portfolio portfolio;
        portfolio.returns = returns_data;
        portfolio.weights = weights;
        portfolio.n_assets = n_assets;
        portfolio.n_periods = n_periods;
        portfolio.risk_free_rate = risk_free_rate;
        
        return monte_carlo_simulation(&portfolio, simulation_years, n_simulations, initial_value);
    }
    
    void get_simulation_results(SimulationResults *results, double *output_simulations,
                               double *output_finals, double *output_percentiles,
                               double *output_metrics) {
        // Copy simulation data
        memcpy(output_simulations, results->simulations, results->n_sims * results->n_periods * sizeof(double));
        memcpy(output_finals, results->final_values, results->n_sims * sizeof(double));
        memcpy(output_percentiles, results->percentiles, 3 * sizeof(double));
        
        // Pack metrics into output array
        output_metrics[0] = results->annual_return;
        output_metrics[1] = results->annual_vol;
        output_metrics[2] = results->sharpe_ratio;
        output_metrics[3] = results->max_drawdown;
        output_metrics[4] = results->var_95;
        output_metrics[5] = results->cvar_95;
        output_metrics[6] = results->var_99;
        output_metrics[7] = results->cvar_99;
        output_metrics[8] = results->skewness;
        output_metrics[9] = results->kurtosis;
    }
}