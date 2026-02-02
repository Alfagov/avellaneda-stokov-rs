# Avellaneda-Stoikov Market Making in Rust

This project implements the seminal **Avellaneda-Stoikov (2008)** high-frequency trading market-making strategy using Rust. It includes a Monte Carlo simulation engine and a parallelized analysis module to study the strategy's sensitivity to risk aversion, volatility, and market trends.

## Features

- **Core Strategy**: Implementation of reservation price and optimal spread logic.
- **Simulation Engine**: Efficient Monte Carlo simulation of mid-price evolution (Geometric Brownian Motion + Drift) and Poisson arrival of limit orders.
- **High-Performance Analysis**: Uses `rayon` for parallel execution of parameter sweeps.
- **Metrics**: Tracks PnL, inventory statistics (mean, max, terminal), Sharpe ratio, and more.
- **Advanced Physics**: Support for **Latency** simulation and alternative **Intensity Models** (e.g., Power Law).

## Usage

### Running the Analysis
The project includes a binary `run_analysis` that performs a parameter sweep over Risk Aversion ($\gamma$), Volatility ($\sigma$), and Drift ($\mu$).

```bash
cargo run --bin run_analysis
```

### Configuration
You can modify the parameter ranges in `src/bin/run_analysis.rs`:

```rust
// Example config in src/bin/run_analysis.rs
let sweep_config = SweepConfig {
    gammas: vec![0.01, 0.1, 1.0], // Risk aversion parameters
    sigmas: vec![0.1, 0.2],       // Volatility scenarios
    ks: vec![1.5],                // Liquidity parameters
    drifts: vec![0.0, 0.05],      // Drift/Trend scenarios
    // ...
};
```

## Key Concepts

- **Reservation Price ($r$)**: The price at which the agent is indifferent between buying and selling. It adjusts based on current inventory $q$ and risk aversion $\gamma$.
  $$ r(s, q, t) = s - q \gamma \sigma^2 (T - t) $$
  
- **Inventory Risk**: The model penalizes holding inventory as time approaches the horizon $T$, widening the spread on the side that increases inventory and tightening on the side that reduces it.

## Dependencies
- `rand` & `rand_distr`: For random number generation.
- `rayon`: For parallel processing.