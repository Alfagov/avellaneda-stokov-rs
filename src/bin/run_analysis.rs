use avellaneda_stoikov_rs::analysis::{SweepConfig, run_sweep};
use avellaneda_stoikov_rs::model::{ExponentialIntensity, Parameters};
use avellaneda_stoikov_rs::sim::SimConfig;
use std::time::Instant;

fn main() {
    let base_params = Parameters {
        gamma: 0.1,
        sigma: 0.2,
        t_horizon: 1.0,
        k: 1.5,
        a: 140.0,
    };

    let sim_config = SimConfig {
        dt: 0.005,
        num_steps: 600,
        s_0: 100.0,
        drift: 0.0, // Base drift
        latency_steps: 0,
    };

    // Define the sweep configuration
    let sweep_config = SweepConfig {
        // Vary Risk Aversion
        gammas: vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        // Vary Volatility
        sigmas: vec![0.1, 0.2, 0.4],
        // Constant K
        ks: vec![1.5],
        // Vary Drift
        drifts: vec![0.0, 0.05, -0.05],
        sim_config,
        iterations_per_param: 1000,
    };

    println!(
        "Starting parameter sweep with {} iterations per config...",
        sweep_config.iterations_per_param
    );
    let start_time = Instant::now();

    let intensity_model = ExponentialIntensity {
        k: base_params.k,
        a: base_params.a,
    };

    let results = run_sweep(base_params, &sweep_config, &intensity_model);

    let duration = start_time.elapsed();
    println!("Sweep completed in {:.2}s", duration.as_secs_f64());
    println!(
        "{:<8} {:<8} {:<6} {:<8} | {:<12} {:<12} {:<10} | {:<10} {:<10}",
        "Gamma", "Sigma", "K", "Drift", "Mean PnL", "Std PnL", "Sharpe", "Mean |Q|", "Final Q"
    );
    println!("{}", "-".repeat(100));

    for res in results {
        println!(
            "{:<8.2} {:<8.2} {:<6.2} {:<8.2} | {:<12.4} {:<12.4} {:<10.4} | {:<10.2} {:<10.2}",
            res.gamma,
            res.sigma,
            res.k,
            res.drift,
            res.mean_pnl,
            res.std_pnl,
            res.sharpe_ratio,
            res.mean_abs_inventory,
            res.terminal_inventory_mean
        );
    }
}
