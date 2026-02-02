use crate::model::{IntensityModel, Parameters};
use crate::sim::{SimConfig, run_trajectory};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub gammas: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub ks: Vec<f64>,
    pub drifts: Vec<f64>,
    pub sim_config: SimConfig,
    pub iterations_per_param: usize,
}

#[derive(Debug, Clone)]
pub struct SweepResult {
    pub gamma: f64,
    pub sigma: f64,
    pub k: f64,
    pub drift: f64,
    pub mean_pnl: f64,
    pub std_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub mean_abs_inventory: f64,
    pub max_inventory: f64,
    pub terminal_inventory_mean: f64,
    pub terminal_inventory_std: f64,
}

fn calculate_sharpe(pnls: &[f64]) -> f64 {
    let n = pnls.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = pnls.iter().sum::<f64>() / n;
    let variance = pnls.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev == 0.0 { 0.0 } else { mean / std_dev }
}

pub fn run_sweep(
    base_params: Parameters,
    sweep_config: &SweepConfig,
    intensity_model: &dyn IntensityModel,
) -> Vec<SweepResult> {
    // Generate all combinations of parameters
    let combinations: Vec<_> = itertools::iproduct!(
        &sweep_config.gammas,
        &sweep_config.sigmas,
        &sweep_config.ks,
        &sweep_config.drifts
    )
    .map(|(&g, &s, &k, &d)| (g, s, k, d))
    .collect();

    // Run simulations in parallel
    let results: Vec<SweepResult> = combinations
        .par_iter()
        .map(|&(gamma, sigma, k, drift)| {
            let params = Parameters {
                gamma,
                sigma,
                k,
                ..base_params
            };

            let mut current_sim_config = sweep_config.sim_config.clone();
            current_sim_config.drift = drift;

            struct RunStats {
                pnl: f64,
                mean_abs_q: f64,
                max_q: f64,
                final_q: f64,
            }

            // Run Monte Carlo for this parameter set
            let run_stats: Vec<RunStats> = (0..sweep_config.iterations_per_param)
                .map(|_| {
                    let res = run_trajectory(&params, &current_sim_config, intensity_model);

                    let final_q = res
                        .trajectory
                        .last()
                        .map(|s| s.inventory as f64)
                        .unwrap_or(0.0);
                    let max_q = res
                        .trajectory
                        .iter()
                        .map(|s| s.inventory.abs())
                        .max()
                        .unwrap_or(0) as f64;
                    let mean_abs_q = res
                        .trajectory
                        .iter()
                        .map(|s| s.inventory.abs() as f64)
                        .sum::<f64>()
                        / res.trajectory.len() as f64;

                    RunStats {
                        pnl: res.final_pnl,
                        mean_abs_q,
                        max_q,
                        final_q,
                    }
                })
                .collect();

            let n = run_stats.len() as f64;
            let pnls: Vec<f64> = run_stats.iter().map(|s| s.pnl).collect();
            let final_qs: Vec<f64> = run_stats.iter().map(|s| s.final_q).collect();

            let mean_pnl = pnls.iter().sum::<f64>() / n;
            let pnl_variance =
                pnls.iter().map(|&x| (x - mean_pnl).powi(2)).sum::<f64>() / (n - 1.0);
            let std_pnl = pnl_variance.sqrt();
            let sharpe = calculate_sharpe(&pnls);

            let mean_abs_inventory = run_stats.iter().map(|s| s.mean_abs_q).sum::<f64>() / n;
            let max_inventory = run_stats.iter().map(|s| s.max_q).sum::<f64>() / n;

            let terminal_inv_mean = final_qs.iter().sum::<f64>() / n;
            let terminal_inv_var = final_qs
                .iter()
                .map(|&x| (x - terminal_inv_mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            let terminal_inventory_std = terminal_inv_var.sqrt();

            SweepResult {
                gamma,
                sigma,
                k,
                drift,
                mean_pnl,
                std_pnl,
                sharpe_ratio: sharpe,
                max_drawdown: 0.0,
                mean_abs_inventory,
                max_inventory,
                terminal_inventory_mean: terminal_inv_mean,
                terminal_inventory_std: terminal_inventory_std,
            }
        })
        .collect();

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ExponentialIntensity, Parameters};
    use crate::sim::SimConfig;

    #[test]
    fn test_sweep_basic() {
        let base_params = Parameters {
            gamma: 0.1,
            sigma: 0.2,
            t_horizon: 1.0,
            k: 1.5,
            a: 140.0,
        };

        let sim_config = SimConfig {
            dt: 0.005,
            num_steps: 200,
            s_0: 100.0,
            drift: 0.0,
            latency_steps: 0,
        };

        let sweep_config = SweepConfig {
            gammas: vec![0.01, 0.1],
            sigmas: vec![0.2],
            ks: vec![1.5],
            drifts: vec![0.0],
            sim_config,
            iterations_per_param: 10,
        };

        let intensity_model = ExponentialIntensity {
            k: base_params.k,
            a: base_params.a,
        };

        let results = run_sweep(base_params, &sweep_config, &intensity_model);
        assert_eq!(results.len(), 2);
        assert!(results[0].mean_pnl != 0.0);
    }
}
