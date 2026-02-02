use crate::model::{IntensityModel, Parameters, optimal_spread, quotes, reservation_price};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub dt: f64,
    pub num_steps: usize,
    pub s_0: f64,
    pub drift: f64,
    pub latency_steps: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    pub time: f64,
    pub mid_price: f64,
    pub inventory: i32,
    pub cash: f64,
    pub wealth: f64,
    pub bid_price: f64,
    pub ask_price: f64,
}

pub struct SimResult {
    pub trajectory: Vec<StepRecord>,
    pub final_pnl: f64,
}

pub fn run_trajectory(
    agent_params: &Parameters,
    config: &SimConfig,
    intensity_model: &dyn IntensityModel,
) -> SimResult {
    let mut rng = rand::rng();

    let mut t = 0.0;
    let mut s = config.s_0;
    let mut q = 0;
    let mut w = 0.0;

    let mut trajectory = Vec::with_capacity(config.num_steps);

    // Low-level latency queue: stores (ask, bid) quotes sent by agent
    // These quotes will be available to the 'Market' after Latency steps.
    let mut quote_queue: VecDeque<(f64, f64)> = VecDeque::new();

    for _ in 0..config.num_steps {
        let r = reservation_price(agent_params, s, q, t);
        let spread = optimal_spread(agent_params, t);
        let (ask, bid) = quotes(r, spread);

        quote_queue.push_back((ask, bid));

        // 2. Market State Determination (Latency)
        // If we have more quotes than latency, the one at the front (oldest) is what reached the market L steps ago
        // and is now active.
        let (effective_ask, effective_bid) = if config.latency_steps == 0 {
            (ask, bid)
        } else {
            // Buffer needs to hold `latency_steps + 1` items to peek at T-L.
            // Example: Latency=1. T=0 push Q0. Len=1. Effective=Q0 (optimistic start) or None?
            // Let's assume optimistic start for simplicity: current quotes apply until buffer fills.
            if quote_queue.len() > config.latency_steps {
                quote_queue.pop_front().unwrap()
            } else {
                *quote_queue.back().unwrap()
            }
        };

        let wealth = w + (q as f64 * s);

        trajectory.push(StepRecord {
            time: t,
            mid_price: s,
            inventory: q,
            cash: w,
            wealth,
            bid_price: effective_bid,
            ask_price: effective_ask,
        });

        // 3. Market Evolution
        let norm_sample: f64 = StandardNormal.sample(&mut rng);
        let return_innovation = agent_params.sigma * config.dt.sqrt() * norm_sample;
        let drift_component = config.drift * config.dt;
        s *= 1.0 + drift_component + return_innovation;

        // 4. Order Fill Logic (using Effective Quotes vs New Price)
        let delta_bid = s - effective_bid;
        let delta_ask = effective_ask - s;

        let lambda_bid = intensity_model.calculate_intensity(delta_bid);
        let lambda_ask = intensity_model.calculate_intensity(delta_ask);

        let prob_bid_fill = lambda_bid * config.dt;
        let prob_ask_fill = lambda_ask * config.dt;

        let bid_hit = rng.random::<f64>() < prob_bid_fill;
        let ask_hit = rng.random::<f64>() < prob_ask_fill;

        if bid_hit {
            q += 1;
            w -= effective_bid;
        }

        if ask_hit {
            q -= 1;
            w += effective_ask;
        }

        t += config.dt;
    }

    let final_wealth = w + (q as f64 * s);

    SimResult {
        trajectory,
        final_pnl: final_wealth,
    }
}
