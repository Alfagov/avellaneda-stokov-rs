use std::f64;

pub struct Parameters {
    pub gamma: f64,     // Risk aversion
    pub sigma: f64,     // Volatility
    pub t_horizon: f64, // T (end time)
    pub k: f64,         // Liquidity parameter (used for strategy calc)
    pub a: f64,         // Base arrival rate (used for strategy calc)
}

/// Defines how the market intensity (arrival rate of fill) depends on the distance from mid-price.
pub trait IntensityModel: Send + Sync {
    fn calculate_intensity(&self, delta: f64) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct ExponentialIntensity {
    pub k: f64,
    pub a: f64,
}

impl IntensityModel for ExponentialIntensity {
    fn calculate_intensity(&self, delta: f64) -> f64 {
        self.a * (-self.k * delta.max(0.0)).exp()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PowerLawIntensity {
    pub a: f64,
    pub k: f64,
    pub beta: f64,
}

impl IntensityModel for PowerLawIntensity {
    fn calculate_intensity(&self, delta: f64) -> f64 {
        // A / (1 + k * delta) ^ beta
        self.a * (1.0 + self.k * delta.max(0.0)).powf(-self.beta)
    }
}

pub fn reservation_price(params: &Parameters, s: f64, q: i32, t: f64) -> f64 {
    s - q as f64 * params.gamma * params.sigma * params.sigma * (params.t_horizon - t)
}

pub fn optimal_spread(parameters: &Parameters, t: f64) -> f64 {
    let sigma_sq = parameters.sigma * parameters.sigma;
    parameters.gamma * sigma_sq * (parameters.t_horizon - t)
        + (2.0 / parameters.gamma) * (1.0 + (parameters.gamma / parameters.k)).ln()
}

pub fn quotes(r_price: f64, spread: f64) -> (f64, f64) {
    let spread_half = spread / 2.0;
    (r_price + spread_half, r_price - spread_half)
}
