//! This crate implements Cohen's class of time-frequency distributions in Rust. It allows computation of the time-frequency distribution of a signal with a chosen kernel function applied in the ambiguity domain.
//! ### Usage
//! ```
//! use bilinear_tf::*;
//! use rand::prelude::*;
//! use rayon::prelude::*;
//! 
//! let mut input = vec![0.0;48000];
//! input.par_iter_mut().for_each(|element| { *element = thread_rng().gen_range(-1.0..1.0); });
//! let result = bilinear_tf_distribution(&input, cone_shape, 0.001);
//! ```

use no_denormals::*;
use num_complex::Complex;
use rayon::prelude::*;

/// Cohen's Class Distribution Function.
pub fn bilinear_tf_distribution<F : Fn(f64, f64, f64) -> f64 + Sync>(input : &[f64], kernel: F, alpha : f64) -> Vec<Vec<f64>>
{
    no_denormals(||
    {
        let len = input.len();
        let ambiguity = ambiguity(input);
        
        (0..len).into_par_iter().map(|time|
        {
            (0..len).into_par_iter().map(|freq|
            {
                let mut sum = 0.0;
                for eta in 0..len
                {
                    for tau in 0..len
                    {
                        let exponent = 2.0 * std::f64::consts::PI * ((time * eta + freq * tau) as f64) / len as f64;
                        sum += (ambiguity[eta][tau] * kernel(eta as f64, tau as f64, alpha) * Complex::new(0.0, exponent).exp()).re;
                    }
                }
                sum / (len * len) as f64 // Normalize by dividing by the total number of samples
            }).collect()
        }).collect()
    })
}

/// Ambiguity Function.
#[inline]
fn ambiguity(input: &[f64]) -> Vec<Vec<Complex<f64>>>
{
    let len = input.len();
    (0..len).map(|eta|
    {
        (0..len).map(|tau|
        {
            let mut sum = Complex::new(0.0, 0.0);
            for t in 0..len
            {
                if t + tau / 2 < len && t >= tau / 2
                {
                    sum += Complex::new(input[t + tau / 2], 0.0) * Complex::new(input[t - tau / 2], 0.0).conj() * Complex::new(0.0, -2.0 * std::f64::consts::PI * (eta as f64) * (t as f64) / len as f64).exp();
                }
            }
            sum
        }).collect()
    }).collect()
}

/// Wigner Distribution Function
#[inline(always)]
pub fn wigner(_ : f64, _ : f64, _ : f64) -> f64 { 1.0 }

/// Choi-Williams Distribution Function
#[inline]
pub fn choi_williams(eta : f64, tau : f64, alpha : f64) -> f64 { (-alpha * (eta * eta) * (tau * tau) / 2.0).exp() }

/// Rihaczek Distribution Function
#[inline]
pub fn rihaczek(eta : f64, tau : f64, alpha : f64) -> f64 { (-alpha * eta * tau).exp() }

/// Zhao-Atlas-Marks Distribution Function
#[inline]
pub fn cone_shape(eta : f64, tau : f64, alpha : f64) -> f64
{
    let pi_eta_tau = std::f64::consts::PI * eta * tau;
    if pi_eta_tau == 0.0 { return 1.0 }
    (pi_eta_tau.sin() / pi_eta_tau) * (-2.0 * std::f64::consts::PI * alpha * tau.powi(2)).exp()
}

#[cfg(test)]
mod tests
{
    use super::*;
    use rand::prelude::*;

    #[test]
    fn main()
    {
        let mut input = vec![0.0;48000];
        input.par_iter_mut().for_each(|element| { *element = thread_rng().gen_range(-1.0..1.0); });
        let result = bilinear_tf_distribution(&input, cone_shape, 0.001);
    }
}