use num_complex::Complex;
use rayon::prelude::*;

/// Cohen's Class Distribution Function.
pub fn bilinear_tf_distribution<F : Fn(f64, f64, f64) -> f64 + Sync>(input : &[f64], kernel: F, alpha : f64) -> Vec<Vec<f64>>
{
    let len = input.len();
    let ambiguity = ambiguity(input);

    let kernel_applied = std::sync::Arc::new(std::sync::Mutex::new(Some(vec![vec![Complex::new(0.0, 0.0); len]; len])));
    (0..len).into_par_iter().for_each(|eta|
    {
        (0..len).into_par_iter().for_each(|tau|
        {
            let mut kernel_applied = kernel_applied.try_lock().unwrap();
            let kernel_applied = kernel_applied.as_mut().unwrap();
            kernel_applied[eta][tau] = ambiguity[eta][tau] * Complex::new(kernel(eta as f64, tau as f64, alpha), 0.0);
        });
    });
    let distribution = std::sync::Arc::new(std::sync::Mutex::new(Some(vec![vec![0.0; len]; len])));
    (0..len).into_par_iter().for_each(|time|
    {
        (0..len).into_par_iter().for_each(|freq|
        {
            let sum = std::sync::Arc::new(std::sync::Mutex::new(Some(Complex::new(0.0, 0.0))));
            (0..len).into_par_iter().for_each(|eta|
            {
                (0..len).into_par_iter().for_each(|tau|
                {
                    let kernel_applied = kernel_applied.lock().unwrap().take().unwrap();
                    let mut sum = sum.try_lock().unwrap();
                    let sum = sum.as_mut().unwrap();
                    (*sum) += kernel_applied[eta][tau] * Complex::new(0.0, 2.0 * std::f64::consts::PI * ((time * eta + freq * tau) as f64) / len as f64).exp();
                });
            });
            let sum = sum.lock().unwrap().take().unwrap();
            let mut distribution = distribution.try_lock().unwrap();
            let distribution = distribution.as_mut().unwrap();
            distribution[time][freq] = sum.re / (len * len) as f64;
        });
    });
    let distribution = distribution.lock().unwrap().take().unwrap();
    distribution
}


/// Ambiguity Function.
fn ambiguity(input: &[f64]) -> Vec<Vec<Complex<f64>>>
{
    let len = input.len();
    let input = std::sync::Arc::new(input);
    let ambiguity = std::sync::Arc::new(std::sync::Mutex::new(Some(vec![vec![Complex::new(0.0, 0.0); len]; len])));
    
    (0..len).into_par_iter().for_each(|eta|
    {
        (0..len).into_par_iter().for_each(|tau|
        {
            let sum = std::sync::Arc::new(std::sync::Mutex::new(Some(Complex::new(0.0, 0.0))));
            (0..len).into_par_iter().for_each(|t|
            {
                let mut sum = sum.try_lock().unwrap();
                let sum = sum.as_mut().unwrap();
                *sum += Complex::new(input[ t + tau / 2], 0.0) * Complex::new(input[t - tau / 2], 0.0).conj() * Complex::new(0.0, -2.0 * std::f64::consts::PI * (eta as f64) * (t as f64) / len as f64).exp();
            });
            let sum = sum.lock().unwrap().take().unwrap();
            let mut ambiguity = ambiguity.try_lock().unwrap();
            let ambiguity = ambiguity.as_mut().unwrap();
            ambiguity[eta][tau] = sum;
        });
    });
    let ambiguity = ambiguity.lock().unwrap().take().unwrap();
    ambiguity
}

/// Wigner Distribution Function
#[inline(always)]
pub fn wigner(_ : f64, _ : f64, _ : f64) -> f64 { 1.0 }

/// Choi-Williams Distribution Function
#[inline]
pub fn choi_williams(eta : f64, tau : f64, alpha : f64) -> f64 { (-alpha * (eta * tau).powi(2)).exp() }

/// Rihaczek Distribution Function
#[inline]
pub fn rihaczek(eta : f64, tau : f64, alpha : f64) -> f64 { (-alpha * (eta * tau).powi(2)).exp() }

/// Zhao-Atlas-Marks Distribution Function
#[inline]
pub fn cone_shape(eta : f64, tau : f64, alpha : f64) -> f64
{
    (std::f64::consts::PI * eta * tau).sin() / (std::f64::consts::PI * eta * tau) * (-2.0 * std::f64::consts::PI * alpha * tau.powi(2)).exp()
}
