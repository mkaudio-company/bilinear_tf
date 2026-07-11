//! This crate implements Cohen's class of time-frequency distributions in Rust. It allows computation of the time-frequency distribution of a signal with a chosen kernel function applied in the ambiguity domain.
//! ### Usage
//! ```
//! use bilinear_tf::*;
//! use rand::prelude::*;
//! use rayon::prelude::*;
//!
//! let mut input = vec![0.0;128];
//! input.par_iter_mut().for_each(|element| { *element = thread_rng().gen_range(-1.0..1.0); });
//! let result = bilinear_tf_distribution(&input, cone_shape, 0.001);
//! ```

use no_denormals::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::TAU;
use wide::f64x4;

/// Cohen's Class Distribution Function.
///
/// `exp(i*2*pi*(time*eta + freq*tau)/len)` is separable into
/// `exp(i*2*pi*time*eta/len) * exp(i*2*pi*freq*tau/len)`, so the distribution is computed as
/// two `O(len^3)` DFT passes (over tau, then over eta) instead of one naive `O(len^4)` double
/// sum, with the inner dot products SIMD-accelerated via [`complex_phase_dot`].
pub fn bilinear_tf_distribution<F: Fn(f64, f64, f64) -> f64 + Sync>(
    input: &[f64],
    kernel: F,
    alpha: f64,
) -> Vec<Vec<f64>> {
    no_denormals(|| {
        let len = input.len();
        let ambiguity = ambiguity(input);

        let weighted: Vec<Vec<Complex<f64>>> = (0..len)
            .into_par_iter()
            .map(|eta| {
                (0..len)
                    .map(|tau| ambiguity[eta][tau] * kernel(eta as f64, tau as f64, alpha))
                    .collect()
            })
            .collect();

        // Pass 1: DFT over tau for each eta row -> intermediate[eta][freq].
        let intermediate: Vec<Vec<Complex<f64>>> = weighted
            .par_iter()
            .map(|row| {
                (0..len)
                    .map(|freq| complex_phase_dot(row, TAU * freq as f64 / len as f64))
                    .collect()
            })
            .collect();

        // Transpose to [freq][eta] so pass 2 can DFT over contiguous eta values.
        let mut columns = vec![vec![Complex::new(0.0, 0.0); len]; len];
        for (eta, row) in intermediate.iter().enumerate() {
            for (freq, value) in row.iter().enumerate() {
                columns[freq][eta] = *value;
            }
        }

        // Pass 2: DFT over eta for each freq column -> by_freq[freq][time].
        let by_freq: Vec<Vec<f64>> = columns
            .par_iter()
            .map(|column| {
                (0..len)
                    .map(|time| {
                        complex_phase_dot(column, TAU * time as f64 / len as f64).re
                            / (len * len) as f64
                    })
                    .collect()
            })
            .collect();

        // Transpose back to [time][freq].
        let mut result = vec![vec![0.0; len]; len];
        for (freq, row) in by_freq.iter().enumerate() {
            for (time, value) in row.iter().enumerate() {
                result[time][freq] = *value;
            }
        }
        result
    })
}

/// Ambiguity Function.
#[inline]
fn ambiguity(input: &[f64]) -> Vec<Vec<Complex<f64>>> {
    let len = input.len();

    // For a fixed tau, the lag product input[t + tau/2] * input[t - tau/2] does not depend on
    // eta, so it is computed once per tau and reused for the DFT over eta below, instead of
    // being recomputed len times as in the naive triple-nested-loop formulation.
    let by_tau: Vec<Vec<Complex<f64>>> = (0..len)
        .into_par_iter()
        .map(|tau| {
            let half = tau / 2;
            if half >= len - half {
                return vec![Complex::new(0.0, 0.0); len];
            }

            let products: Vec<f64> = (half..len - half)
                .map(|t| input[t + half] * input[t - half])
                .collect();

            (0..len)
                .map(|eta| {
                    let angle_step = -TAU * eta as f64 / len as f64;
                    let rotation = Complex::new(0.0, angle_step * half as f64).exp();
                    rotation * phase_dot_real(&products, angle_step)
                })
                .collect()
        })
        .collect();

    // Transpose from [tau][eta] to [eta][tau].
    let mut result = vec![vec![Complex::new(0.0, 0.0); len]; len];
    for (tau, row) in by_tau.iter().enumerate() {
        for (eta, value) in row.iter().enumerate() {
            result[eta][tau] = *value;
        }
    }
    result
}

/// Computes `sum_k values[k] * exp(i * angle_step * k)`, i.e. a single-frequency DFT
/// coefficient, using 4-lane SIMD for the trigonometry and multiply-accumulate.
#[inline]
fn phase_dot_real(values: &[f64], angle_step: f64) -> Complex<f64> {
    let mut acc_re = f64x4::ZERO;
    let mut acc_im = f64x4::ZERO;
    let mut angles = f64x4::new([0.0, angle_step, 2.0 * angle_step, 3.0 * angle_step]);
    let step = f64x4::splat(4.0 * angle_step);

    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let (sin_a, cos_a) = angles.sin_cos();
        acc_re += v * cos_a;
        acc_im += v * sin_a;
        angles += step;
    }

    let mut re = acc_re.reduce_add();
    let mut im = acc_im.reduce_add();

    let processed = values.len() - remainder.len();
    for (k, &value) in remainder.iter().enumerate() {
        let angle = angle_step * (processed + k) as f64;
        re += value * angle.cos();
        im += value * angle.sin();
    }

    Complex::new(re, im)
}

/// Computes `sum_k values[k] * exp(i * angle_step * k)` for complex `values`, using 4-lane
/// SIMD for the trigonometry and multiply-accumulate.
#[inline]
fn complex_phase_dot(values: &[Complex<f64>], angle_step: f64) -> Complex<f64> {
    let mut acc_re = f64x4::ZERO;
    let mut acc_im = f64x4::ZERO;
    let mut angles = f64x4::new([0.0, angle_step, 2.0 * angle_step, 3.0 * angle_step]);
    let step = f64x4::splat(4.0 * angle_step);

    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let vre = f64x4::new([chunk[0].re, chunk[1].re, chunk[2].re, chunk[3].re]);
        let vim = f64x4::new([chunk[0].im, chunk[1].im, chunk[2].im, chunk[3].im]);
        let (sin_a, cos_a) = angles.sin_cos();
        acc_re += vre * cos_a - vim * sin_a;
        acc_im += vre * sin_a + vim * cos_a;
        angles += step;
    }

    let mut re = acc_re.reduce_add();
    let mut im = acc_im.reduce_add();

    let processed = values.len() - remainder.len();
    for (k, value) in remainder.iter().enumerate() {
        let angle = angle_step * (processed + k) as f64;
        let (sin_a, cos_a) = angle.sin_cos();
        re += value.re * cos_a - value.im * sin_a;
        im += value.re * sin_a + value.im * cos_a;
    }

    Complex::new(re, im)
}

/// Wigner Distribution Function
#[inline(always)]
pub fn wigner(_: f64, _: f64, _: f64) -> f64 {
    1.0
}

/// Choi-Williams Distribution Function
#[inline]
pub fn choi_williams(eta: f64, tau: f64, alpha: f64) -> f64 {
    (-alpha * (eta * eta) * (tau * tau) / 2.0).exp()
}

/// Rihaczek Distribution Function
#[inline]
pub fn rihaczek(eta: f64, tau: f64, alpha: f64) -> f64 {
    (-alpha * eta * tau).exp()
}

/// Zhao-Atlas-Marks Distribution Function
#[inline]
pub fn cone_shape(eta: f64, tau: f64, alpha: f64) -> f64 {
    let pi_eta_tau = std::f64::consts::PI * eta * tau;
    if pi_eta_tau == 0.0 {
        return 1.0;
    }
    (pi_eta_tau.sin() / pi_eta_tau) * (-2.0 * std::f64::consts::PI * alpha * tau.powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// Direct, unoptimized O(len^4) evaluation of Cohen's class distribution, kept only as a
    /// correctness reference for the SIMD/DFT-based implementation above.
    fn naive_reference<F: Fn(f64, f64, f64) -> f64>(
        input: &[f64],
        kernel: F,
        alpha: f64,
    ) -> Vec<Vec<f64>> {
        let len = input.len();
        let ambiguity = ambiguity(input);
        (0..len)
            .map(|time| {
                (0..len)
                    .map(|freq| {
                        let mut sum = 0.0;
                        for eta in 0..len {
                            for tau in 0..len {
                                let exponent =
                                    TAU * ((time * eta + freq * tau) as f64) / len as f64;
                                sum += (ambiguity[eta][tau]
                                    * kernel(eta as f64, tau as f64, alpha)
                                    * Complex::new(0.0, exponent).exp())
                                .re;
                            }
                        }
                        sum / (len * len) as f64
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_close(a: &[Vec<f64>], b: &[Vec<f64>], tolerance: f64) {
        assert_eq!(a.len(), b.len());
        for (row_a, row_b) in a.iter().zip(b) {
            assert_eq!(row_a.len(), row_b.len());
            for (&x, &y) in row_a.iter().zip(row_b) {
                assert!((x - y).abs() < tolerance, "expected {y}, got {x}");
            }
        }
    }

    #[test]
    fn matches_naive_reference() {
        let mut rng = StdRng::seed_from_u64(42);
        let input: Vec<f64> = (0..21).map(|_| rng.gen_range(-1.0..1.0)).collect();

        for (kernel, alpha) in [
            (wigner as fn(f64, f64, f64) -> f64, 0.001),
            (choi_williams, 0.05),
            (rihaczek, 0.02),
            (cone_shape, 0.001),
        ] {
            let optimized = bilinear_tf_distribution(&input, kernel, alpha);
            let reference = naive_reference(&input, kernel, alpha);
            assert_close(&optimized, &reference, 1e-8);
        }
    }

    #[test]
    fn runs_on_realistic_signal_length() {
        let mut input = vec![0.0; 128];
        input.par_iter_mut().for_each(|element| {
            *element = thread_rng().gen_range(-1.0..1.0);
        });
        let result = bilinear_tf_distribution(&input, cone_shape, 0.001);

        assert_eq!(result.len(), 128);
        for row in &result {
            assert_eq!(row.len(), 128);
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
}
