use num_complex::Complex;

/// Cohen's Class Distribution Function.
pub fn bilinear_tf_distribution<F : Fn(f64, f64, f64) -> f64>(input : &[f64], kernel: F, alpha : f64) -> Vec<Vec<f64>>
{
    let len = input.len();
    let ambiguity = ambiguity(input);

    let mut kernel_applied = vec![vec![Complex::new(0.0, 0.0); len]; len];
    for eta in 0..len 
    {
        for tau in 0..len
        {
            kernel_applied[eta][tau] = ambiguity[eta][tau] * Complex::new(kernel(eta as f64, tau as f64, alpha), 0.0);
        }
    }

    let mut distribution = vec![vec![0.0; len]; len];
    for t in 0..len
    {
        for f in 0..len
        {
            let mut sum = Complex::new(0.0, 0.0);
            for eta in 0..len
            {
                for tau in 0..len { sum += kernel_applied[eta][tau] * Complex::new(0.0, 2.0 * std::f64::consts::PI * ((t * eta + f * tau) as f64) / len as f64).exp(); }
            }
            distribution[t][f] = sum.re / (len * len) as f64;
        }
    }
    distribution
}


/// Ambiguity Function.
fn ambiguity(input: &[f64]) -> Vec<Vec<Complex<f64>>>
{
    let len = input.len();
    let mut ambiguity = vec![vec![Complex::new(0.0, 0.0); len]; len];
    
    for eta in 0..len
    {
        for tau in 0..len
        {
            let mut sum = Complex::new(0.0, 0.0);
            for t in 0..len
            {
                if (t + tau / 2) < len && (t - tau / 2) > 0
                {
                    sum += Complex::new(input[ t + tau / 2], 0.0) * Complex::new(input[t - tau / 2], 0.0).conj() * Complex::new(0.0, -2.0 * std::f64::consts::PI * (eta as f64) * (t as f64) / len as f64).exp();
                }
            }
            ambiguity[eta][tau] = sum;
        }
    }
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
