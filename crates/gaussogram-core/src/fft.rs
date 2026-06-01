//! FFT backend behind a trait so the algorithm is independent of the FFT
//! library. The default backend is pure-Rust `rustfft` + `realfft`
//! (feature `fft-rustfft`).

use num_complex::Complex;
use std::sync::Arc;

/// A cached, reusable complex FFT plan (forward or inverse, unnormalised).
pub trait ComplexPlan: Send + Sync {
    fn len(&self) -> usize;
    fn scratch_len(&self) -> usize;
    /// Transform `buf` in place using caller-provided `scratch`.
    fn process(&self, buf: &mut [Complex<f64>], scratch: &mut [Complex<f64>]);
}

/// A cached real-to-complex forward FFT plan producing the half-spectrum
/// `0..=N/2` (length `N/2 + 1`).
pub trait RealForwardPlan: Send + Sync {
    fn len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn scratch_len(&self) -> usize;
    fn process(&self, input: &mut [f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]);
}

/// A cached complex-to-real inverse FFT plan (consumes the half-spectrum
/// `0..=N/2`, produces `N` real samples; unnormalised).
pub trait RealInversePlan: Send + Sync {
    fn len(&self) -> usize;
    fn input_len(&self) -> usize;
    fn scratch_len(&self) -> usize;
    fn process(&self, input: &mut [Complex<f64>], output: &mut [f64], scratch: &mut [Complex<f64>]);
}

/// Plan factory. Implementations cache nothing themselves; the engine builds
/// each plan once and shares it via `Arc`.
pub trait FftBackend: Send + Sync {
    fn complex_forward(&self, len: usize) -> Arc<dyn ComplexPlan>;
    fn complex_inverse(&self, len: usize) -> Arc<dyn ComplexPlan>;
    fn real_forward(&self, len: usize) -> Arc<dyn RealForwardPlan>;
    fn real_inverse(&self, len: usize) -> Arc<dyn RealInversePlan>;
}

#[cfg(feature = "fft-rustfft")]
mod rustfft_backend {
    use super::*;
    use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
    use rustfft::{Fft, FftDirection, FftPlanner};

    struct RustfftComplexPlan {
        fft: Arc<dyn Fft<f64>>,
    }
    impl ComplexPlan for RustfftComplexPlan {
        fn len(&self) -> usize {
            self.fft.len()
        }
        fn scratch_len(&self) -> usize {
            self.fft.get_inplace_scratch_len()
        }
        fn process(&self, buf: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
            self.fft.process_with_scratch(buf, scratch);
        }
    }

    struct RustRealForwardPlan {
        plan: Arc<dyn RealToComplex<f64>>,
    }
    impl RealForwardPlan for RustRealForwardPlan {
        fn len(&self) -> usize {
            self.plan.len()
        }
        fn output_len(&self) -> usize {
            self.plan.len() / 2 + 1
        }
        fn scratch_len(&self) -> usize {
            self.plan.get_scratch_len()
        }
        fn process(
            &self,
            input: &mut [f64],
            output: &mut [Complex<f64>],
            scratch: &mut [Complex<f64>],
        ) {
            self.plan
                .process_with_scratch(input, output, scratch)
                .expect("real forward fft");
        }
    }

    struct RustRealInversePlan {
        plan: Arc<dyn ComplexToReal<f64>>,
    }
    impl RealInversePlan for RustRealInversePlan {
        fn len(&self) -> usize {
            self.plan.len()
        }
        fn input_len(&self) -> usize {
            self.plan.len() / 2 + 1
        }
        fn scratch_len(&self) -> usize {
            self.plan.get_scratch_len()
        }
        fn process(
            &self,
            input: &mut [Complex<f64>],
            output: &mut [f64],
            scratch: &mut [Complex<f64>],
        ) {
            self.plan
                .process_with_scratch(input, output, scratch)
                .expect("real inverse fft");
        }
    }

    /// Default backend: `rustfft` for complex, `realfft` for real transforms.
    pub struct RustFftBackend {
        complex: std::sync::Mutex<FftPlanner<f64>>,
        real: std::sync::Mutex<RealFftPlanner<f64>>,
    }

    impl RustFftBackend {
        pub fn new() -> Self {
            RustFftBackend {
                complex: std::sync::Mutex::new(FftPlanner::new()),
                real: std::sync::Mutex::new(RealFftPlanner::new()),
            }
        }
    }

    impl Default for RustFftBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FftBackend for RustFftBackend {
        fn complex_forward(&self, len: usize) -> Arc<dyn ComplexPlan> {
            let fft = self
                .complex
                .lock()
                .unwrap()
                .plan_fft(len, FftDirection::Forward);
            Arc::new(RustfftComplexPlan { fft })
        }
        fn complex_inverse(&self, len: usize) -> Arc<dyn ComplexPlan> {
            let fft = self
                .complex
                .lock()
                .unwrap()
                .plan_fft(len, FftDirection::Inverse);
            Arc::new(RustfftComplexPlan { fft })
        }
        fn real_forward(&self, len: usize) -> Arc<dyn RealForwardPlan> {
            let plan = self.real.lock().unwrap().plan_fft_forward(len);
            Arc::new(RustRealForwardPlan { plan })
        }
        fn real_inverse(&self, len: usize) -> Arc<dyn RealInversePlan> {
            let plan = self.real.lock().unwrap().plan_fft_inverse(len);
            Arc::new(RustRealInversePlan { plan })
        }
    }
}

#[cfg(feature = "fft-rustfft")]
pub use rustfft_backend::RustFftBackend;

/// The default backend type.
#[cfg(feature = "fft-rustfft")]
pub type DefaultBackend = RustFftBackend;
