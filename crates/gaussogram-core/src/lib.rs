//! gaussogram-core — a parsimonious adaptive time-frequency representation
//! (fast Gaussian S-transform / GFT). Pure Rust, no FFI, no NumPy.
//!
//! The partitioning scheme is *data* (`Scheme` = `Band`s + `Window`s); the
//! [`Gaussogram1d`] engine walks it with no per-scheme branching. The FFT
//! library sits behind the [`FftBackend`] trait (default: `rustfft`/`realfft`).

pub mod engine;
pub mod error;
pub mod fft;
pub mod scheme;
pub mod window;

pub use engine::Gaussogram1d;
pub use error::{
    GaussogramError, STATUS_INPUT_LEN_MISMATCH, STATUS_INTERNAL, STATUS_NOT_INVERTIBLE, STATUS_OK,
    STATUS_OUTPUT_LEN_MISMATCH, STATUS_POWER_OF_TWO, STATUS_TOO_SMALL, STATUS_UNKNOWN_WINDOW,
    STATUS_WRONG_INPUT_KIND,
};
pub use fft::{ComplexPlan, FftBackend, RealForwardPlan, RealInversePlan};
pub use scheme::{
    complex_partitions, dyadic_complex, dyadic_complex_with, dyadic_dual_real,
    dyadic_dual_real_with, dyadic_real, dyadic_real_with, legacy_real_partitions, Band, Scheme,
    Window,
};
pub use window::WindowKind;

#[cfg(feature = "fft-rustfft")]
pub use fft::{DefaultBackend, RustFftBackend};

/// Convenience: build a `Gaussogram1d` for the named scheme using the default
/// backend. Window kind defaults to gaussian.
#[cfg(feature = "fft-rustfft")]
pub fn build(scheme: Scheme) -> Gaussogram1d {
    let backend = RustFftBackend::new();
    Gaussogram1d::new(scheme, &backend)
}
