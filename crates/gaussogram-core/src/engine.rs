//! The transform engine: holds a `Scheme`, cached FFT plans, and runs the
//! forward / inverse pipelines. The scheme is data — `forward` walks
//! `scheme.bands` with no per-scheme branching in the hot loop.

use num_complex::Complex;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::GaussogramError;
use crate::fft::{ComplexPlan, FftBackend, RealForwardPlan, RealInversePlan};
use crate::scheme::Scheme;

/// Reusable per-thread scratch. Allocated once per thread (via `forward_batch`'s
/// `for_each_init`) or once per call (`forward`), never inside the band loop.
pub struct Scratch {
    /// Source spectrum: length `N/2+1` (real schemes) or `N` (complex scheme).
    spectrum: Vec<Complex<f64>>,
    /// Real-input copy buffer (real schemes only).
    real_in: Vec<f64>,
    /// Per-band working buffer, sized to the maximum band width.
    band: Vec<Complex<f64>>,
    /// FFT scratch, sized to the maximum plan scratch length.
    fft: Vec<Complex<f64>>,
}

impl Scratch {
    fn new(spectrum_len: usize, n: usize, max_width: usize, max_scratch: usize) -> Self {
        Scratch {
            spectrum: vec![Complex::new(0.0, 0.0); spectrum_len],
            real_in: vec![0.0; n],
            band: vec![Complex::new(0.0, 0.0); max_width.max(1)],
            fft: vec![Complex::new(0.0, 0.0); max_scratch.max(1)],
        }
    }
}

enum Forward {
    Real(Arc<dyn RealForwardPlan>),
    Complex(Arc<dyn ComplexPlan>),
}

pub struct Gaussogram1d {
    scheme: Scheme,
    forward: Forward,
    /// Inverse band plans keyed by band width (used by `forward`).
    inverse_band: HashMap<usize, Arc<dyn ComplexPlan>>,
    /// Forward band plans keyed by band width (used by `inverse`).
    forward_band: HashMap<usize, Arc<dyn ComplexPlan>>,
    /// Real inverse plan of length N (used by `inverse`).
    real_inverse: Option<Arc<dyn RealInversePlan>>,
    spectrum_len: usize,
    max_width: usize,
    max_scratch: usize,
}

impl Gaussogram1d {
    pub fn new(scheme: Scheme, backend: &dyn FftBackend) -> Self {
        let n = scheme.n;
        let spectrum_len = if scheme.complex_input { n } else { n / 2 + 1 };

        let forward = if scheme.complex_input {
            Forward::Complex(backend.complex_forward(n))
        } else {
            Forward::Real(backend.real_forward(n))
        };

        let mut inverse_band: HashMap<usize, Arc<dyn ComplexPlan>> = HashMap::new();
        let mut forward_band: HashMap<usize, Arc<dyn ComplexPlan>> = HashMap::new();
        let mut max_width = 1usize;
        let mut max_scratch = match &forward {
            Forward::Real(p) => p.scratch_len(),
            Forward::Complex(p) => p.scratch_len(),
        };

        for band in &scheme.bands {
            let w = band.width();
            max_width = max_width.max(w);
            inverse_band.entry(w).or_insert_with(|| {
                let p = backend.complex_inverse(w);
                max_scratch = max_scratch.max(p.scratch_len());
                p
            });
            if scheme.invertible {
                forward_band.entry(w).or_insert_with(|| {
                    let p = backend.complex_forward(w);
                    max_scratch = max_scratch.max(p.scratch_len());
                    p
                });
            }
        }

        let real_inverse = if scheme.invertible {
            let p = backend.real_inverse(n);
            max_scratch = max_scratch.max(p.scratch_len());
            Some(p)
        } else {
            None
        };

        Gaussogram1d {
            scheme,
            forward,
            inverse_band,
            forward_band,
            real_inverse,
            spectrum_len,
            max_width,
            max_scratch,
        }
    }

    pub fn scheme(&self) -> &Scheme {
        &self.scheme
    }

    pub fn output_len(&self) -> usize {
        self.scheme.output_len
    }

    /// Allocate a reusable [`Scratch`] sized for this engine. Callers running
    /// many transforms (tight loops, per-thread workers, a Python handle) should
    /// allocate this once and pass it to the `*_with` methods to avoid
    /// re-allocating `spectrum`, `real_in`, `band`, and `fft` on every call.
    pub fn alloc_scratch(&self) -> Scratch {
        Scratch::new(
            self.spectrum_len,
            self.scheme.n,
            self.max_width,
            self.max_scratch,
        )
    }

    /// Walk the bands: copy source slice -> apply window -> inverse FFT -> write.
    /// Assumes `scratch.spectrum` already holds the (un-windowed) source spectrum.
    fn run_bands(&self, out: &mut [Complex<f64>], scratch: &mut Scratch) {
        for (band, window) in self.scheme.bands.iter().zip(self.scheme.windows.iter()) {
            let w = band.width();
            let src = &scratch.spectrum[band.src_lo..band.src_hi];
            let buf = &mut scratch.band[..w];
            for (b, (&s, &sc)) in buf.iter_mut().zip(src.iter().zip(window.screen.iter())) {
                *b = s * sc;
            }
            let plan = &self.inverse_band[&w];
            plan.process(buf, &mut scratch.fft);
            let inv = 1.0 / w as f64;
            for (o, b) in out[band.out_off..band.out_off + w].iter_mut().zip(buf.iter()) {
                *o = *b * inv;
            }
        }
    }

    /// Forward transform for real-input schemes, reusing a caller-owned
    /// [`Scratch`]. This is the allocation-free hot path: build the engine and
    /// scratch once, then call this repeatedly.
    ///
    /// Note: `realfft` mutates its input buffer, so the real signal is copied
    /// into `scratch.real_in` once per call. This single copy is unavoidable
    /// with the immutable `&[f64]` contract (a c2r plan cannot borrow the
    /// caller's slice read-only); it is *not* a per-call heap allocation.
    pub fn forward_with(
        &self,
        signal: &[f64],
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        if self.scheme.complex_input {
            return Err(GaussogramError::WrongInputKind {
                expected_complex: true,
            });
        }
        if signal.len() != self.scheme.n {
            return Err(GaussogramError::InputLenMismatch {
                expected: self.scheme.n,
                got: signal.len(),
            });
        }
        if out.len() != self.scheme.output_len {
            return Err(GaussogramError::OutputLenMismatch {
                expected: self.scheme.output_len,
                got: out.len(),
            });
        }
        let Forward::Real(plan) = &self.forward else {
            unreachable!("real scheme has a real forward plan");
        };
        scratch.real_in.copy_from_slice(signal);
        let (input, _) = scratch.real_in.split_at_mut(self.scheme.n);
        plan.process(input, &mut scratch.spectrum, &mut scratch.fft);
        self.run_bands(out, scratch);
        Ok(())
    }

    /// Forward transform for real-input schemes (`dyadic_real`, `dyadic_dual_real`).
    /// Convenience wrapper that allocates scratch per call; use
    /// [`Gaussogram1d::forward_with`] in tight loops.
    pub fn forward(&self, signal: &[f64], out: &mut [Complex<f64>]) -> Result<(), GaussogramError> {
        let mut scratch = self.alloc_scratch();
        self.forward_with(signal, out, &mut scratch)
    }

    /// Batch forward over independent segments, parallelised with rayon.
    /// `out` is the concatenation of each segment's `output_len` coefficients.
    pub fn forward_batch(
        &self,
        signals: &[&[f64]],
        out: &mut [Complex<f64>],
    ) -> Result<(), GaussogramError> {
        if self.scheme.complex_input {
            return Err(GaussogramError::WrongInputKind {
                expected_complex: true,
            });
        }
        let olen = self.scheme.output_len;
        if out.len() != olen * signals.len() {
            return Err(GaussogramError::OutputLenMismatch {
                expected: olen * signals.len(),
                got: out.len(),
            });
        }
        for s in signals {
            if s.len() != self.scheme.n {
                return Err(GaussogramError::InputLenMismatch {
                    expected: self.scheme.n,
                    got: s.len(),
                });
            }
        }

        use rayon::prelude::*;
        let result: Result<(), GaussogramError> = out
            .par_chunks_mut(olen)
            .zip(signals.par_iter())
            .try_for_each_init(
                || self.alloc_scratch(),
                |scratch, (chunk, signal)| self.forward_with(signal, chunk, scratch),
            );
        result
    }

    /// Forward transform for the complex-input scheme (`dyadic_complex`),
    /// reusing a caller-owned [`Scratch`].
    pub fn forward_complex_with(
        &self,
        signal: &[Complex<f64>],
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        if !self.scheme.complex_input {
            return Err(GaussogramError::WrongInputKind {
                expected_complex: false,
            });
        }
        if signal.len() != self.scheme.n {
            return Err(GaussogramError::InputLenMismatch {
                expected: self.scheme.n,
                got: signal.len(),
            });
        }
        if out.len() != self.scheme.output_len {
            return Err(GaussogramError::OutputLenMismatch {
                expected: self.scheme.output_len,
                got: out.len(),
            });
        }
        let Forward::Complex(plan) = &self.forward else {
            unreachable!("complex scheme has a complex forward plan");
        };
        scratch.spectrum.copy_from_slice(signal);
        plan.process(&mut scratch.spectrum, &mut scratch.fft);
        self.run_bands(out, scratch);
        Ok(())
    }

    /// Forward transform for the complex-input scheme (`dyadic_complex`).
    /// Convenience wrapper that allocates scratch per call.
    pub fn forward_complex(
        &self,
        signal: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), GaussogramError> {
        let mut scratch = self.alloc_scratch();
        self.forward_complex_with(signal, out, &mut scratch)
    }

    /// Inverse transform (only for invertible schemes, currently `dyadic_real`),
    /// reusing a caller-owned [`Scratch`].
    pub fn inverse_with(
        &self,
        coeffs: &[Complex<f64>],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        if !self.scheme.invertible {
            return Err(GaussogramError::NotInvertible(self.scheme.name));
        }
        if coeffs.len() != self.scheme.output_len {
            return Err(GaussogramError::OutputLenMismatch {
                expected: self.scheme.output_len,
                got: coeffs.len(),
            });
        }
        if out.len() != self.scheme.n {
            return Err(GaussogramError::InputLenMismatch {
                expected: self.scheme.n,
                got: out.len(),
            });
        }

        // Rebuild the half-spectrum band by band: FFT(coeffs_band) recovers
        // S_slice * screen; divide by screen to recover S_slice.
        for (band, window) in self.scheme.bands.iter().zip(self.scheme.windows.iter()) {
            let w = band.width();
            let buf = &mut scratch.band[..w];
            buf.copy_from_slice(&coeffs[band.out_off..band.out_off + w]);
            let plan = &self.forward_band[&w];
            plan.process(buf, &mut scratch.fft);
            for (dst, (&val, &sc)) in scratch.spectrum[band.src_lo..band.src_hi]
                .iter_mut()
                .zip(buf.iter().zip(window.screen.iter()))
            {
                *dst = val / sc;
            }
        }

        let plan = self
            .real_inverse
            .as_ref()
            .expect("invertible scheme has a real inverse plan");
        let n = self.scheme.n;
        // The half-spectrum's DC and Nyquist bins are real for a real signal;
        // clear any imaginary float noise so the c2r inverse accepts them.
        scratch.spectrum[0].im = 0.0;
        scratch.spectrum[n / 2].im = 0.0;
        plan.process(&mut scratch.spectrum, out, &mut scratch.fft);
        let inv = 1.0 / n as f64;
        for v in out.iter_mut() {
            *v *= inv;
        }
        Ok(())
    }

    /// Inverse transform (only for invertible schemes, currently `dyadic_real`).
    /// Convenience wrapper that allocates scratch per call.
    pub fn inverse(
        &self,
        coeffs: &[Complex<f64>],
        out: &mut [f64],
    ) -> Result<(), GaussogramError> {
        let mut scratch = self.alloc_scratch();
        self.inverse_with(coeffs, out, &mut scratch)
    }
}
