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
    /// Per-band working buffer for the forward, sized to the maximum band width.
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

        // The inverse works in place on the spectrum buffer, which relies on
        // invertible schemes packing a disjoint half-spectrum cover in order.
        assert!(
            !scheme.invertible
                || (scheme.output_len == spectrum_len
                    && scheme.bands.iter().all(|b| b.out_off == b.src_lo)),
            "invertible scheme '{}' must pack its bands at out_off == src_lo",
            scheme.name
        );

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

    /// Copy a real signal into `scratch`, ready for [`Gaussogram1d::forward_loaded`].
    ///
    /// `realfft` uses its input buffer as workspace, so the signal must be
    /// copied once per call to leave the caller's slice untouched. Splitting the
    /// copy from the transform lets a binding read the caller's buffer while it
    /// still holds its interpreter lock and run the transform without it, so
    /// the caller's memory is never read concurrently with foreign mutation.
    pub fn load_real(&self, signal: &[f64], scratch: &mut Scratch) -> Result<(), GaussogramError> {
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
        scratch.real_in.copy_from_slice(signal);
        Ok(())
    }

    /// Forward transform of the real signal previously copied into `scratch` by
    /// [`Gaussogram1d::load_real`]. Consumes the loaded signal (`realfft`
    /// overwrites it), so call `load_real` again before the next transform.
    pub fn forward_loaded(
        &self,
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        if self.scheme.complex_input {
            return Err(GaussogramError::WrongInputKind {
                expected_complex: true,
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
        let (input, _) = scratch.real_in.split_at_mut(self.scheme.n);
        plan.process(input, &mut scratch.spectrum, &mut scratch.fft);
        self.run_bands(out, scratch);
        Ok(())
    }

    /// Forward transform for real-input schemes, reusing a caller-owned
    /// [`Scratch`]. This is the allocation-free hot path: build the engine and
    /// scratch once, then call this repeatedly. Equivalent to
    /// [`Gaussogram1d::load_real`] followed by [`Gaussogram1d::forward_loaded`].
    pub fn forward_with(
        &self,
        signal: &[f64],
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        self.load_real(signal, scratch)?;
        self.forward_loaded(out, scratch)
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

    /// Copy a complex signal into `scratch`, ready for
    /// [`Gaussogram1d::forward_complex_loaded`]. The complex FFT runs in place,
    /// so this copy is required; see [`Gaussogram1d::load_real`].
    pub fn load_complex(
        &self,
        signal: &[Complex<f64>],
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
        scratch.spectrum.copy_from_slice(signal);
        Ok(())
    }

    /// Forward transform of the complex signal previously copied into
    /// `scratch` by [`Gaussogram1d::load_complex`] (consumes it).
    pub fn forward_complex_loaded(
        &self,
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        if !self.scheme.complex_input {
            return Err(GaussogramError::WrongInputKind {
                expected_complex: false,
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
        plan.process(&mut scratch.spectrum, &mut scratch.fft);
        self.run_bands(out, scratch);
        Ok(())
    }

    /// Forward transform for the complex-input scheme (`dyadic_complex`),
    /// reusing a caller-owned [`Scratch`].
    pub fn forward_complex_with(
        &self,
        signal: &[Complex<f64>],
        out: &mut [Complex<f64>],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        self.load_complex(signal, scratch)?;
        self.forward_complex_loaded(out, scratch)
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

    /// Copy packed coefficients into `scratch`, ready for
    /// [`Gaussogram1d::inverse_loaded`]. As with [`Gaussogram1d::load_real`],
    /// this lets a binding read the caller's buffer only while it holds its
    /// interpreter lock.
    ///
    /// An invertible scheme is a disjoint cover of the half-spectrum packed in
    /// order (every band's `out_off == src_lo`, `output_len == N/2 + 1`), so the
    /// coefficients are copied straight into the spectrum buffer and each band
    /// is transformed in place there: no copy beyond this one.
    pub fn load_coeffs(
        &self,
        coeffs: &[Complex<f64>],
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
        scratch.spectrum.copy_from_slice(coeffs);
        Ok(())
    }

    /// Inverse transform of the coefficients previously copied into `scratch`
    /// by [`Gaussogram1d::load_coeffs`] (consumes them).
    pub fn inverse_loaded(&self, out: &mut [f64], scratch: &mut Scratch) -> Result<(), GaussogramError> {
        if !self.scheme.invertible {
            return Err(GaussogramError::NotInvertible(self.scheme.name));
        }
        if out.len() != self.scheme.n {
            return Err(GaussogramError::InputLenMismatch {
                expected: self.scheme.n,
                got: out.len(),
            });
        }

        // Rebuild the half-spectrum band by band, in place: FFT(coeffs_band)
        // recovers S_slice * screen; divide by screen to recover S_slice.
        for (band, window) in self.scheme.bands.iter().zip(self.scheme.windows.iter()) {
            let w = band.width();
            let buf = &mut scratch.spectrum[band.src_lo..band.src_hi];
            let plan = &self.forward_band[&w];
            plan.process(buf, &mut scratch.fft);
            for (v, &sc) in buf.iter_mut().zip(window.screen.iter()) {
                *v /= sc;
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

    /// Inverse transform (only for invertible schemes, currently `dyadic_real`),
    /// reusing a caller-owned [`Scratch`]. Equivalent to
    /// [`Gaussogram1d::load_coeffs`] followed by [`Gaussogram1d::inverse_loaded`].
    pub fn inverse_with(
        &self,
        coeffs: &[Complex<f64>],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), GaussogramError> {
        self.load_coeffs(coeffs, scratch)?;
        self.inverse_loaded(out, scratch)
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
