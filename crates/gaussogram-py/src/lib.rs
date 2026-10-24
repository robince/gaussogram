//! PyO3 extension module `_native`, wrapped by the thin `python/gaussogram`
//! package. Accepts/returns NumPy arrays; all heavy lifting is in
//! `gaussogram-core`.
//!
//! Two entry styles:
//! * free functions (`gft1d_real`, `gft1d`, `inverse_real`) — convenience, they
//!   build a fresh engine per call;
//! * the [`Gaussogram1d`] class — builds the engine, FFT plans, and scratch once
//!   and reuses them across `.forward()` / `.inverse()` calls (the hot path).
//!
//! Zero-copy + GIL release: inputs are borrowed from NumPy (never converted or
//! copied at the interface), and the GIL is released during the transform.
//! The FFTs work in place, so the core copies its input into scratch once per
//! call anyway; that copy is done *while the GIL is held* (`load_real` /
//! `load_complex` / `load_coeffs`), and the GIL-free part only touches scratch
//! and the output. Python threads can therefore never race the reader, and
//! the caller's array flags are never touched. The one exception is
//! `forward_batch`, which reads its rows in place (copying a whole batch would
//! double its memory) and so marks the input read-only for the duration (see
//! [`WriteableGuard`]).

use num_complex::Complex;
use numpy::npyffi::{PyArrayObject, NPY_ARRAY_WRITEABLE};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use std::os::raw::c_int;
use std::sync::{Arc, Mutex};

use gaussogram_core::{
    cached_engine, clear_engine_cache as core_clear_engine_cache, complex_partitions as core_complex_partitions, legacy_real_partitions,
    Gaussogram1d as CoreEngine, GaussogramError, SchemeKind, SchemeSpec, Scratch, WindowKind,
};

fn map_err(e: GaussogramError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn window_kind(name: &str) -> PyResult<WindowKind> {
    WindowKind::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "window_type must be 'gaussian' or 'box' (got '{name}')"
        ))
    })
}

/// Shared (cached) engine for the named scheme. Building windows and FFT plans
/// is far more expensive than a transform, so the free functions and handles
/// reuse engines from the core's process-wide cache.
fn build_engine(
    scheme: &str,
    n: usize,
    window: WindowKind,
    nyquist_flat_top: bool,
    bands_per_octave: usize,
) -> PyResult<Arc<CoreEngine>> {
    // Reject non-default options a scheme does not support rather than
    // silently ignoring them (a reproducibility hazard).
    let kind = match scheme {
        "dyadic_dual_real" => SchemeKind::DyadicDualReal,
        "dyadic_real" => {
            if nyquist_flat_top {
                return Err(PyValueError::new_err(
                    "nyquist_flat_top is only supported by 'dyadic_dual_real'",
                ));
            }
            SchemeKind::DyadicReal
        }
        "dyadic_complex" => {
            if nyquist_flat_top || bands_per_octave != 1 {
                return Err(PyValueError::new_err(
                    "'dyadic_complex' supports neither nyquist_flat_top nor bands_per_octave",
                ));
            }
            SchemeKind::DyadicComplex
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme '{other}' (expected 'dyadic_dual_real', 'dyadic_real', or 'dyadic_complex')"
            )))
        }
    };
    cached_engine(SchemeSpec {
        kind,
        n,
        window,
        nyquist_flat_top,
        bands_per_octave,
    })
    .map_err(map_err)
}

/// Arrays currently held read-only by a [`WriteableGuard`]: (array address,
/// active guards, whether the first guard cleared `WRITEABLE`).
static GUARDED: Mutex<Vec<(usize, usize, bool)>> = Mutex::new(Vec::new());

/// RAII guard that clears a NumPy array's `WRITEABLE` flag while the GIL is
/// released for a transform that reads the array in place, and sets it again
/// when the last overlapping guard on that array is dropped. This turns
/// concurrent Python-side mutation *through this array object* into a hard
/// error rather than a silent data race against the Rust reader. It is
/// best-effort: other views or the base array sharing the same buffer keep
/// their own flags and can still mutate the data.
///
/// Guards on the same array are reference counted, so one call finishing
/// cannot re-enable writes while another is still reading, and an array that
/// was read-only to begin with stays read-only.
///
/// Construct and drop this only while holding the GIL.
struct WriteableGuard {
    ptr: *mut PyArrayObject,
}

impl WriteableGuard {
    /// SAFETY: `ptr` must be a valid NumPy array object that outlives the
    /// guard, and the GIL must be held.
    unsafe fn lock(ptr: *mut PyArrayObject) -> Self {
        let mut guarded = GUARDED.lock().unwrap_or_else(|e| e.into_inner());
        let key = ptr as usize;
        if let Some(entry) = guarded.iter_mut().find(|e| e.0 == key) {
            entry.1 += 1;
        } else {
            let bit = NPY_ARRAY_WRITEABLE as c_int;
            let cleared = (*ptr).flags & bit != 0;
            if cleared {
                (*ptr).flags &= !bit;
            }
            guarded.push((key, 1, cleared));
        }
        WriteableGuard { ptr }
    }
}

impl Drop for WriteableGuard {
    fn drop(&mut self) {
        // GIL is held here (drop runs back on the Python-facing side).
        let mut guarded = GUARDED.lock().unwrap_or_else(|e| e.into_inner());
        let key = self.ptr as usize;
        let Some(pos) = guarded.iter().position(|e| e.0 == key) else {
            return;
        };
        guarded[pos].1 -= 1;
        if guarded[pos].1 == 0 {
            let (_, _, cleared) = guarded.swap_remove(pos);
            if cleared {
                unsafe {
                    (*self.ptr).flags |= NPY_ARRAY_WRITEABLE as c_int;
                }
            }
        }
    }
}

/// Forward GFT of a real signal. Default scheme `dyadic_dual_real` (output N-1).
///
/// Convenience wrapper: engines come from a small process-wide cache, but
/// scratch and output are allocated per call. For repeated transforms of the
/// same size/scheme, construct a `Gaussogram1d` once and reuse it.
#[pyfunction]
#[pyo3(signature = (x, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false, bands_per_octave=1))]
fn gft1d_real<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
    bands_per_octave: usize,
) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
    let kind = window_kind(window_type)?;
    let signal = x.as_slice()?;
    let n = signal.len();
    let engine = build_engine(scheme, n, kind, nyquist_flat_top, bands_per_octave)?;
    if engine.scheme().complex_input {
        return Err(PyValueError::new_err(
            "gft1d_real requires a real-input scheme; use gft1d for dyadic_complex",
        ));
    }
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    engine.load_real(signal, &mut scratch).map_err(map_err)?;
    py.allow_threads(|| engine.forward_loaded(&mut out, &mut scratch))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

/// Forward GFT of a complex signal using the symmetric `dyadic_complex` scheme
/// (output length N). Port of the legacy `gft1d`.
#[pyfunction]
#[pyo3(signature = (z, window_type="gaussian"))]
fn gft1d<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, Complex<f64>>,
    window_type: &str,
) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
    let kind = window_kind(window_type)?;
    let signal = z.as_slice()?;
    let n = signal.len();
    let engine = build_engine("dyadic_complex", n, kind, false, 1)?;
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    engine.load_complex(signal, &mut scratch).map_err(map_err)?;
    py.allow_threads(|| engine.forward_complex_loaded(&mut out, &mut scratch))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

/// Inverse of the invertible `dyadic_real` scheme. `coeffs` has length N/2+1.
/// `bands_per_octave` must match the value used for the forward transform.
#[pyfunction]
#[pyo3(signature = (coeffs, window_type="gaussian", bands_per_octave=1))]
fn inverse_real<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArray1<'py, Complex<f64>>,
    window_type: &str,
    bands_per_octave: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = window_kind(window_type)?;
    let c = coeffs.as_slice()?;
    if c.len() < 3 {
        return Err(PyValueError::new_err(format!(
            "coeffs must have length N/2 + 1 >= 3 (got {})",
            c.len()
        )));
    }
    // Recover N from output_len = N/2 + 1 (invariant to bands_per_octave).
    let n = (c.len() - 1) * 2;
    let engine = build_engine("dyadic_real", n, kind, false, bands_per_octave)?;
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![0.0f64; n];
    engine.load_coeffs(c, &mut scratch).map_err(map_err)?;
    py.allow_threads(|| engine.inverse_loaded(&mut out, &mut scratch))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

/// Reusable transform handle: builds the FFT plans and scratch buffers once and
/// reuses them across calls. Use this instead of the free functions for tight
/// loops or repeated transforms of the same size and scheme.
///
/// An instance owns mutable scratch, so a single instance is **not** safe to
/// call concurrently from multiple Python threads — create one per thread.
#[pyclass(name = "Gaussogram1d", module = "gaussogram._native")]
struct PyGaussogram {
    engine: Arc<CoreEngine>,
    scratch: Scratch,
    n: usize,
    output_len: usize,
    complex_input: bool,
    invertible: bool,
}

#[pymethods]
impl PyGaussogram {
    #[new]
    #[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false, bands_per_octave=1))]
    fn new(
        n: usize,
        scheme: &str,
        window_type: &str,
        nyquist_flat_top: bool,
        bands_per_octave: usize,
    ) -> PyResult<Self> {
        let kind = window_kind(window_type)?;
        let engine = build_engine(scheme, n, kind, nyquist_flat_top, bands_per_octave)?;
        let complex_input = engine.scheme().complex_input;
        let invertible = engine.scheme().invertible;
        let output_len = engine.output_len();
        let scratch = engine.alloc_scratch();
        Ok(PyGaussogram {
            engine,
            scratch,
            n,
            output_len,
            complex_input,
            invertible,
        })
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn output_len(&self) -> usize {
        self.output_len
    }

    #[getter]
    fn invertible(&self) -> bool {
        self.invertible
    }

    /// Forward transform of a real signal of length `n`, reusing the handle's
    /// plans and scratch. The input is borrowed from NumPy and copied into the
    /// handle's scratch (the FFT works in place) before the GIL is released.
    ///
    /// `out`, if given, must be a C-contiguous complex128 array of length
    /// `output_len`; the coefficients are written into it (no allocation) and
    /// it is returned. As with NumPy's `out=`, it is written with the GIL
    /// released, so other threads must not use it during the call.
    #[pyo3(signature = (x, out=None))]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        out: Option<Bound<'py, PyArray1<Complex<f64>>>>,
    ) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
        if self.complex_input {
            return Err(PyValueError::new_err(
                "this handle uses a complex-input scheme; use forward_complex",
            ));
        }
        let signal = x.as_slice()?;
        // Disjoint field borrows: engine (shared) + scratch (exclusive).
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        engine.load_real(signal, scratch).map_err(map_err)?;
        // Without `out`, write into a fresh NumPy array through the same path.
        let arr = out.unwrap_or_else(|| PyArray1::zeros(py, self.output_len, false));
        {
            let mut rw = arr.try_readwrite()?;
            let dst = rw.as_slice_mut()?;
            py.allow_threads(|| engine.forward_loaded(dst, scratch))
                .map_err(map_err)?;
        }
        Ok(arr)
    }

    /// Forward transform of a batch of real signals, one per row of a
    /// C-contiguous `(count, n)` float64 array, parallelised across rows with
    /// the GIL released. Returns `(count, output_len)` complex128, or writes
    /// into `out` of that shape (C-contiguous) and returns it.
    ///
    /// Rows are read in place (no copy of the batch); the input array is
    /// marked read-only for the duration. `out` follows the same rule as in
    /// `forward`: other threads must not use it during the call.
    #[pyo3(signature = (x, out=None))]
    fn forward_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        out: Option<Bound<'py, PyArray2<Complex<f64>>>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
        if self.complex_input {
            return Err(PyValueError::new_err(
                "forward_batch supports real-input schemes only",
            ));
        }
        let [count, n] = [x.shape()[0], x.shape()[1]];
        if n != self.n {
            return Err(PyValueError::new_err(format!(
                "x must have shape (count, {}) (got (count, {n}))",
                self.n
            )));
        }
        // `as_slice` also accepts Fortran order, which would hand us the
        // columns interleaved; require row-major rows explicitly.
        if !x.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "x must be C-contiguous; call np.ascontiguousarray() explicitly",
            ));
        }
        let arr_ptr = x.as_array_ptr();
        let flat = x.as_slice()?;
        let segs: Vec<&[f64]> = if n == 0 { Vec::new() } else { flat.chunks_exact(n).collect() };
        let engine = &self.engine;
        let olen = self.output_len;
        let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
        let arr = match out {
            Some(arr) => {
                if arr.shape() != [count, olen] || !arr.is_c_contiguous() {
                    return Err(PyValueError::new_err(format!(
                        "out must be C-contiguous with shape ({count}, {olen}) (got {:?})",
                        arr.shape()
                    )));
                }
                arr
            }
            // Without `out`, write into a fresh NumPy array through the same path.
            None => PyArray2::zeros(py, [count, olen], false),
        };
        {
            let mut rw = arr.try_readwrite()?;
            let dst = rw.as_slice_mut()?;
            py.allow_threads(|| engine.forward_batch(&segs, dst))
                .map_err(map_err)?;
        }
        Ok(arr)
    }

    /// Forward transform of a complex signal (for `dyadic_complex` handles).
    fn forward_complex<'py>(
        &mut self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, Complex<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
        if !self.complex_input {
            return Err(PyValueError::new_err(
                "this handle uses a real-input scheme; use forward",
            ));
        }
        let signal = z.as_slice()?;
        let mut out = vec![Complex::new(0.0, 0.0); self.output_len];
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        engine.load_complex(signal, scratch).map_err(map_err)?;
        py.allow_threads(|| engine.forward_complex_loaded(&mut out, scratch))
            .map_err(map_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Inverse transform (only for invertible schemes, e.g. `dyadic_real`).
    fn inverse<'py>(
        &mut self,
        py: Python<'py>,
        coeffs: PyReadonlyArray1<'py, Complex<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.invertible {
            return Err(PyValueError::new_err("this scheme is not invertible"));
        }
        let c = coeffs.as_slice()?;
        let mut out = vec![0.0f64; self.n];
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        engine.load_coeffs(c, scratch).map_err(map_err)?;
        py.allow_threads(|| engine.inverse_loaded(&mut out, scratch))
            .map_err(map_err)?;
        Ok(out.into_pyarray(py))
    }
}

/// Complex-scheme partition boundaries (legacy `partitions`).
#[pyfunction]
fn partitions<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let pars = core_complex_partitions(n)
        .map_err(|_| PyValueError::new_err("size must be a power of two and at least 8"))?;
    let pars: Vec<i32> = pars.iter().map(|&x| x as i32).collect();
    Ok(pars.into_pyarray(py))
}

/// Legacy real-scheme partition boundaries (legacy `real_partitions`).
#[pyfunction]
fn real_partitions<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let pars = legacy_real_partitions(n)
        .map_err(|_| PyValueError::new_err("size must be a power of two and at least 4"))?;
    let pars: Vec<i32> = pars.iter().map(|&x| x as i32).collect();
    Ok(pars.into_pyarray(py))
}

/// Band layout of a scheme: parallel arrays (src_lo, width, fcentre, out_off).
/// Used by Python helpers to map packed coefficients onto a frequency-time grid.
#[pyfunction]
#[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false, bands_per_octave=1))]
fn scheme_bands<'py>(
    py: Python<'py>,
    n: usize,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
    bands_per_octave: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let kind = window_kind(window_type)?;
    let engine = build_engine(scheme, n, kind, nyquist_flat_top, bands_per_octave)?;
    let s = engine.scheme();
    let mut lo = Vec::with_capacity(s.bands.len());
    let mut width = Vec::with_capacity(s.bands.len());
    let mut fcentre = Vec::with_capacity(s.bands.len());
    let mut out_off = Vec::with_capacity(s.bands.len());
    for (b, w) in s.bands.iter().zip(s.windows.iter()) {
        lo.push(b.src_lo as i64);
        width.push(b.width() as i64);
        fcentre.push(w.fcentre as i64);
        out_off.push(b.out_off as i64);
    }
    Ok((
        lo.into_pyarray(py),
        width.into_pyarray(py),
        fcentre.into_pyarray(py),
        out_off.into_pyarray(py),
    ))
}

/// Packed output length for a scheme.
#[pyfunction]
#[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false, bands_per_octave=1))]
fn output_len(
    n: usize,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
    bands_per_octave: usize,
) -> PyResult<usize> {
    let kind = window_kind(window_type)?;
    Ok(build_engine(scheme, n, kind, nyquist_flat_top, bands_per_octave)?.output_len())
}

/// Drop the process-wide engine cache, so the next construction is cold.
#[pyfunction]
fn clear_engine_cache() {
    core_clear_engine_cache();
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gft1d_real, m)?)?;
    m.add_function(wrap_pyfunction!(gft1d, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_real, m)?)?;
    m.add_function(wrap_pyfunction!(partitions, m)?)?;
    m.add_function(wrap_pyfunction!(real_partitions, m)?)?;
    m.add_function(wrap_pyfunction!(scheme_bands, m)?)?;
    m.add_function(wrap_pyfunction!(output_len, m)?)?;
    m.add_function(wrap_pyfunction!(clear_engine_cache, m)?)?;
    m.add_class::<PyGaussogram>()?;
    Ok(())
}
