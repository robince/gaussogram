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
//! Zero-copy + GIL release: inputs are borrowed (never silently copied), and the
//! GIL is released during the transform. To make that sound, the input array's
//! NumPy `WRITEABLE` flag is cleared for the duration (see [`WriteableGuard`]),
//! so a concurrent Python thread that tries to mutate it raises instead of
//! racing the reader.

use std::os::raw::c_int;

use num_complex::Complex;
use numpy::npyffi::{PyArrayObject, NPY_ARRAY_WRITEABLE};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use gaussogram_core::{
    build, complex_partitions as core_complex_partitions, dyadic_complex_with,
    dyadic_dual_real_with, dyadic_real_with, legacy_real_partitions, Gaussogram1d as CoreEngine,
    GaussogramError, Scheme, Scratch, WindowKind,
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

fn build_scheme(
    scheme: &str,
    n: usize,
    kind: WindowKind,
    nyquist_flat_top: bool,
) -> PyResult<Scheme> {
    let s = match scheme {
        "dyadic_dual_real" => dyadic_dual_real_with(n, kind, nyquist_flat_top),
        "dyadic_real" => dyadic_real_with(n, kind),
        "dyadic_complex" => dyadic_complex_with(n, kind),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme '{other}' (expected 'dyadic_dual_real', 'dyadic_real', or 'dyadic_complex')"
            )))
        }
    };
    s.map_err(map_err)
}

/// RAII guard that clears a NumPy array's `WRITEABLE` flag while the GIL is
/// released for a transform, then restores the original flags on drop. This
/// turns concurrent Python-side mutation of a borrowed input into a hard error
/// rather than a silent data race against the Rust reader.
///
/// Construct and drop this only while holding the GIL.
struct WriteableGuard {
    ptr: *mut PyArrayObject,
    saved: c_int,
}

impl WriteableGuard {
    /// SAFETY: `ptr` must be a valid NumPy array object and the GIL must be held.
    unsafe fn lock(ptr: *mut PyArrayObject) -> Self {
        let saved = (*ptr).flags;
        (*ptr).flags = saved & !(NPY_ARRAY_WRITEABLE as c_int);
        WriteableGuard { ptr, saved }
    }
}

impl Drop for WriteableGuard {
    fn drop(&mut self) {
        // GIL is held here (drop runs back on the Python-facing side).
        unsafe {
            (*self.ptr).flags = self.saved;
        }
    }
}

/// Forward GFT of a real signal. Default scheme `dyadic_dual_real` (output N-1).
///
/// Convenience wrapper: builds a fresh engine per call. For repeated transforms
/// of the same size/scheme, construct a `Gaussogram1d` once and reuse it.
#[pyfunction]
#[pyo3(signature = (x, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false))]
fn gft1d_real<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
    let kind = window_kind(window_type)?;
    let arr_ptr = x.as_array_ptr();
    let signal = x.as_slice()?;
    let n = signal.len();
    let s = build_scheme(scheme, n, kind, nyquist_flat_top)?;
    if s.complex_input {
        return Err(PyValueError::new_err(
            "gft1d_real requires a real-input scheme; use gft1d for dyadic_complex",
        ));
    }
    let engine = build(s);
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    {
        let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
        py.allow_threads(|| engine.forward_with(signal, &mut out, &mut scratch))
            .map_err(map_err)?;
    }
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
    let arr_ptr = z.as_array_ptr();
    let signal = z.as_slice()?;
    let n = signal.len();
    let s = build_scheme("dyadic_complex", n, kind, false)?;
    let engine = build(s);
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    {
        let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
        py.allow_threads(|| engine.forward_complex_with(signal, &mut out, &mut scratch))
            .map_err(map_err)?;
    }
    Ok(out.into_pyarray(py))
}

/// Inverse of the invertible `dyadic_real` scheme. `coeffs` has length N/2+1.
#[pyfunction]
#[pyo3(signature = (coeffs, window_type="gaussian"))]
fn inverse_real<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArray1<'py, Complex<f64>>,
    window_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = window_kind(window_type)?;
    let arr_ptr = coeffs.as_array_ptr();
    let c = coeffs.as_slice()?;
    // Recover N from output_len = N/2 + 1.
    let n = (c.len() - 1) * 2;
    let s = build_scheme("dyadic_real", n, kind, false)?;
    let engine = build(s);
    let mut scratch = engine.alloc_scratch();
    let mut out = vec![0.0f64; n];
    {
        let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
        py.allow_threads(|| engine.inverse_with(c, &mut out, &mut scratch))
            .map_err(map_err)?;
    }
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
    engine: CoreEngine,
    scratch: Scratch,
    n: usize,
    output_len: usize,
    complex_input: bool,
    invertible: bool,
}

#[pymethods]
impl PyGaussogram {
    #[new]
    #[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false))]
    fn new(n: usize, scheme: &str, window_type: &str, nyquist_flat_top: bool) -> PyResult<Self> {
        let kind = window_kind(window_type)?;
        let s = build_scheme(scheme, n, kind, nyquist_flat_top)?;
        let complex_input = s.complex_input;
        let invertible = s.invertible;
        let output_len = s.output_len;
        let engine = build(s);
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
    /// plans and scratch. The input is borrowed (not copied) and locked
    /// read-only while the GIL is released.
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
        if self.complex_input {
            return Err(PyValueError::new_err(
                "this handle uses a complex-input scheme; use forward_complex",
            ));
        }
        let arr_ptr = x.as_array_ptr();
        let signal = x.as_slice()?;
        let mut out = vec![Complex::new(0.0, 0.0); self.output_len];
        // Disjoint field borrows: engine (shared) + scratch (exclusive).
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        {
            let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
            py.allow_threads(|| engine.forward_with(signal, &mut out, scratch))
                .map_err(map_err)?;
        }
        Ok(out.into_pyarray(py))
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
        let arr_ptr = z.as_array_ptr();
        let signal = z.as_slice()?;
        let mut out = vec![Complex::new(0.0, 0.0); self.output_len];
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        {
            let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
            py.allow_threads(|| engine.forward_complex_with(signal, &mut out, scratch))
                .map_err(map_err)?;
        }
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
        let arr_ptr = coeffs.as_array_ptr();
        let c = coeffs.as_slice()?;
        let mut out = vec![0.0f64; self.n];
        let engine = &self.engine;
        let scratch = &mut self.scratch;
        {
            let _guard = unsafe { WriteableGuard::lock(arr_ptr) };
            py.allow_threads(|| engine.inverse_with(c, &mut out, scratch))
                .map_err(map_err)?;
        }
        Ok(out.into_pyarray(py))
    }
}

/// Complex-scheme partition boundaries (legacy `partitions`).
#[pyfunction]
fn partitions<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if n < 8 || (n & (n - 1)) != 0 {
        return Err(PyValueError::new_err(
            "size must be a power of two and at least 8",
        ));
    }
    let pars: Vec<i32> = core_complex_partitions(n).iter().map(|&x| x as i32).collect();
    Ok(pars.into_pyarray(py))
}

/// Legacy real-scheme partition boundaries (legacy `real_partitions`).
#[pyfunction]
fn real_partitions<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if n < 4 || (n & (n - 1)) != 0 {
        return Err(PyValueError::new_err(
            "size must be a power of two and at least 4",
        ));
    }
    let pars: Vec<i32> = legacy_real_partitions(n).iter().map(|&x| x as i32).collect();
    Ok(pars.into_pyarray(py))
}

/// Band layout of a scheme: parallel arrays (src_lo, width, fcentre, out_off).
/// Used by Python helpers to map packed coefficients onto a frequency-time grid.
#[pyfunction]
#[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false))]
fn scheme_bands<'py>(
    py: Python<'py>,
    n: usize,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let kind = window_kind(window_type)?;
    let s = build_scheme(scheme, n, kind, nyquist_flat_top)?;
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
#[pyo3(signature = (n, scheme="dyadic_dual_real", window_type="gaussian", nyquist_flat_top=false))]
fn output_len(
    n: usize,
    scheme: &str,
    window_type: &str,
    nyquist_flat_top: bool,
) -> PyResult<usize> {
    let kind = window_kind(window_type)?;
    Ok(build_scheme(scheme, n, kind, nyquist_flat_top)?.output_len)
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
    m.add_class::<PyGaussogram>()?;
    Ok(())
}
