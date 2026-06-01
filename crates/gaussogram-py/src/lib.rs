//! PyO3 extension module `_native`, wrapped by the thin `python/gaussogram`
//! package. Accepts/returns NumPy arrays; all heavy lifting is in
//! `gaussogram-core`.

use num_complex::Complex;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use gaussogram_core::{
    build, complex_partitions as core_complex_partitions, dyadic_complex_with,
    dyadic_dual_real_with, dyadic_real_with, legacy_real_partitions, GaussogramError, Scheme,
    WindowKind,
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

/// Forward GFT of a real signal. Default scheme `dyadic_dual_real` (output N-1).
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
    let signal = x.as_slice()?;
    let n = signal.len();
    let s = build_scheme(scheme, n, kind, nyquist_flat_top)?;
    if s.complex_input {
        return Err(PyValueError::new_err(
            "gft1d_real requires a real-input scheme; use gft1d for dyadic_complex",
        ));
    }
    let engine = build(s);
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    py.allow_threads(|| engine.forward(signal, &mut out))
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
    let s = build_scheme("dyadic_complex", n, kind, false)?;
    let engine = build(s);
    let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
    py.allow_threads(|| engine.forward_complex(signal, &mut out))
        .map_err(map_err)?;
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
    let c = coeffs.as_slice()?;
    // Recover N from output_len = N/2 + 1.
    let n = (c.len() - 1) * 2;
    let s = build_scheme("dyadic_real", n, kind, false)?;
    let engine = build(s);
    let mut out = vec![0.0f64; n];
    py.allow_threads(|| engine.inverse(c, &mut out))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
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
    Ok(())
}
