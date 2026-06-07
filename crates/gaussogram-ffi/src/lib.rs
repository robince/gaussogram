//! C ABI shim for MATLAB / other C callers. No panics cross the boundary
//! (bodies are wrapped in `catch_unwind`); errors are integer status codes.

use std::os::raw::c_char;
use std::panic;
use std::ptr;
use std::slice;

use num_complex::Complex;

use gaussogram_core::{
    build, complex_partitions, dyadic_complex_with, dyadic_dual_real_with, dyadic_real_with,
    legacy_real_partitions, Gaussogram1d, WindowKind, STATUS_INTERNAL, STATUS_OK,
    STATUS_UNKNOWN_WINDOW,
};

pub use gaussogram_core::{
    STATUS_NOT_INVERTIBLE, STATUS_OUTPUT_LEN_MISMATCH, STATUS_POWER_OF_TWO, STATUS_TOO_SMALL,
    STATUS_WRONG_INPUT_KIND,
};

/// Opaque handle wrapping a constructed engine so callers build once and
/// transform many segments (plan reuse across calls).
pub struct GaussogramHandle {
    engine: Gaussogram1d,
}

fn scheme_from_id(
    scheme_id: i32,
    n: usize,
    kind: WindowKind,
    nyquist_flat_top: bool,
) -> Option<gaussogram_core::Scheme> {
    // The C ABI does not yet expose bands_per_octave; the one-band-per-octave
    // cover (bpo = 1) keeps the existing ABI and golden parity unchanged.
    match scheme_id {
        0 => dyadic_dual_real_with(n, kind, nyquist_flat_top, 1).ok(),
        1 => dyadic_real_with(n, kind, 1).ok(),
        2 => dyadic_complex_with(n, kind).ok(),
        _ => None,
    }
}

fn kind_from_id(window_id: i32) -> Option<WindowKind> {
    match window_id {
        0 => Some(WindowKind::Gaussian),
        1 => Some(WindowKind::Box),
        _ => None,
    }
}

/// Create an engine. `scheme_id`: 0=dyadic_dual_real, 1=dyadic_real,
/// 2=dyadic_complex. `window_id`: 0=gaussian, 1=box. On success writes the
/// handle pointer to `*out_handle` and returns STATUS_OK.
///
/// # Safety
/// `out_handle` must be a valid pointer to a `*mut GaussogramHandle`.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_create(
    scheme_id: i32,
    n: usize,
    window_id: i32,
    nyquist_flat_top: i32,
    out_handle: *mut *mut GaussogramHandle,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let Some(kind) = kind_from_id(window_id) else {
            return STATUS_UNKNOWN_WINDOW;
        };
        let Some(scheme) = scheme_from_id(scheme_id, n, kind, nyquist_flat_top != 0) else {
            return STATUS_POWER_OF_TWO; // closest generic construction failure
        };
        let engine = build(scheme);
        let boxed = Box::new(GaussogramHandle { engine });
        unsafe {
            *out_handle = Box::into_raw(boxed);
        }
        STATUS_OK
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Destroy a handle created by `gaussogram_create`.
///
/// # Safety
/// `handle` must have been returned by `gaussogram_create` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_destroy(handle: *mut GaussogramHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Query the packed output length (number of complex coefficients).
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_output_len(handle: *const GaussogramHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).engine.output_len() }
}

/// Forward transform of one real segment. `signal` has `n` doubles; `out` has
/// `2 * output_len` doubles (interleaved re,im).
///
/// # Safety
/// Pointers must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_forward_real(
    handle: *const GaussogramHandle,
    signal: *const f64,
    signal_len: usize,
    out: *mut f64,
    out_len_complex: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if handle.is_null() || signal.is_null() || out.is_null() {
            return STATUS_INTERNAL;
        }
        let engine = unsafe { &(*handle).engine };
        let sig = unsafe { slice::from_raw_parts(signal, signal_len) };
        let out_complex =
            unsafe { slice::from_raw_parts_mut(out as *mut Complex<f64>, out_len_complex) };
        match engine.forward(sig, out_complex) {
            Ok(()) => STATUS_OK,
            Err(e) => e.status_code(),
        }
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Batch forward over `count` real segments laid out contiguously
/// (`count * n` doubles in, `count * 2 * output_len` doubles out).
///
/// # Safety
/// Pointers must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_forward_real_batch(
    handle: *const GaussogramHandle,
    signals: *const f64,
    n: usize,
    count: usize,
    out: *mut f64,
    out_len_complex: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if handle.is_null() || signals.is_null() || out.is_null() {
            return STATUS_INTERNAL;
        }
        let engine = unsafe { &(*handle).engine };
        let flat = unsafe { slice::from_raw_parts(signals, n * count) };
        let segs: Vec<&[f64]> = (0..count).map(|i| &flat[i * n..(i + 1) * n]).collect();
        let out_complex =
            unsafe { slice::from_raw_parts_mut(out as *mut Complex<f64>, out_len_complex) };
        match engine.forward_batch(&segs, out_complex) {
            Ok(()) => STATUS_OK,
            Err(e) => e.status_code(),
        }
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Write the legacy real-partition boundaries into `out` (caller provides a
/// buffer of at least `2*log2(N/2)+1` ints). Returns the count written, or -1.
///
/// # Safety
/// `out` must be valid for `out_cap` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_real_partitions(
    n: usize,
    out: *mut i32,
    out_cap: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let pars = legacy_real_partitions(n);
        if out.is_null() || pars.len() > out_cap {
            return -1;
        }
        let dst = unsafe { slice::from_raw_parts_mut(out, pars.len()) };
        for (d, &p) in dst.iter_mut().zip(pars.iter()) {
            *d = p as i32;
        }
        pars.len() as i32
    }));
    result.unwrap_or(-1)
}

/// Write the complex-partition boundaries. See `gaussogram_real_partitions`.
///
/// # Safety
/// `out` must be valid for `out_cap` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_complex_partitions(
    n: usize,
    out: *mut i32,
    out_cap: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let pars = complex_partitions(n);
        if out.is_null() || pars.len() > out_cap {
            return -1;
        }
        let dst = unsafe { slice::from_raw_parts_mut(out, pars.len()) };
        for (d, &p) in dst.iter_mut().zip(pars.iter()) {
            *d = p as i32;
        }
        pars.len() as i32
    }));
    result.unwrap_or(-1)
}

/// Inverse transform for invertible real schemes (currently `dyadic_real`).
/// `coeffs` has `2 * coeffs_len_complex` doubles (interleaved re,im); `out` has
/// `out_len` doubles (the reconstructed real signal, length N).
///
/// # Safety
/// Pointers must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_inverse_real(
    handle: *const GaussogramHandle,
    coeffs: *const f64,
    coeffs_len_complex: usize,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if handle.is_null() || coeffs.is_null() || out.is_null() {
            return STATUS_INTERNAL;
        }
        let engine = unsafe { &(*handle).engine };
        let c = unsafe { slice::from_raw_parts(coeffs as *const Complex<f64>, coeffs_len_complex) };
        let out_real = unsafe { slice::from_raw_parts_mut(out, out_len) };
        match engine.inverse(c, out_real) {
            Ok(()) => STATUS_OK,
            Err(e) => e.status_code(),
        }
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Number of bands in the engine's scheme (rows to fill when building a
/// time-frequency display grid). Returns -1 on a null handle.
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_num_bands(handle: *const GaussogramHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    unsafe { (*handle).engine.scheme().bands.len() as i32 }
}

/// Write the per-band layout into four caller-provided `i64` buffers
/// (`src_lo`, `width`, `fcentre`, `out_off`), each of capacity `cap`. Returns
/// the number of bands written, or -1 on error. Mirrors the Python
/// `scheme_bands` helper; used to map packed coefficients onto a freq-time grid.
///
/// # Safety
/// All four output pointers must be valid for `cap` `i64`s.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_scheme_bands(
    handle: *const GaussogramHandle,
    out_lo: *mut i64,
    out_width: *mut i64,
    out_fcentre: *mut i64,
    out_off: *mut i64,
    cap: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if handle.is_null()
            || out_lo.is_null()
            || out_width.is_null()
            || out_fcentre.is_null()
            || out_off.is_null()
        {
            return -1;
        }
        let scheme = unsafe { (*handle).engine.scheme() };
        let nb = scheme.bands.len();
        if nb > cap {
            return -1;
        }
        let lo = unsafe { slice::from_raw_parts_mut(out_lo, nb) };
        let width = unsafe { slice::from_raw_parts_mut(out_width, nb) };
        let fcentre = unsafe { slice::from_raw_parts_mut(out_fcentre, nb) };
        let off = unsafe { slice::from_raw_parts_mut(out_off, nb) };
        for (i, (b, w)) in scheme.bands.iter().zip(scheme.windows.iter()).enumerate() {
            lo[i] = b.src_lo as i64;
            width[i] = b.width() as i64;
            fcentre[i] = w.fcentre as i64;
            off[i] = b.out_off as i64;
        }
        nb as i32
    }));
    result.unwrap_or(-1)
}

/// Reserved for a future name-based constructor; currently unused.
///
/// # Safety
/// `_name` must be a valid C string if non-null.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_unused_name_marker(_name: *const c_char) -> *const c_char {
    ptr::null()
}
