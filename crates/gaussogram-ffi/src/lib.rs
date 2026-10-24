//! C ABI shim for MATLAB / other C callers. No panics cross the boundary
//! (bodies are wrapped in `catch_unwind`); errors are integer status codes.

use std::panic;
use std::slice;
use std::sync::{Arc, Mutex};

use num_complex::Complex;

use gaussogram_core::{
    cached_engine, complex_partitions, legacy_real_partitions, Gaussogram1d, GaussogramError,
    SchemeKind, SchemeSpec, Scratch, WindowKind, STATUS_INTERNAL, STATUS_OK, STATUS_UNKNOWN_SCHEME,
    STATUS_UNKNOWN_WINDOW, STATUS_UNSUPPORTED_OPTION,
};

pub use gaussogram_core::{
    STATUS_BANDS_PER_OCTAVE, STATUS_INPUT_LEN_MISMATCH, STATUS_NOT_INVERTIBLE,
    STATUS_OUTPUT_LEN_MISMATCH, STATUS_POWER_OF_TWO, STATUS_TOO_SMALL, STATUS_WRONG_INPUT_KIND,
};

/// Opaque handle wrapping a constructed engine so callers build once and
/// transform many segments (plan reuse across calls). Engines are immutable and
/// come from the core's process-wide cache, so creating a handle for a
/// recently used configuration (e.g. the MEX one-shot path) is cheap.
pub struct GaussogramHandle {
    engine: Arc<Gaussogram1d>,
    /// Reused across single-segment calls so a persistent handle does not
    /// allocate per transform. Behind a mutex because the C API takes
    /// `const GaussogramHandle*`; if a caller does use one handle from several
    /// threads at once, the losers fall back to a temporary scratch.
    scratch: Mutex<Scratch>,
}

impl GaussogramHandle {
    fn with_scratch<R>(&self, f: impl FnOnce(&Gaussogram1d, &mut Scratch) -> R) -> R {
        match self.scratch.try_lock() {
            Ok(mut scratch) => f(&self.engine, &mut scratch),
            Err(_) => f(&self.engine, &mut self.engine.alloc_scratch()),
        }
    }
}

fn engine_from_id(
    scheme_id: i32,
    n: usize,
    window: WindowKind,
    nyquist_flat_top: bool,
    bands_per_octave: usize,
) -> Result<Arc<Gaussogram1d>, i32> {
    let kind = match scheme_id {
        0 => SchemeKind::DyadicDualReal,
        1 => SchemeKind::DyadicReal,
        2 => SchemeKind::DyadicComplex,
        _ => return Err(STATUS_UNKNOWN_SCHEME),
    };
    // Reject options the scheme does not support instead of silently
    // ignoring them.
    if nyquist_flat_top && kind != SchemeKind::DyadicDualReal {
        return Err(STATUS_UNSUPPORTED_OPTION);
    }
    if bands_per_octave != 1 && kind == SchemeKind::DyadicComplex {
        return Err(STATUS_UNSUPPORTED_OPTION);
    }
    // The core validates bands_per_octave (a power of two >= 1).
    cached_engine(SchemeSpec {
        kind,
        n,
        window,
        nyquist_flat_top,
        bands_per_octave,
    })
    .map_err(|e| e.status_code())
}

fn kind_from_id(window_id: i32) -> Option<WindowKind> {
    match window_id {
        0 => Some(WindowKind::Gaussian),
        1 => Some(WindowKind::Box),
        _ => None,
    }
}

/// Create an engine with one band per octave. `scheme_id`: 0=dyadic_dual_real,
/// 1=dyadic_real, 2=dyadic_complex. `window_id`: 0=gaussian, 1=box. On
/// success writes the handle pointer to `*out_handle` and returns STATUS_OK.
/// Equivalent to `gaussogram_create_ex(..., bands_per_octave = 1, ...)`.
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
    unsafe { gaussogram_create_ex(scheme_id, n, window_id, nyquist_flat_top, 1, out_handle) }
}

/// Create an engine with `bands_per_octave` sub-bands per dyadic octave (a
/// power of two >= 1; `1` is the `gaussogram_create` cover). Both real schemes
/// accept it and keep their output length; `dyadic_complex` accepts only `1`
/// (`STATUS_UNSUPPORTED_OPTION` otherwise). An invalid value returns
/// `STATUS_BANDS_PER_OCTAVE`. Other arguments as for `gaussogram_create`.
///
/// # Safety
/// `out_handle` must be a valid pointer to a `*mut GaussogramHandle`.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_create_ex(
    scheme_id: i32,
    n: usize,
    window_id: i32,
    nyquist_flat_top: i32,
    bands_per_octave: usize,
    out_handle: *mut *mut GaussogramHandle,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if out_handle.is_null() {
            return STATUS_INTERNAL;
        }
        let Some(kind) = kind_from_id(window_id) else {
            return STATUS_UNKNOWN_WINDOW;
        };
        let engine =
            match engine_from_id(scheme_id, n, kind, nyquist_flat_top != 0, bands_per_octave) {
                Ok(e) => e,
                Err(status) => return status,
            };
        let scratch = Mutex::new(engine.alloc_scratch());
        let boxed = Box::new(GaussogramHandle { engine, scratch });
        unsafe {
            *out_handle = Box::into_raw(boxed);
        }
        STATUS_OK
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Destroy a handle created by `gaussogram_create` or `gaussogram_create_ex`.
///
/// # Safety
/// `handle` must have been returned by `gaussogram_create` /
/// `gaussogram_create_ex` and not yet freed.
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
        let handle = unsafe { &*handle };
        let sig = unsafe { slice::from_raw_parts(signal, signal_len) };
        let out_complex =
            unsafe { slice::from_raw_parts_mut(out as *mut Complex<f64>, out_len_complex) };
        match handle.with_scratch(|engine, scratch| engine.forward_with(sig, out_complex, scratch)) {
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
        let Some(total) = n.checked_mul(count) else {
            return STATUS_INTERNAL;
        };
        let flat = unsafe { slice::from_raw_parts(signals, total) };
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

/// Forward transform of one complex segment (`dyadic_complex` handles only;
/// a real-input handle returns `STATUS_WRONG_INPUT_KIND`). `signal` has
/// `2 * signal_len_complex` doubles and `out` has `2 * out_len_complex`
/// doubles, both interleaved (re,im).
///
/// # Safety
/// Pointers must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_forward_complex(
    handle: *const GaussogramHandle,
    signal: *const f64,
    signal_len_complex: usize,
    out: *mut f64,
    out_len_complex: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if handle.is_null() || signal.is_null() || out.is_null() {
            return STATUS_INTERNAL;
        }
        let handle = unsafe { &*handle };
        let sig =
            unsafe { slice::from_raw_parts(signal as *const Complex<f64>, signal_len_complex) };
        let out_complex =
            unsafe { slice::from_raw_parts_mut(out as *mut Complex<f64>, out_len_complex) };
        match handle
            .with_scratch(|engine, scratch| engine.forward_complex_with(sig, out_complex, scratch))
        {
            Ok(()) => STATUS_OK,
            Err(e) => e.status_code(),
        }
    }));
    result.unwrap_or(STATUS_INTERNAL)
}

/// Copy partition boundaries into the caller's buffer. Returns the count
/// written, or a negated status code: the core's validation error, or
/// `-STATUS_OUTPUT_LEN_MISMATCH` for a null or too-small buffer.
unsafe fn write_partitions(
    pars: Result<Vec<usize>, GaussogramError>,
    out: *mut i32,
    out_cap: usize,
) -> i32 {
    let pars = match pars {
        Ok(p) => p,
        Err(e) => return -e.status_code(),
    };
    if out.is_null() || pars.len() > out_cap {
        return -STATUS_OUTPUT_LEN_MISMATCH;
    }
    let dst = unsafe { slice::from_raw_parts_mut(out, pars.len()) };
    for (d, &p) in dst.iter_mut().zip(pars.iter()) {
        *d = p as i32;
    }
    pars.len() as i32
}

/// Write the legacy real-partition boundaries into `out` (caller provides a
/// buffer of at least `2*log2(N/2)+1` ints). Returns the count written, or a
/// negated status code (`-STATUS_POWER_OF_TWO`, `-STATUS_TOO_SMALL` for
/// `n < 4`, `-STATUS_OUTPUT_LEN_MISMATCH` for a null or too-small buffer).
///
/// # Safety
/// `out` must be valid for `out_cap` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_real_partitions(
    n: usize,
    out: *mut i32,
    out_cap: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
        write_partitions(legacy_real_partitions(n), out, out_cap)
    }));
    result.unwrap_or(-STATUS_INTERNAL)
}

/// Write the complex-partition boundaries (`n` a power of two >= 8). See
/// `gaussogram_real_partitions`.
///
/// # Safety
/// `out` must be valid for `out_cap` `i32`s.
#[no_mangle]
pub unsafe extern "C" fn gaussogram_complex_partitions(
    n: usize,
    out: *mut i32,
    out_cap: usize,
) -> i32 {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
        write_partitions(complex_partitions(n), out, out_cap)
    }));
    result.unwrap_or(-STATUS_INTERNAL)
}

/// Inverse transform for the real schemes: exact for `dyadic_real`, the
/// least-squares frame inverse for `dyadic_dual_real`.
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
        let handle = unsafe { &*handle };
        let c = unsafe { slice::from_raw_parts(coeffs as *const Complex<f64>, coeffs_len_complex) };
        let out_real = unsafe { slice::from_raw_parts_mut(out, out_len) };
        match handle.with_scratch(|engine, scratch| engine.inverse_with(c, out_real, scratch)) {
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
