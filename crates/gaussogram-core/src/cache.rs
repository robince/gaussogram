//! Process-wide cache of built engines, so one-shot entry points (the Python
//! free functions, the C ABI's `gaussogram_create`) do not rebuild windows and
//! FFT plans on every call. Engines are immutable once built and shared via
//! `Arc`; each caller still owns its own [`crate::Scratch`].

use std::sync::{Arc, Mutex};

use crate::engine::Gaussogram1d;
use crate::error::GaussogramError;
use crate::scheme::{dyadic_complex_with, dyadic_dual_real_with, dyadic_real_with, Scheme};
use crate::window::{TopWindow, WindowKind};

/// Number of engines kept. An engine holds roughly `40 * N` bytes of windows
/// and FFT twiddles (about 40 MB at N = 2^20), so keep this small.
pub const ENGINE_CACHE_CAPACITY: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SchemeKind {
    DyadicDualReal,
    DyadicReal,
    DyadicComplex,
}

/// Everything that determines a built engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SchemeSpec {
    pub kind: SchemeKind,
    pub n: usize,
    pub window: WindowKind,
    /// Only meaningful for `DyadicDualReal`.
    pub top_window: TopWindow,
    /// Ignored by `DyadicComplex`.
    pub bands_per_octave: usize,
}

impl SchemeSpec {
    pub fn build_scheme(&self) -> Result<Scheme, GaussogramError> {
        match self.kind {
            SchemeKind::DyadicDualReal => dyadic_dual_real_with(
                self.n,
                self.window,
                self.top_window,
                self.bands_per_octave,
            ),
            SchemeKind::DyadicReal => dyadic_real_with(self.n, self.window, self.bands_per_octave),
            SchemeKind::DyadicComplex => dyadic_complex_with(self.n, self.window),
        }
    }

    /// Fields a scheme ignores are normalised so they do not split cache entries.
    fn canonical(mut self) -> Self {
        if self.kind != SchemeKind::DyadicDualReal || self.window == WindowKind::Box {
            self.top_window = TopWindow::default();
        }
        if self.kind == SchemeKind::DyadicComplex {
            self.bands_per_octave = 1;
        }
        self
    }
}

/// Most-recently-used first.
type Entries = Vec<(SchemeSpec, Arc<Gaussogram1d>)>;

static CACHE: Mutex<Entries> = Mutex::new(Vec::new());

/// Return a shared engine for `spec`, building it on a miss.
///
/// The lock is not held while building, so a slow build does not block other
/// threads; two threads missing on the same spec may both build it, and the
/// first one inserted wins.
#[cfg(feature = "fft-rustfft")]
pub fn cached_engine(spec: SchemeSpec) -> Result<Arc<Gaussogram1d>, GaussogramError> {
    let spec = spec.canonical();
    if let Some(engine) = lookup(&spec) {
        return Ok(engine);
    }
    let engine = Arc::new(crate::build(spec.build_scheme()?));
    let mut entries = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = entries.iter().position(|(s, _)| *s == spec) {
        let entry = entries.remove(pos);
        let shared = entry.1.clone();
        entries.insert(0, entry);
        return Ok(shared);
    }
    entries.insert(0, (spec, engine.clone()));
    entries.truncate(ENGINE_CACHE_CAPACITY);
    Ok(engine)
}

/// Drop every cached engine (engines still held by callers stay alive). Used
/// by benchmarks to time cold construction.
pub fn clear_engine_cache() {
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

fn lookup(spec: &SchemeSpec) -> Option<Arc<Gaussogram1d>> {
    let mut entries = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let pos = entries.iter().position(|(s, _)| s == spec)?;
    let entry = entries.remove(pos);
    let engine = entry.1.clone();
    entries.insert(0, entry);
    Some(engine)
}
