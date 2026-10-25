//! Partitioning schemes as *data*. A `Scheme` is a list of `Band`s plus a
//! parallel list of `Window`s; the transform engine walks them without any
//! per-scheme branching.

use crate::error::GaussogramError;
use crate::window::{
    box_band_screen, complex_global_screen, gaussian_sigma_screen, wide_top_sigma, GaussianScreens,
    TopWindow, WindowKind,
};

/// A band reads the contiguous spectrum slice `[src_lo, src_hi)` and writes its
/// `width` complex outputs to `[out_off, out_off + width)` of the packed buffer.
/// `src` and `out` are tracked separately because overlapping tilings (A and B
/// in `dyadic_dual_real`) reuse the same source bins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Band {
    pub src_lo: usize,
    pub src_hi: usize,
    pub out_off: usize,
}

impl Band {
    pub fn width(&self) -> usize {
        self.src_hi - self.src_lo
    }
}

/// Per-band frequency-domain window, applied against a private copy of the
/// band's source slice.
#[derive(Clone, Debug)]
pub struct Window {
    pub kind: WindowKind,
    pub fcentre: usize,
    /// §3.2: the top-window shape, set only on tiling A's top band in
    /// `dyadic_dual_real` (the band just below the width-1 Nyquist band);
    /// `None` on every other band. Gaussian windows only: a box window ignores
    /// it.
    pub top_window: Option<TopWindow>,
    /// Precomputed frequency-domain taper for this band, length = band width.
    pub screen: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Scheme {
    pub name: &'static str,
    pub n: usize,
    pub complex_input: bool,
    pub bands: Vec<Band>,
    pub windows: Vec<Window>,
    pub output_len: usize,
    pub invertible: bool,
    /// True when two or more bands read overlapping source ranges (the dual
    /// scheme). When false, a single global-screen multiply is valid.
    pub overlapping_src: bool,
}

fn check_pow2(n: usize) -> Result<(), GaussogramError> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(GaussogramError::PowerOfTwo(n));
    }
    Ok(())
}

/// `bands_per_octave` must be a power of two >= 1 so each dyadic octave (a
/// power-of-two number of bins) divides into equal integer sub-bands.
fn check_bpo(bpo: usize) -> Result<(), GaussogramError> {
    if bpo == 0 || (bpo & (bpo - 1)) != 0 {
        return Err(GaussogramError::BandsPerOctave(bpo));
    }
    Ok(())
}

/// Split `[lo, hi)` into `min(bpo, width)` equal sub-bands. The octave `width`
/// and `bpo` are both powers of two, so when `bpo <= width` the division is
/// exact; the `min` caps the count for low octaves that have fewer bins than
/// `bpo` (e.g. the width-1 cover `[1, 2)` always stays a single band).
fn split_pow2(lo: usize, hi: usize, bpo: usize) -> Vec<(usize, usize)> {
    let width = hi - lo;
    let k = bpo.min(width).max(1);
    let sub = width / k;
    (0..k).map(|i| (lo + i * sub, lo + (i + 1) * sub)).collect()
}

fn make_window(kind: WindowKind, screens: &mut GaussianScreens, lo: usize, width: usize) -> Window {
    let fcentre = lo + width / 2;
    let screen = match kind {
        WindowKind::Gaussian => screens.band_screen(fcentre as i64, width),
        WindowKind::Box => box_band_screen(width),
    };
    Window {
        kind,
        fcentre,
        top_window: None,
        screen,
    }
}

/// Tiling A: lossless dyadic cover of the positive half-spectrum `0..=N/2`
/// (octaves `[0,1), [1,2), [2,4), …, [N/4, N/2)`, then the Nyquist bin
/// `[N/2, N/2+1)` as its own width-1 band).
/// Each dyadic octave is subdivided into `min(bands_per_octave, octave_width)`
/// equal sub-bands (`bpo = 1` reproduces the single-band-per-octave cover).
/// Returns the bands (out_off filled sequentially from 0).
fn tiling_a(
    n: usize,
    kind: WindowKind,
    bpo: usize,
    screens: &mut GaussianScreens,
) -> (Vec<Band>, Vec<Window>) {
    let mut bands = Vec::new();
    let mut windows = Vec::new();
    let mut out_off = 0usize;
    let push = |lo: usize,
                hi: usize,
                bands: &mut Vec<Band>,
                windows: &mut Vec<Window>,
                out_off: &mut usize,
                screens: &mut GaussianScreens| {
        windows.push(make_window(kind, screens, lo, hi - lo));
        bands.push(Band {
            src_lo: lo,
            src_hi: hi,
            out_off: *out_off,
        });
        *out_off += hi - lo;
    };

    // Octave covers [prev, edge) with right edges 1, 2, 4, ..., N/4, each split
    // into power-of-two sub-bands.
    let mut prev = 0usize;
    let mut edge = 1usize;
    while edge <= n / 4 {
        for (lo, hi) in split_pow2(prev, edge, bpo) {
            push(lo, hi, &mut bands, &mut windows, &mut out_off, screens);
        }
        prev = edge;
        edge *= 2;
    }
    // Top octave [N/4, N/2), subdivided like the others.
    for (lo, hi) in split_pow2(prev, n / 2, bpo) {
        push(lo, hi, &mut bands, &mut windows, &mut out_off, screens);
    }
    // The Nyquist bin is its own width-1 band (untapered, like every width-1
    // band). Folding it into the top band would make that band, the widest
    // and most expensive one, an odd length (N/4 + 1, often prime) and its
    // inverse FFT several times slower than a power of two.
    push(n / 2, n / 2 + 1, &mut bands, &mut windows, &mut out_off, screens);
    (bands, windows)
}

/// `dyadic_real` — baseline lossless single tiling (tiling A). Invertible.
pub fn dyadic_real(n: usize) -> Result<Scheme, GaussogramError> {
    dyadic_real_with(n, WindowKind::Gaussian, 1)
}

/// `dyadic_real` with explicit window and `bands_per_octave` (power of two).
/// Subdividing octaves keeps the cover a disjoint integer partition, so the
/// scheme stays invertible and its output length is `N/2 + 1` for every `bpo` —
/// it only rotates the time/frequency trade-off (finer frequency, coarser time).
pub fn dyadic_real_with(
    n: usize,
    kind: WindowKind,
    bands_per_octave: usize,
) -> Result<Scheme, GaussogramError> {
    check_pow2(n)?;
    check_bpo(bands_per_octave)?;
    if n < 4 {
        return Err(GaussogramError::TooSmall {
            scheme: "dyadic_real",
            min: 4,
            got: n,
        });
    }
    let (bands, windows) = tiling_a(n, kind, bands_per_octave, &mut GaussianScreens::new(n));
    let output_len = bands.iter().map(Band::width).sum();
    Ok(Scheme {
        name: "dyadic_real",
        n,
        complex_input: false,
        bands,
        windows,
        output_len,
        invertible: true,
        overlapping_src: false,
    })
}

/// `dyadic_dual_real` — the DEFAULT scheme. Tiling A followed by tiling B.
/// A ~2x redundant frame, invertible by the engine's least-squares frame
/// inverse (least squares in signal energy).
pub fn dyadic_dual_real(n: usize) -> Result<Scheme, GaussogramError> {
    dyadic_dual_real_with(n, WindowKind::Gaussian, TopWindow::Wide, 1)
}

/// `dyadic_dual_real` with explicit options. `bands_per_octave` (power of two)
/// subdivides **both** tiling A and the offset tiling-B dead-zone cover in
/// lockstep, so the dual scheme stays ~2x redundant (output length `N - 1`) for
/// every `bpo`.
///
/// `top_window` shapes tiling A's top band (width `w = N / (4 bpo)`), whose
/// upper half no tiling-B band covers (one would need bins above `N/2`):
///
/// * [`TopWindow::Wide`] (default): the same peak at local index `w/2`, with
///   `sigma = max(fcentre / (2 pi), wide_top_sigma(w))`, so the gain at both
///   band edges is at least [`crate::window::TOP_EDGE_GAIN`]. This lifts the
///   frame's lower bound to the floor of the rest of the frame (condition
///   number ~3.5 instead of ~84 at `bpo = 1`) and, being symmetric, leaves no
///   step where the band's FFT wraps round. At `bpo >= 2` the plain Gaussian
///   is already the wider one, so the screen is unchanged.
/// * [`TopWindow::Gaussian`]: the plain Gaussian every other band uses.
/// * [`TopWindow::Flat`]: the plain Gaussian held at its peak value from the
///   peak to Nyquist. Conditions the frame too, but the step at the wrap
///   makes the band's time response ring.
///
/// Box windows ignore `top_window`.
pub fn dyadic_dual_real_with(
    n: usize,
    kind: WindowKind,
    top_window: TopWindow,
    bands_per_octave: usize,
) -> Result<Scheme, GaussogramError> {
    check_pow2(n)?;
    check_bpo(bands_per_octave)?;
    if n < 8 {
        return Err(GaussogramError::TooSmall {
            scheme: "dyadic_dual_real",
            min: 8,
            got: n,
        });
    }
    let mut screens = GaussianScreens::new(n);
    let (mut bands, mut windows) = tiling_a(n, kind, bands_per_octave, &mut screens);

    // Shape tiling A's top band: the band just below the width-1 Nyquist band
    // that ends tiling A.
    if kind == WindowKind::Gaussian {
        let top_idx = bands.len() - 2;
        let top = &mut windows[top_idx];
        let width = bands[top_idx].width();
        match top_window {
            TopWindow::Gaussian => {}
            TopWindow::Wide => {
                // Never narrower than the plain Gaussian; when that is already
                // the wider one, keep its screen bit for bit.
                let sigma_default = top.fcentre as f64 / (2.0 * std::f64::consts::PI);
                let sigma_wide = wide_top_sigma(width);
                if sigma_wide > sigma_default {
                    top.screen = gaussian_sigma_screen(n, sigma_wide, width);
                }
            }
            TopWindow::Flat => {
                let peak = top.fcentre - bands[top_idx].src_lo; // local index of the peak
                let peak_val = top.screen[peak];
                for v in top.screen[peak..].iter_mut() {
                    *v = peak_val;
                }
            }
        }
        top.top_window = Some(top_window);
    }

    // Tiling B: the dead-zone cover, offset by HALF A SUB-BAND from tiling A so
    // each B band is centred on a tiling-A *join* (the weak point), never on an
    // A sub-band. For octave [b, 2b) (b in {2,4,...,N/4}) with `k` sub-bands of
    // width `s = b/k`, band m spans [b + m*s - s/2, b + m*s + s/2) — i.e. A's
    // sub-band grid shifted down by s/2 — so its centre falls on join b + m*s.
    // `k = min(bpo, b/2)` keeps the half-sub-band shift s/2 an integer (it needs
    // 2k | b). At bpo=1 this is the single band [b/2, 3b/2) centred on join b.
    let mut out_off: usize = bands.iter().map(Band::width).sum();
    let mut b = 2usize;
    while b <= n / 4 {
        let k = bands_per_octave.min(b / 2).max(1);
        let s = b / k;
        for m in 0..k {
            let centre = b + m * s;
            let lo = centre - s / 2;
            let hi = centre + s / 2;
            let w = make_window(kind, &mut screens, lo, hi - lo); // fcentre = lo + s/2 = centre
            windows.push(w);
            bands.push(Band {
                src_lo: lo,
                src_hi: hi,
                out_off,
            });
            out_off += hi - lo;
        }
        b *= 2;
    }

    let output_len = bands.iter().map(Band::width).sum();
    Ok(Scheme {
        name: "dyadic_dual_real",
        n,
        complex_input: false,
        bands,
        windows,
        output_len,
        invertible: true,
        overlapping_src: true,
    })
}

/// `dyadic_complex` — faithful port of the symmetric complex dyadic scheme
/// (`gft_1dPartitions` + `windows`). Output length N, complex input.
pub fn dyadic_complex(n: usize) -> Result<Scheme, GaussogramError> {
    dyadic_complex_with(n, WindowKind::Gaussian)
}

pub fn dyadic_complex_with(n: usize, kind: WindowKind) -> Result<Scheme, GaussogramError> {
    check_pow2(n)?;
    // The ported `gft_1dPartitions` algorithm degenerates below 8 (it produces
    // unfilled partition slots for N=4), matching the legacy Cython contract
    // that rejects complex sizes < 8.
    if n < 8 {
        return Err(GaussogramError::TooSmall {
            scheme: "dyadic_complex",
            min: 8,
            got: n,
        });
    }
    let boundaries = complex_partitions(n)?;
    let global = complex_global_screen(n, kind);

    let mut bands = Vec::new();
    let mut windows = Vec::new();
    let mut prev = 0usize;
    let mut out_off = 0usize;
    for &hi in &boundaries {
        let lo = prev;
        let width = hi - lo;
        let screen = global[lo..hi].to_vec();
        // fcentre is informational for the complex scheme (screen comes from the
        // global window); record the positive-band centre.
        let fcentre = if lo < n / 2 {
            lo + width / 2
        } else {
            ((n as i64 - lo as i64).unsigned_abs() as usize).saturating_sub(width / 2)
        };
        windows.push(Window {
            kind,
            fcentre,
            top_window: None,
            screen,
        });
        bands.push(Band {
            src_lo: lo,
            src_hi: hi,
            out_off,
        });
        out_off += width;
        prev = hi;
    }

    let output_len = bands.iter().map(Band::width).sum();
    Ok(Scheme {
        name: "dyadic_complex",
        n,
        complex_input: true,
        bands,
        windows,
        output_len,
        invertible: false,
        overlapping_src: false,
    })
}

/// Legacy real-partition boundaries, faithful port of `gft_1dRealPartitions`.
/// Kept for the `scheme="legacy"` parity path and `real_partitions()`.
/// `real_partitions(8) == [1,2,4,6,8]`, `real_partitions(16) == [1,2,4,8,11,14,16]`.
/// Requires a power-of-two `n >= 4`; smaller or non-dyadic `n` would produce
/// corrupt boundaries (or panic) in the faithful port.
pub fn legacy_real_partitions(n: usize) -> Result<Vec<usize>, GaussogramError> {
    check_pow2(n)?;
    if n < 4 {
        return Err(GaussogramError::TooSmall {
            scheme: "real partitions",
            min: 4,
            got: n,
        });
    }
    let npar = (n / 2).ilog2() as usize; // round(log2(N/2)) for power-of-two N
    let size = 2 * npar + 1; // boundary count (excluding sentinel)
    let mut p = vec![0i64; size + 1];
    p[0] = 1; // DC
    p[size - 1] = n as i64; // last value (index 2*Npar in C is size-1 here)

    // positive freq partitions
    let mut width = 1i64;
    let mut i = 1usize;
    for _ in 1..=npar {
        p[i] = p[i - 1] + width;
        width *= 2;
        i += 1;
    }
    // negative freq partitions
    let mut i = (2 * npar) as i64 - 1; // index into the size-2*Npar+1 layout
    for pi in 1..=npar {
        let w = 2i64.pow((pi - 1) as u32) + 1;
        let newpar = p[(i + 1) as usize] - w;
        if p[i as usize] > 0 {
            break;
        } else {
            p[i as usize] = newpar;
        }
        i -= 1;
    }

    Ok(p[..size].iter().map(|&x| x as usize).collect())
}

/// Complex-partition boundaries, faithful port of `gft_1dPartitions`.
/// `complex_partitions(8) == [1,2,4,6,7,8]`. Requires a power-of-two `n >= 8`
/// (the same minimum as `dyadic_complex`); smaller `n` gives corrupt boundaries.
pub fn complex_partitions(n: usize) -> Result<Vec<usize>, GaussogramError> {
    check_pow2(n)?;
    if n < 8 {
        return Err(GaussogramError::TooSmall {
            scheme: "complex partitions",
            min: 8,
            got: n,
        });
    }
    let nn = n as i64;
    let logn = (n as f64).log2().round() as i64;
    let size = (logn * 2 + 1) as usize;
    let mut partitions = vec![0i64; size];
    let p_off = (logn * 2 - 1) as usize;

    let mut sf = 1i64;
    let mut cf = 1i64;
    let mut width = 1i64;
    let mut pcount = 0usize;

    while sf < nn / 2 {
        let mut ep = cf + width / 2 - 1;
        let mut sn = nn - cf - width / 2 + 1;
        let en = nn - cf + width / 2 + 1;
        if ep > nn {
            ep = nn;
        }
        if sn < 0 {
            sn = 0;
        }
        if width / 2 == 0 {
            ep += 1;
            sn -= 1;
        }
        let _ = sn;
        partitions[pcount] = ep;
        partitions[p_off - pcount] = en;
        pcount += 1;

        sf += width;
        if sf > 2 {
            width *= 2;
        }
        cf = sf + width / 2;
    }

    // The C terminates with a -1 sentinel at p_off+1; boundaries are [0..=p_off].
    Ok(partitions[..=p_off].iter().map(|&x| x as usize).collect())
}
