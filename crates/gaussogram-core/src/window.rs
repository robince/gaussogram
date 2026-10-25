//! Window-screen construction, reproducing the numerics of `gaussian`,
//! `windows`, and `windowsFromPars` in the reference `gft.c`.
//!
//! Each band's window is a frequency-domain taper, stored as a real `Vec<f64>`.
//! The frequency-domain Gaussian is real because the underlying time-domain
//! Gaussian (after the half-length circular shift that centres its peak on
//! sample 0) is real and even, so its DFT is real and even.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowKind {
    Gaussian,
    Box,
}

impl WindowKind {
    pub fn from_name(name: &str) -> Option<WindowKind> {
        match name {
            "gaussian" => Some(WindowKind::Gaussian),
            "box" => Some(WindowKind::Box),
            _ => None,
        }
    }
}

/// Shape of tiling A's top band window in `dyadic_dual_real` (the band just
/// below the width-1 Nyquist band). Tiling B has no band centred on Nyquist,
/// so the upper half of that band is covered by it alone; a plain Gaussian
/// leaves a dead zone there that sets the frame's condition number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TopWindow {
    /// Default. A Gaussian with the usual peak, widened so its gain at the
    /// band edges is [`TOP_EDGE_GAIN`]; never narrower than the plain one.
    #[default]
    Wide,
    /// The plain Gaussian every other band uses.
    Gaussian,
    /// The plain Gaussian, held at its peak value from the peak to Nyquist.
    Flat,
}

impl TopWindow {
    pub fn from_name(name: &str) -> Option<TopWindow> {
        match name {
            "wide" => Some(TopWindow::Wide),
            "gaussian" => Some(TopWindow::Gaussian),
            "flat" => Some(TopWindow::Flat),
            _ => None,
        }
    }
}

/// Gain of the wide top-band Gaussian at the band edges, half a band from its
/// peak. About `sqrt(0.40)`: it lifts the top band's edge to the floor of the
/// rest of the dual frame (`min sum_b W_b^2 ~ 0.40` at one band per octave).
pub const TOP_EDGE_GAIN: f64 = 0.63;

/// Standard deviation in bins of the wide top-band Gaussian for a band of
/// `width` bins: the gain at distance `width/2` from the peak is
/// [`TOP_EDGE_GAIN`].
pub fn wide_top_sigma(width: usize) -> f64 {
    (width as f64 / 2.0) / (2.0 * (1.0 / TOP_EDGE_GAIN).ln()).sqrt()
}

const SQRT_2PI: f64 = 2.506_628_274_631_000_7; // sqrt(2*PI)

/// `exp(x)` is exactly `0.0` in f64 for `x` below about -745.13; samples whose
/// exponent is below this bound are skipped (they would contribute exact zeros).
const EXP_UNDERFLOW: f64 = -746.0;

/// Smallest `|fcentre|` for which [`GaussianScreens`] evaluates the screen in
/// closed form instead of by DFT. The time-domain Gaussian has standard
/// deviation `N/fc` samples and is truncated at `±N/2`, i.e. at `fc/2` standard
/// deviations; the neglected tail is `~exp(-fc^2/8)` relative (`< 1e-31` here),
/// far below f64 rounding, so the two paths agree to rounding error.
const CLOSED_FORM_MIN_FC: f64 = 24.0;

/// Builds frequency-domain Gaussian screens for one transform length `n`,
/// reusing a single FFT plan and its buffers across bands (scheme construction
/// computes one length-`n` Gaussian per band, so this dominates setup time).
pub(crate) struct GaussianScreens {
    n: usize,
    time: Vec<f64>,
    /// `F[m]` for `m = 0..=n/2`; `F` is real and even, so this is all of it.
    half: Vec<f64>,
    #[cfg(feature = "fft-rustfft")]
    fft: Option<RealDft>,
}

#[cfg(feature = "fft-rustfft")]
struct RealDft {
    plan: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    spectrum: Vec<num_complex::Complex<f64>>,
    scratch: Vec<num_complex::Complex<f64>>,
}

impl GaussianScreens {
    pub(crate) fn new(n: usize) -> Self {
        GaussianScreens {
            n,
            time: Vec::new(),
            half: Vec::new(),
            #[cfg(feature = "fft-rustfft")]
            fft: None,
        }
    }

    /// Frequency-domain Gaussian `F[m]`, `m = 0..=n/2`, matching the real part of
    /// `fft(shift(gaussian(N, fcentre), -N/2))` in `gft.c`.
    ///
    /// The time-domain Gaussian is
    /// `g[s] = |fc|/sqrt(2*PI) * exp(-((s/N - 0.5)^2) * fc^2 / 2)`, normalised to
    /// unit sum, then circularly shifted so its peak (sample N/2) lands on
    /// sample 0. `F` is its DFT, which is real and even.
    fn freq_gaussian(&mut self, fcentre: i64) -> &[f64] {
        let n = self.n;
        let nf = n as f64;
        let fc = fcentre.abs() as f64;
        let half = n / 2;
        // Build the circularly shifted Gaussian directly:
        // groll[s] = g[(s + N/2) mod N]. Only samples with a representable
        // exponential are evaluated; the rest stay exactly zero.
        self.time.clear();
        self.time.resize(n, 0.0);
        let mut sum = 0.0;
        let reach = if fc > 0.0 {
            ((-2.0 * EXP_UNDERFLOW).sqrt() / fc * nf).ceil() as usize + 1
        } else {
            n
        };
        let lo = half.saturating_sub(reach);
        let hi = (half + reach).min(n - 1);
        // Accumulate in the original sample order s = 0..n so the normalising
        // sum is bit-identical to summing every (possibly zero) sample.
        for s in lo..=hi {
            let x = s as f64 / nf;
            let v = fc / SQRT_2PI * (-((x - 0.5).powi(2)) * fc * fc / 2.0).exp();
            self.time[(s + n - half) % n] = v;
            sum += v;
        }
        if sum != 0.0 {
            for v in self.time.iter_mut() {
                *v /= sum;
            }
        }
        self.transform();
        &self.half
    }

    /// `half[m] = Re(DFT(time))[m]` for `m = 0..=n/2`.
    #[cfg(feature = "fft-rustfft")]
    fn transform(&mut self) {
        let n = self.n;
        let dft = self.fft.get_or_insert_with(|| {
            let plan = realfft::RealFftPlanner::<f64>::new().plan_fft_forward(n);
            let spectrum = plan.make_output_vec();
            let scratch = plan.make_scratch_vec();
            RealDft {
                plan,
                spectrum,
                scratch,
            }
        });
        dft.plan
            .process_with_scratch(&mut self.time, &mut dft.spectrum, &mut dft.scratch)
            .expect("gaussian screen fft");
        self.half.clear();
        self.half.extend(dft.spectrum.iter().map(|c| c.re));
    }

    #[cfg(not(feature = "fft-rustfft"))]
    fn transform(&mut self) {
        let n = self.n;
        let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;
        let time = &self.time;
        self.half = (0..=n / 2)
            .map(|m| {
                let w = two_pi_over_n * m as f64;
                time.iter()
                    .enumerate()
                    .map(|(s, &gs)| gs * (w * s as f64).cos())
                    .sum()
            })
            .collect();
    }

    /// `screen[k] = F[(k - width/2) mod N]`, `k = 0..width`: the Gaussian
    /// shifted so its peak sits at local index `width/2`.
    fn band(&mut self, fcentre: i64, width: usize) -> Vec<f64> {
        let n = self.n;
        let shift = width / 2;
        let fc = fcentre.abs() as f64;
        if fc >= CLOSED_FORM_MIN_FC {
            // Untruncated, the sampled Gaussian's DFT is the continuous one,
            // exp(-2 pi^2 m^2 / fc^2), periodised with period N (Poisson
            // summation); the unit-sum normalisation makes F[0] = 1. Only the
            // nearest period images can matter (|m| <= N/2), so sum k = -1..=1.
            let alpha = 2.0 * std::f64::consts::PI.powi(2) / (fc * fc);
            return periodised_screen(n, alpha, width);
        }
        let f = self.freq_gaussian(fcentre);
        (0..width)
            .map(|k| {
                let idx = (k as i64 - shift as i64).rem_euclid(n as i64) as usize;
                // F is even: F[idx] = F[n - idx].
                f[idx.min(n - idx)]
            })
            .collect()
    }

    /// Window screen for a single positive band; see [`gaussian_band_screen`].
    pub(crate) fn band_screen(&mut self, fcentre: i64, width: usize) -> Vec<f64> {
        if width == 1 {
            return vec![1.0];
        }
        self.band(fcentre, width)
    }
}

/// `screen[k] = P(k - width/2) / P(0)`, `k = 0..width`, where
/// `P(m) = sum_{j=-1..=1} exp(-alpha (m + jN)^2)` is the Gaussian
/// `exp(-alpha m^2)` periodised with period `n`. Peak value exactly 1.
fn periodised_screen(n: usize, alpha: f64, width: usize) -> Vec<f64> {
    let shift = width / 2;
    let nf = n as f64;
    let periodised = |m: f64| -> f64 {
        (-1..=1)
            .map(|k| {
                let d = m + k as f64 * nf;
                (-alpha * d * d).exp()
            })
            .sum()
    };
    let p0 = periodised(0.0);
    (0..width)
        .map(|k| periodised(k as f64 - shift as f64) / p0)
        .collect()
}

/// Gaussian screen with an explicit standard deviation `sigma` (in bins) for a
/// band of `width` bins, peak at local index `width/2`. The plain band screen
/// of [`gaussian_band_screen`] is this with `sigma = fcentre / (2 pi)` (for
/// `fcentre >= 24`, up to rounding).
pub fn gaussian_sigma_screen(n: usize, sigma: f64, width: usize) -> Vec<f64> {
    if width == 1 {
        return vec![1.0];
    }
    periodised_screen(n, 1.0 / (2.0 * sigma * sigma), width)
}

/// Window screen for a single positive band `[lo, lo+width)` with the given
/// frequency-domain centre parameter `fcentre`. The peak sits at local index
/// `width/2` (i.e. on source bin `lo + width/2`).
///
/// Width-1 bands are exact (no taper): screen is `[1.0]`, matching the C
/// "width-1 bands are always weighted 1.0" rule.
pub fn gaussian_band_screen(n: usize, fcentre: i64, width: usize) -> Vec<f64> {
    GaussianScreens::new(n).band_screen(fcentre, width)
}

/// Box window screen: all ones (matches C `box`, which fills the band with 1.0).
pub fn box_band_screen(width: usize) -> Vec<f64> {
    vec![1.0; width]
}

/// Faithful port of `windows()` in `gft.c`: the full length-`n` global screen for
/// the symmetric complex dyadic scheme, including the `win[0]=1`, `win[N-1]=1`
/// edge cases and the positive/negative band mirroring. Returns the real part
/// (the imaginary part is zero up to rounding).
pub fn complex_global_screen(n: usize, kind: WindowKind) -> Vec<f64> {
    let mut win = vec![0.0f64; n];
    let mut screens = GaussianScreens::new(n);
    // Frequencies 0 and -1 are special cases.
    win[0] = 1.0;
    win[n - 1] = 1.0;

    let mut fstart = 1usize;
    while fstart < n / 2 {
        let (fwidth, fcentre): (usize, i64) = if fstart < 2 {
            (1, (fstart + 1) as i64)
        } else {
            (fstart, (fstart + fstart / 2 + 1) as i64)
        };

        let band = match kind {
            // The C `windows()` runs the Gaussian path even for width-1 bands
            // (unlike `windowsFromPars`), so no width-1 special case here.
            WindowKind::Gaussian => screens.band(fcentre, fwidth),
            WindowKind::Box => box_band_screen(fwidth),
        };

        for (k, &val) in band.iter().enumerate() {
            // positive band: win[fstart + k]
            win[fstart + k] = val;
            // negative (mirror) band: win[N - fstart - 1 - k]
            let neg = n as i64 - fstart as i64 - 1 - k as i64;
            if neg >= 0 {
                win[neg as usize] = val;
            }
        }

        fstart *= 2;
    }
    win
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_screen_matches_dft_screen() {
        // The closed form is only used for fc >= CLOSED_FORM_MIN_FC; it must
        // agree with the DFT of the truncated Gaussian to rounding there.
        for n in [64usize, 1024, 8192] {
            let mut screens = GaussianScreens::new(n);
            let mut fc = CLOSED_FORM_MIN_FC as usize;
            while fc <= n / 2 {
                for width in [2usize, 3, fc / 2, fc.min(n / 4 + 1)] {
                    let closed = screens.band(fc as i64, width);
                    let f = screens.freq_gaussian(fc as i64).to_vec();
                    for (k, &c) in closed.iter().enumerate() {
                        let idx = (k as i64 - (width / 2) as i64).rem_euclid(n as i64) as usize;
                        let d = f[idx.min(n - idx)];
                        assert!(
                            (c - d).abs() <= 1e-13 * d.abs().max(1e-300) + 1e-300,
                            "n={n} fc={fc} width={width} k={k}: {c} vs {d}"
                        );
                    }
                }
                fc = fc * 3 / 2;
            }
        }
    }
}
