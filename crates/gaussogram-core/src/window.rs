//! Window-screen construction, reproducing the numerics of `gaussian`,
//! `windows`, and `windowsFromPars` in the reference `gft.c`.
//!
//! Each band's window is a frequency-domain taper, stored as a real `Vec<f64>`.
//! The frequency-domain Gaussian is real because the underlying time-domain
//! Gaussian (after the half-length circular shift that centres its peak on
//! sample 0) is real and even, so its DFT is real and even.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

const SQRT_2PI: f64 = 2.506_628_274_631_000_7; // sqrt(2*PI)

/// Frequency-domain Gaussian `F[m]`, `m = 0..n`, matching the real part of
/// `fft(shift(gaussian(N, fcentre), -N/2))` in `gft.c`.
///
/// The time-domain Gaussian is `g[s] = |fc|/sqrt(2*PI) * exp(-((s/N - 0.5)^2) * fc^2 / 2)`,
/// normalised to unit sum, then circularly shifted so its peak (sample N/2) lands
/// on sample 0. `F` is computed directly as `F[m] = sum_s groll[s] * cos(2*PI*s*m/N)`.
fn freq_gaussian(n: usize, fcentre: i64) -> Vec<f64> {
    let nf = n as f64;
    let fc = fcentre.abs() as f64;
    // Time-domain Gaussian, unnormalised.
    let mut g = vec![0.0f64; n];
    let mut sum = 0.0;
    for (s, gs) in g.iter_mut().enumerate() {
        let x = s as f64 / nf;
        let v = fc / SQRT_2PI * (-((x - 0.5).powi(2)) * fc * fc / 2.0).exp();
        *gs = v;
        sum += v;
    }
    if sum != 0.0 {
        for gs in g.iter_mut() {
            *gs /= sum;
        }
    }
    // Circularly shift so the peak (at sample N/2) lands on sample 0:
    // groll[s] = g[(s + N/2) mod N].
    let half = n / 2;
    let mut groll = vec![0.0f64; n];
    for s in 0..n {
        groll[s] = g[(s + half) % n];
    }
    freq_gaussian_from_time(&groll, n)
}

/// DFT of the (real, even) shifted Gaussian. The result is real; we return the
/// real part. Uses `rustfft` (O(N log N)) when available, else a direct DFT.
#[cfg(feature = "fft-rustfft")]
fn freq_gaussian_from_time(groll: &[f64], n: usize) -> Vec<f64> {
    use num_complex::Complex;
    use rustfft::{FftDirection, FftPlanner};
    let mut buf: Vec<Complex<f64>> = groll.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let fft = FftPlanner::<f64>::new().plan_fft(n, FftDirection::Forward);
    fft.process(&mut buf);
    buf.iter().map(|c| c.re).collect()
}

#[cfg(not(feature = "fft-rustfft"))]
fn freq_gaussian_from_time(groll: &[f64], n: usize) -> Vec<f64> {
    let nf = n as f64;
    let two_pi_over_n = 2.0 * std::f64::consts::PI / nf;
    (0..n)
        .map(|m| {
            let w = two_pi_over_n * m as f64;
            groll
                .iter()
                .enumerate()
                .map(|(s, &gs)| gs * (w * s as f64).cos())
                .sum()
        })
        .collect()
}

/// Window screen for a single positive band `[lo, lo+width)` with the given
/// frequency-domain centre parameter `fcentre`. The peak sits at local index
/// `width/2` (i.e. on source bin `lo + width/2`).
///
/// Width-1 bands are exact (no taper): screen is `[1.0]`, matching the C
/// "width-1 bands are always weighted 1.0" rule.
pub fn gaussian_band_screen(n: usize, fcentre: i64, width: usize) -> Vec<f64> {
    if width == 1 {
        return vec![1.0];
    }
    let f = freq_gaussian(n, fcentre);
    let shift = width / 2;
    let mut screen = vec![0.0f64; width];
    for (k, sc) in screen.iter_mut().enumerate() {
        // shift(F, +width/2) then take first `width`: screen[k] = F[(k - width/2) mod N].
        let idx = ((k as i64 - shift as i64).rem_euclid(n as i64)) as usize;
        *sc = f[idx];
    }
    screen
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
            WindowKind::Gaussian => gaussian_band_screen_centred(n, fcentre, fwidth),
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

/// Helper used by `complex_global_screen`: identical to `gaussian_band_screen`
/// but keeps width-1 bands as a real Gaussian DC value of 1.0 too. For width 1
/// the C `windows()` still runs the Gaussian path (it does not special-case
/// width 1 the way `windowsFromPars` does), so reproduce that.
fn gaussian_band_screen_centred(n: usize, fcentre: i64, width: usize) -> Vec<f64> {
    let f = freq_gaussian(n, fcentre);
    let shift = width / 2;
    let mut screen = vec![0.0f64; width];
    for (k, sc) in screen.iter_mut().enumerate() {
        let idx = ((k as i64 - shift as i64).rem_euclid(n as i64)) as usize;
        *sc = f[idx];
    }
    screen
}
