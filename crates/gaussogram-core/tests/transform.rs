//! Transform tests: C parity (dyadic_complex), round-trip (dyadic_real and the
//! dyadic_dual_real frame inverse), band-edge tone win (dyadic_dual_real), and
//! batch consistency.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

use gaussogram_core::{
    build, dyadic_complex, dyadic_dual_real, dyadic_dual_real_with, dyadic_real, dyadic_real_with,
    GaussogramError, TopWindow, WindowKind, TOP_EDGE_GAIN,
};
use num_complex::Complex;

/// Deterministic complex input — MUST match `golden/gen_golden.c::make_input`.
fn golden_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let ii = i as f64;
            let t = ii / n as f64;
            let mut re = (2.0 * PI * 2.0 * ii / n as f64).sin()
                + 0.5 * (2.0 * PI * 5.0 * ii / n as f64).cos();
            let im = 0.25 * (2.0 * PI * 3.0 * ii / n as f64).sin()
                - 0.1 * (2.0 * PI * 1.0 * ii / n as f64).cos();
            re += 0.05 * t;
            Complex::new(re, im)
        })
        .collect()
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/complex_golden.txt")
}

/// Deterministic real input — MUST match `golden/gen_real_golden.py::make_input`.
fn real_golden_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let ii = i as f64;
            (2.0 * PI * 2.0 * ii / n as f64).sin()
                + 0.5 * (2.0 * PI * 5.0 * ii / n as f64).cos()
                + 0.05 * ii / n as f64
        })
        .collect()
}

fn real_golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/real_golden.txt")
}

/// Regression vectors for the real schemes. Unlike the complex parity test
/// (checked against the reference C library), these were generated from this
/// implementation after its forward values were validated against an
/// independent numpy implementation of the definition
/// (python/tests/test_reference_impl.py); they guard against numerical drift.
#[test]
fn real_golden_regression() {
    let text = fs::read_to_string(real_golden_path()).expect("golden file present");
    let mut lines = text.lines();
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let mut it = header.split_whitespace();
        assert_eq!(it.next(), Some("SCHEME"));
        let scheme_name = it.next().unwrap();
        assert_eq!(it.next(), Some("N"));
        let n: usize = it.next().unwrap().parse().unwrap();
        assert_eq!(it.next(), Some("BPO"));
        let bpo: usize = it.next().unwrap().parse().unwrap();
        assert_eq!(it.next(), Some("LEN"));
        let len: usize = it.next().unwrap().parse().unwrap();

        let mut expected = Vec::with_capacity(len);
        for _ in 0..len {
            let line = lines.next().unwrap();
            let mut p = line.split_whitespace();
            let re: f64 = p.next().unwrap().parse().unwrap();
            let im: f64 = p.next().unwrap().parse().unwrap();
            expected.push(Complex::new(re, im));
        }

        let scheme = match scheme_name {
            "dyadic_real" => dyadic_real_with(n, WindowKind::Gaussian, bpo).unwrap(),
            "dyadic_dual_real" => {
                // Frozen before the wide top window became the default.
                dyadic_dual_real_with(n, WindowKind::Gaussian, TopWindow::Gaussian, bpo).unwrap()
            }
            other => panic!("unknown scheme in golden file: {other}"),
        };
        let engine = build(scheme);
        assert_eq!(engine.output_len(), len, "{scheme_name} N={n} bpo={bpo}");
        let signal = real_golden_input(n);
        let mut out = vec![Complex::new(0.0, 0.0); len];
        engine.forward(&signal, &mut out).unwrap();

        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "{scheme_name} N={n} bpo={bpo} coeff {i}: got {a}, expected {b}"
            );
        }
        cases += 1;
    }
    assert!(cases >= 10, "expected at least 10 golden cases, got {cases}");
}

#[test]
fn complex_parity_against_c_reference() {
    let text = fs::read_to_string(golden_path()).expect("golden file present");
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let mut it = header.split_whitespace();
        assert_eq!(it.next(), Some("N"));
        let n: usize = it.next().unwrap().parse().unwrap();

        let mut expected = Vec::with_capacity(n);
        for _ in 0..n {
            let line = lines.next().unwrap();
            let mut p = line.split_whitespace();
            let re: f64 = p.next().unwrap().parse().unwrap();
            let im: f64 = p.next().unwrap().parse().unwrap();
            expected.push(Complex::new(re, im));
        }

        let engine = build(gaussogram_core::dyadic_complex(n).unwrap());
        let signal = golden_input(n);
        let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
        engine.forward_complex(&signal, &mut out).unwrap();

        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "N={n} coeff {i}: got {a}, expected {b}, |diff|={}",
                (a - b).norm()
            );
        }
    }
}

#[test]
fn real_round_trip() {
    for k in 3..=12 {
        let n = 1usize << k;
        let engine = build(dyadic_real(n).unwrap());
        // A deterministic real signal with multiple components.
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let ii = i as f64;
                (2.0 * PI * 3.0 * ii / n as f64).sin()
                    + 0.4 * (2.0 * PI * 11.0 * ii / n as f64).cos()
                    + 0.1 * ii / n as f64
            })
            .collect();
        let mut coeffs = vec![Complex::new(0.0, 0.0); engine.output_len()];
        engine.forward(&x, &mut coeffs).unwrap();
        let mut recon = vec![0.0f64; n];
        engine.inverse(&coeffs, &mut recon).unwrap();
        for (i, (a, b)) in x.iter().zip(recon.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "N={n} sample {i}: x={a}, recon={b}, diff={}",
                (a - b).abs()
            );
        }
    }
}

#[test]
fn real_round_trip_subdivided_octaves() {
    // dyadic_real stays exactly invertible for every power-of-two bands_per_octave.
    for &bpo in &[2usize, 4, 8] {
        for k in 4..=12 {
            let n = 1usize << k;
            let engine = build(dyadic_real_with(n, WindowKind::Gaussian, bpo).unwrap());
            assert_eq!(engine.output_len(), n / 2 + 1, "N={n} bpo={bpo}");
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    let ii = i as f64;
                    (2.0 * PI * 3.0 * ii / n as f64).sin()
                        + 0.4 * (2.0 * PI * 11.0 * ii / n as f64).cos()
                        + 0.1 * ii / n as f64
                })
                .collect();
            let mut coeffs = vec![Complex::new(0.0, 0.0); engine.output_len()];
            engine.forward(&x, &mut coeffs).unwrap();
            let mut recon = vec![0.0f64; n];
            engine.inverse(&coeffs, &mut recon).unwrap();
            for (i, (a, b)) in x.iter().zip(recon.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10,
                    "N={n} bpo={bpo} sample {i}: diff={}",
                    (a - b).abs()
                );
            }
        }
    }
}

#[test]
fn complex_is_not_invertible() {
    let engine = build(dyadic_complex(64).unwrap());
    let coeffs = vec![Complex::new(0.0, 0.0); engine.output_len()];
    let mut out = vec![0.0f64; 64];
    assert_eq!(
        engine.inverse(&coeffs, &mut out),
        Err(GaussogramError::NotInvertible("dyadic_complex"))
    );
}

/// Deterministic pseudo-random sequence in [-1, 1) (LCG; no rand dependency).
fn lcg(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

#[test]
fn dual_round_trip() {
    // The frame inverse of dyadic_dual_real is exact for unmodified
    // coefficients, for every bands_per_octave, window and top window.
    for &bpo in &[1usize, 2, 4, 8] {
        for kind in [WindowKind::Gaussian, WindowKind::Box] {
            for top in [TopWindow::Wide, TopWindow::Gaussian, TopWindow::Flat] {
                for k in 3..=12 {
                    let n = 1usize << k;
                    let Ok(scheme) = dyadic_dual_real_with(n, kind, top, bpo) else {
                        continue; // bpo too large for this N
                    };
                    let engine = build(scheme);
                    let x = lcg(k as u64, n);
                    let mut coeffs = vec![Complex::new(0.0, 0.0); engine.output_len()];
                    engine.forward(&x, &mut coeffs).unwrap();
                    let mut recon = vec![0.0f64; n];
                    engine.inverse(&coeffs, &mut recon).unwrap();
                    let err = x
                        .iter()
                        .zip(&recon)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        err < 1e-10,
                        "N={n} bpo={bpo} {kind:?} {top:?}: err={err}"
                    );
                }
            }
        }
    }
}

fn residual(
    engine: &gaussogram_core::Gaussogram1d,
    c: &[Complex<f64>],
    x: &[f64],
) -> Vec<Complex<f64>> {
    let mut tx = vec![Complex::new(0.0, 0.0); c.len()];
    engine.forward(x, &mut tx).unwrap();
    c.iter().zip(&tx).map(|(a, b)| a - b).collect()
}

/// Per-coefficient band width, so `sqrt(width) * c` is in energy-preserving
/// units (each band stores a `1/width`-normalised inverse FFT).
fn coeff_widths(engine: &gaussogram_core::Gaussogram1d) -> Vec<f64> {
    let mut w = vec![0.0; engine.output_len()];
    for band in &engine.scheme().bands {
        w[band.out_off..band.out_off + band.width()].fill(band.width() as f64);
    }
    w
}

/// Norm in signal-energy units: `sqrt(sum_i width_i * |v_i|^2)`.
fn energy_norm(v: &[Complex<f64>], widths: &[f64]) -> f64 {
    v.iter().zip(widths).map(|(z, w)| w * z.norm_sqr()).sum::<f64>().sqrt()
}

#[test]
fn dual_inverse_is_least_squares() {
    // For perturbed coefficients the frame inverse returns the least-squares
    // signal in signal energy: the coefficient residual is orthogonal to the
    // range of the transform under the width-weighted inner product, and no
    // larger (in that norm) than that of inverting tiling A alone.
    for &bpo in &[1usize, 4] {
        let n = 512;
        let engine =
            build(dyadic_dual_real_with(n, WindowKind::Gaussian, TopWindow::Wide, bpo).unwrap());
        let a_only = build(dyadic_real_with(n, WindowKind::Gaussian, bpo).unwrap());
        let olen = engine.output_len();
        let widths = coeff_widths(&engine);

        let x = lcg(1, n);
        let mut c = vec![Complex::new(0.0, 0.0); olen];
        engine.forward(&x, &mut c).unwrap();
        let (re, im) = (lcg(2, olen), lcg(3, olen));
        for (z, (r, i)) in c.iter_mut().zip(re.iter().zip(&im)) {
            *z += Complex::new(*r, *i) * 0.05;
        }

        let mut xhat = vec![0.0f64; n];
        engine.inverse(&c, &mut xhat).unwrap();
        let r = residual(&engine, &c, &xhat);

        // Orthogonality: sum_i width_i * Re(conj(r_i) * (T y)_i) = 0 for any real y.
        for seed in 10..15 {
            let mut ty = vec![Complex::new(0.0, 0.0); olen];
            engine.forward(&lcg(seed, n), &mut ty).unwrap();
            let dot: f64 = r
                .iter()
                .zip(&ty)
                .zip(&widths)
                .map(|((a, b), w)| w * (a.conj() * b).re)
                .sum();
            let rel = dot.abs() / (energy_norm(&r, &widths) * energy_norm(&ty, &widths));
            assert!(
                rel < 1e-10,
                "bpo={bpo} seed={seed}: <r, Ty>_w/(|r||Ty|) = {rel}"
            );
        }

        let mut x_a = vec![0.0f64; n];
        a_only.inverse(&c[..n / 2 + 1], &mut x_a).unwrap();
        let r_a = residual(&engine, &c, &x_a);
        assert!(
            energy_norm(&r, &widths) <= energy_norm(&r_a, &widths) * (1.0 + 1e-12),
            "bpo={bpo}: frame residual {} > tiling-A residual {}",
            energy_norm(&r, &widths),
            energy_norm(&r_a, &widths)
        );
    }
}

/// Reference frame inverse with a chosen blend of the overlapping bands: each
/// band's FFT is `W_b * X` on its bins, and bin `k` is estimated as
/// `sum_b weight_b[k] * FFT(c_b)[k] * W_b[k] / sum_b weight_b[k] * W_b[k]^2`.
fn reference_frame_inverse(
    engine: &gaussogram_core::Gaussogram1d,
    c: &[Complex<f64>],
    band_weight: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let scheme = engine.scheme();
    let n = scheme.n;
    let mut num = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    let mut den = vec![0.0f64; n / 2 + 1];
    let mut planner = rustfft::FftPlanner::<f64>::new();
    for (band, window) in scheme.bands.iter().zip(scheme.windows.iter()) {
        let w = band.width();
        let mut y = c[band.out_off..band.out_off + w].to_vec();
        planner.plan_fft_forward(w).process(&mut y);
        let bw = band_weight(w);
        for (j, (&yj, &sc)) in y.iter().zip(window.screen.iter()).enumerate() {
            num[band.src_lo + j] += yj * (bw * sc);
            den[band.src_lo + j] += bw * sc * sc;
        }
    }
    let mut spec: Vec<Complex<f64>> = num.iter().zip(&den).map(|(a, d)| a / d).collect();
    spec[0].im = 0.0;
    spec[n / 2].im = 0.0;
    let mut out = vec![0.0f64; n];
    realfft::RealFftPlanner::<f64>::new()
        .plan_fft_inverse(n)
        .process(&mut spec, &mut out)
        .unwrap();
    out.iter().map(|v| v / n as f64).collect()
}

#[test]
fn dual_inverse_energy_weighting_beats_stored_weighting_for_independent_noise() {
    // Error model: zero-mean errors on the stored coefficients, uncorrelated
    // within each band with covariance proportional to I/width, and
    // uncorrelated across bands (here independent uniform noise scaled by
    // 1/sqrt(width)). Every band's estimate of a bin is then equally noisy, so
    // blending by window power alone (the engine) must beat the
    // stored-coefficient least squares that also divides each band's weight by
    // its width. White noise added to the input before `forward` does not fit
    // this model and would not test it: overlapping bands carry W_b[k] times
    // the same noise bin, so every exact inverse returns the same noisy
    // signal. The gain is small (about 0.02-0.04 dB overall, up to ~0.5 dB at
    // individual overlap bins), so the check is strict ordering on fixed seeds
    // rather than a margin.
    for &bpo in &[1usize, 4] {
        let n = 1024;
        let engine =
            build(dyadic_dual_real_with(n, WindowKind::Gaussian, TopWindow::Wide, bpo).unwrap());
        let olen = engine.output_len();
        let widths = coeff_widths(&engine);
        let (mut err_energy, mut err_stored) = (0.0f64, 0.0f64);
        for trial in 0..20u64 {
            let x = lcg(100 + trial, n);
            let mut c = vec![Complex::new(0.0, 0.0); olen];
            engine.forward(&x, &mut c).unwrap();
            let (re, im) = (lcg(200 + trial, olen), lcg(300 + trial, olen));
            for ((z, w), (r, i)) in c.iter_mut().zip(&widths).zip(re.iter().zip(&im)) {
                *z += Complex::new(*r, *i) * (0.05 / w.sqrt());
            }

            let mut x_energy = vec![0.0f64; n];
            engine.inverse(&c, &mut x_energy).unwrap();
            // The reference with energy weights reproduces the engine, which
            // validates the reference used for the stored-coefficient blend.
            let x_ref = reference_frame_inverse(&engine, &c, |_| 1.0);
            for (a, b) in x_energy.iter().zip(&x_ref) {
                assert!((a - b).abs() < 1e-12, "bpo={bpo}: engine {a} vs reference {b}");
            }
            let x_stored = reference_frame_inverse(&engine, &c, |w| 1.0 / w as f64);

            let sq = |xh: &[f64]| xh.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            err_energy += sq(&x_energy);
            err_stored += sq(&x_stored);
        }
        let gain_db = 10.0 * (err_stored / err_energy).log10();
        eprintln!("bpo={bpo}: energy weighting improves independent-noise error by {gain_db:.2} dB");
        assert!(
            err_energy < err_stored,
            "bpo={bpo}: energy-weighted error {err_energy} not below stored-weighted {err_stored}"
        );
    }
}

/// Energy captured for a pure tone at a given integer frequency bin, summed
/// over the bands whose source range includes that bin.
fn band_response_at_bin(
    scheme: &gaussogram_core::Scheme,
    out: &[Complex<f64>],
    bin: usize,
    a_band_count: usize,
) -> (f64, f64) {
    // Returns (A-band energy, B-band energy) for bands covering `bin`.
    let mut a_energy = 0.0;
    let mut b_energy = 0.0;
    for (idx, band) in scheme.bands.iter().enumerate() {
        if bin >= band.src_lo && bin < band.src_hi {
            let slice = &out[band.out_off..band.out_off + band.width()];
            let e: f64 = slice.iter().map(|c| c.norm_sqr()).sum();
            if idx < a_band_count {
                a_energy += e;
            } else {
                b_energy += e;
            }
        }
    }
    (a_energy, b_energy)
}

#[test]
fn band_edge_tone_rescued_by_tiling_b() {
    // The point of the dual scheme: tones at A's octave joins (dead zones) are
    // captured by the corresponding symmetric B band.
    let n = 256;
    let scheme = dyadic_dual_real(n).unwrap();
    let a_band_count = scheme
        .bands
        .iter()
        .take_while(|b| b.src_hi <= n / 2 + 1 && {
            // A bands are the first run summing to N/2+1; detect by out_off < N/2+1
            b.out_off < n / 2 + 1
        })
        .count();
    let engine = build(scheme.clone());

    // Joins (A's interior boundaries) are the B band centres: 8,16,32,64.
    for &join in &[8usize, 16, 32, 64] {
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * join as f64 * i as f64 / n as f64).cos())
            .collect();
        let mut out = vec![Complex::new(0.0, 0.0); engine.output_len()];
        engine.forward(&x, &mut out).unwrap();

        let (a_e, b_e) = band_response_at_bin(&scheme, &out, join, a_band_count);
        // The tone sits exactly on a B-band peak, so B captures more energy than
        // the A dead-zone response at the same bin.
        assert!(
            b_e > a_e,
            "join={join}: expected B energy ({b_e}) > A dead-zone energy ({a_e})"
        );
    }
}

/// Index of tiling A's top band: the one ending at N/2, just below the
/// width-1 Nyquist band.
fn top_band_idx(s: &gaussogram_core::Scheme) -> usize {
    s.bands.iter().position(|b| b.src_hi == s.n / 2).unwrap()
}

/// Frame diagonal `D[k] = sum over bands covering k of screen^2`, k = 0..=N/2.
fn frame_diagonal(s: &gaussogram_core::Scheme) -> Vec<f64> {
    let mut d = vec![0.0f64; s.n / 2 + 1];
    for (b, w) in s.bands.iter().zip(&s.windows) {
        for (k, v) in w.screen.iter().enumerate() {
            d[b.src_lo + k] += v * v;
        }
    }
    d
}

fn min_max(d: &[f64]) -> (f64, f64) {
    d.iter()
        .fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

#[test]
fn wide_top_window_bounds_the_dual_frame() {
    // The default (wide) top window lifts the dead zone below Nyquist to the
    // floor of the rest of the frame.
    for n in [256usize, 1024, 4096] {
        for bpo in [1usize, 2, 4] {
            let s = dyadic_dual_real_with(n, WindowKind::Gaussian, TopWindow::Wide, bpo).unwrap();
            let (lo, hi) = min_max(&frame_diagonal(&s));
            assert!(lo >= 0.39, "N={n} bpo={bpo}: min D = {lo}");
            if bpo == 1 {
                assert!(hi / lo < 4.0, "N={n}: condition number {}", hi / lo);
            }
        }
    }
    // The plain Gaussian leaves the dead zone (condition number ~84 at N=1024).
    let s = dyadic_dual_real_with(1024, WindowKind::Gaussian, TopWindow::Gaussian, 1).unwrap();
    let (lo, hi) = min_max(&frame_diagonal(&s));
    assert!(lo < 0.02 && hi / lo > 50.0, "min D = {lo}, cond = {}", hi / lo);
}

#[test]
fn top_window_shapes() {
    let n = 1024;
    let build_top = |top, bpo| {
        let s = dyadic_dual_real_with(n, WindowKind::Gaussian, top, bpo).unwrap();
        let i = top_band_idx(&s);
        (s.windows[i].clone(), s.bands[i])
    };
    // bpo = 1: the wide window reaches TOP_EDGE_GAIN at its lower edge, peaks
    // at 1 on w/2, and is symmetric about the peak.
    let (wide, band) = build_top(TopWindow::Wide, 1);
    let (plain, _) = build_top(TopWindow::Gaussian, 1);
    let (flat, _) = build_top(TopWindow::Flat, 1);
    let w = band.width();
    assert_eq!(w, n / 4);
    assert_eq!(wide.top_window, Some(TopWindow::Wide));
    assert!((wide.screen[0] - TOP_EDGE_GAIN).abs() < 0.005, "{}", wide.screen[0]);
    assert_eq!(wide.screen[w / 2], 1.0);
    for m in 1..w / 2 {
        assert!((wide.screen[w / 2 - m] - wide.screen[w / 2 + m]).abs() < 1e-12);
    }
    // Never narrower than the plain Gaussian.
    for (a, b) in wide.screen.iter().zip(&plain.screen) {
        assert!(a >= b);
    }
    // The plain option is the ordinary band screen; the flat one holds the peak
    // value from the peak to Nyquist and keeps the plain lower half.
    let ordinary = gaussogram_core::window::gaussian_band_screen(n, wide.fcentre as i64, w);
    assert_eq!(plain.screen, ordinary);
    assert!(plain.screen[w - 1] < 0.2);
    assert_eq!(&flat.screen[..w / 2], &plain.screen[..w / 2]);
    for v in &flat.screen[w / 2..] {
        assert!((v - plain.screen[w / 2]).abs() < 1e-15);
    }
    // bpo >= 2: the plain Gaussian is already the wider one, so the default
    // leaves it unchanged.
    for bpo in [2usize, 4] {
        let (wide, _) = build_top(TopWindow::Wide, bpo);
        let (plain, _) = build_top(TopWindow::Gaussian, bpo);
        assert_eq!(wide.screen, plain.screen, "bpo={bpo}");
    }
    // Only tiling A's top band carries the option; box windows ignore it.
    let s = dyadic_dual_real(n).unwrap();
    let i = top_band_idx(&s);
    for (j, win) in s.windows.iter().enumerate() {
        assert_eq!(win.top_window.is_some(), j == i);
    }
    let boxed = dyadic_dual_real_with(n, WindowKind::Box, TopWindow::Wide, 1).unwrap();
    assert!(boxed
        .windows
        .iter()
        .all(|w| w.top_window.is_none() && w.screen.iter().all(|&v| v == 1.0)));
}

#[test]
fn batch_matches_single() {
    let n = 128;
    let engine = build(dyadic_dual_real(n).unwrap());
    let olen = engine.output_len();
    let sigs: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..n)
                .map(|i| (2.0 * PI * (s + 2) as f64 * i as f64 / n as f64).sin())
                .collect()
        })
        .collect();
    let refs: Vec<&[f64]> = sigs.iter().map(|v| v.as_slice()).collect();

    let mut batched = vec![Complex::new(0.0, 0.0); olen * sigs.len()];
    engine.forward_batch(&refs, &mut batched).unwrap();

    for (s, sig) in sigs.iter().enumerate() {
        let mut single = vec![Complex::new(0.0, 0.0); olen];
        engine.forward(sig, &mut single).unwrap();
        for (a, b) in single.iter().zip(&batched[s * olen..(s + 1) * olen]) {
            assert!((a - b).norm() < 1e-14);
        }
    }
}

#[test]
fn forward_with_reused_scratch_matches_forward() {
    // A single reused Scratch across many calls must produce bit-identical
    // results to the allocate-per-call `forward` (no state leaks between calls).
    let n = 256;
    let engine = build(dyadic_dual_real(n).unwrap());
    let olen = engine.output_len();
    let mut scratch = engine.alloc_scratch();

    for f in [3.0_f64, 17.0, 64.0, 100.0] {
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / n as f64).cos())
            .collect();

        let mut reused = vec![Complex::new(0.0, 0.0); olen];
        engine.forward_with(&sig, &mut reused, &mut scratch).unwrap();

        let mut fresh = vec![Complex::new(0.0, 0.0); olen];
        engine.forward(&sig, &mut fresh).unwrap();

        assert_eq!(reused, fresh);
    }
}

#[test]
fn engine_cache_shares_engines_and_normalises_ignored_options() {
    use gaussogram_core::{cached_engine, SchemeKind, SchemeSpec, WindowKind};
    use std::sync::Arc;
    let spec = SchemeSpec {
        kind: SchemeKind::DyadicReal,
        n: 512,
        window: WindowKind::Gaussian,
        top_window: TopWindow::Wide,
        bands_per_octave: 2,
    };
    let a = cached_engine(spec).unwrap();
    let b = cached_engine(spec).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    // top_window is meaningless for dyadic_real: same engine.
    let c = cached_engine(SchemeSpec {
        top_window: TopWindow::Flat,
        ..spec
    })
    .unwrap();
    assert!(Arc::ptr_eq(&a, &c));
    // A different bands_per_octave is a different engine with a different layout.
    let d = cached_engine(SchemeSpec {
        bands_per_octave: 1,
        ..spec
    })
    .unwrap();
    assert!(!Arc::ptr_eq(&a, &d));
    assert_ne!(a.scheme().bands.len(), d.scheme().bands.len());
    // Errors are not cached.
    assert!(cached_engine(SchemeSpec { n: 500, ..spec }).is_err());
}
