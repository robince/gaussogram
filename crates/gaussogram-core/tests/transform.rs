//! Transform tests: C parity (dyadic_complex), round-trip (dyadic_real and the
//! dyadic_dual_real frame inverse), band-edge tone win (dyadic_dual_real), and
//! batch consistency.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

use gaussogram_core::{
    build, dyadic_complex, dyadic_dual_real, dyadic_dual_real_with, dyadic_real, dyadic_real_with,
    GaussogramError, WindowKind,
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
                dyadic_dual_real_with(n, WindowKind::Gaussian, false, bpo).unwrap()
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
    // coefficients, for every bands_per_octave, window and Nyquist option.
    for &bpo in &[1usize, 2, 4, 8] {
        for kind in [WindowKind::Gaussian, WindowKind::Box] {
            for flat in [false, true] {
                for k in 3..=12 {
                    let n = 1usize << k;
                    let Ok(scheme) = dyadic_dual_real_with(n, kind, flat, bpo) else {
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
                        "N={n} bpo={bpo} {kind:?} flat={flat}: err={err}"
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

fn norm(v: &[Complex<f64>]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

#[test]
fn dual_inverse_is_least_squares() {
    // For perturbed coefficients the frame inverse returns the least-squares
    // signal: the coefficient residual is orthogonal to the range of the
    // transform, and no larger than that of inverting tiling A alone.
    for &bpo in &[1usize, 4] {
        let n = 512;
        let engine = build(dyadic_dual_real_with(n, WindowKind::Gaussian, false, bpo).unwrap());
        let a_only = build(dyadic_real_with(n, WindowKind::Gaussian, bpo).unwrap());
        let olen = engine.output_len();

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

        // Orthogonality: Re<r, T y> = 0 for any real y.
        for seed in 10..15 {
            let mut ty = vec![Complex::new(0.0, 0.0); olen];
            engine.forward(&lcg(seed, n), &mut ty).unwrap();
            let dot: f64 = r.iter().zip(&ty).map(|(a, b)| (a.conj() * b).re).sum();
            let rel = dot.abs() / (norm(&r) * norm(&ty));
            assert!(
                rel < 1e-10,
                "bpo={bpo} seed={seed}: Re<r, Ty>/(|r||Ty|) = {rel}"
            );
        }

        let mut x_a = vec![0.0f64; n];
        a_only.inverse(&c[..n / 2 + 1], &mut x_a).unwrap();
        let r_a = residual(&engine, &c, &x_a);
        assert!(
            norm(&r) <= norm(&r_a) * (1.0 + 1e-12),
            "bpo={bpo}: frame residual {} > tiling-A residual {}",
            norm(&r),
            norm(&r_a)
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

#[test]
fn nyquist_flat_top_changes_top_band_window() {
    let n = 256;
    let plain = dyadic_dual_real_with(n, WindowKind::Gaussian, false, 1).unwrap();
    let flat = dyadic_dual_real_with(n, WindowKind::Gaussian, true, 1).unwrap();
    // Top A band is the one ending at N/2, just below the width-1 Nyquist band.
    let top_idx = plain.bands.iter().position(|b| b.src_hi == n / 2).unwrap();
    let nyq_idx = plain.bands.iter().position(|b| b.src_lo == n / 2).unwrap();
    assert_eq!(plain.bands[nyq_idx].width(), 1);
    // The Nyquist band itself is untapered either way.
    assert_eq!(plain.windows[nyq_idx].screen, vec![1.0]);
    assert_eq!(flat.windows[nyq_idx].screen, vec![1.0]);
    let w_plain = &plain.windows[top_idx];
    let w_flat = &flat.windows[top_idx];
    assert!(w_flat.nyquist_flat_top);
    let peak = w_flat.fcentre - plain.bands[top_idx].src_lo;
    // From the peak to Nyquist, the flat-top window is constant at the peak value.
    let peak_val = w_flat.screen[peak];
    for v in &w_flat.screen[peak..] {
        assert!((v - peak_val).abs() < 1e-15);
    }
    // The plain window decays after the peak (Nyquist-adjacent bins attenuated).
    assert!(w_plain.screen[w_plain.screen.len() - 1] < peak_val - 1e-6);
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
        nyquist_flat_top: false,
        bands_per_octave: 2,
    };
    let a = cached_engine(spec).unwrap();
    let b = cached_engine(spec).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    // nyquist_flat_top is meaningless for dyadic_real: same engine.
    let c = cached_engine(SchemeSpec {
        nyquist_flat_top: true,
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
