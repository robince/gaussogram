//! Transform tests: C parity (dyadic_complex), round-trip (dyadic_real),
//! band-edge tone win (dyadic_dual_real), and batch consistency.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

use gaussogram_core::{
    build, dyadic_dual_real, dyadic_dual_real_with, dyadic_real, dyadic_real_with, GaussogramError,
    WindowKind,
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
fn dual_is_not_invertible() {
    let engine = build(dyadic_dual_real(64).unwrap());
    let coeffs = vec![Complex::new(0.0, 0.0); engine.output_len()];
    let mut out = vec![0.0f64; 64];
    assert_eq!(
        engine.inverse(&coeffs, &mut out),
        Err(GaussogramError::NotInvertible("dyadic_dual_real"))
    );
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
    // Top A band is the one ending at N/2+1.
    let top_idx = plain
        .bands
        .iter()
        .position(|b| b.src_hi == n / 2 + 1)
        .unwrap();
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
