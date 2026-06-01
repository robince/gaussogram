//! Scheme-invariant tests (§8) for the partitioning constructors.

use gaussogram_core::{
    complex_partitions, dyadic_complex, dyadic_dual_real, dyadic_real, legacy_real_partitions,
    Band, GaussogramError,
};
use proptest::prelude::*;

fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Tiling A occupies the first `N/2+1` output coefficients; return its bands.
fn tiling_a_bands(bands: &[Band], n: usize) -> Vec<Band> {
    let a_len = n / 2 + 1;
    let mut out = Vec::new();
    let mut acc = 0;
    for b in bands {
        if acc >= a_len {
            break;
        }
        out.push(*b);
        acc += b.width();
    }
    out
}

#[test]
fn dual_total_lengths_match_spec() {
    for k in 3..=14 {
        let n = 1usize << k;
        let s = dyadic_dual_real(n).unwrap();
        assert_eq!(s.output_len, n - 1, "N={n}");
        assert_eq!(s.output_len, s.bands.iter().map(Band::width).sum::<usize>());
    }
}

#[test]
fn dual_worked_example_n256() {
    let s = dyadic_dual_real(256).unwrap();
    // Tiling A bands.
    let a: Vec<(usize, usize)> = tiling_a_bands(&s.bands, 256)
        .iter()
        .map(|b| (b.src_lo, b.src_hi))
        .collect();
    assert_eq!(
        a,
        vec![
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 129),
        ]
    );
    // Tiling B bands follow.
    let b: Vec<(usize, usize)> = s.bands[a.len()..]
        .iter()
        .map(|b| (b.src_lo, b.src_hi))
        .collect();
    assert_eq!(
        b,
        vec![
            (1, 3),
            (2, 6),
            (4, 12),
            (8, 24),
            (16, 48),
            (32, 96),
        ]
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn dual_invariants(k in 3usize..=13) {
        let n = 1usize << k;
        let s = dyadic_dual_real(n).unwrap();

        // A is a valid exact tiling of 0..=N/2 (contiguous, no gaps/overlaps).
        let a = tiling_a_bands(&s.bands, n);
        prop_assert_eq!(a[0].src_lo, 0);
        for w in a.windows(2) {
            prop_assert_eq!(w[0].src_hi, w[1].src_lo);
        }
        prop_assert_eq!(a.last().unwrap().src_hi, n / 2 + 1);
        prop_assert_eq!(a.iter().map(Band::width).sum::<usize>(), n / 2 + 1);

        // B bands: one per interior join b in {2,4,...,N/4}.
        let b_bands = &s.bands[a.len()..];
        let mut expected_joins = Vec::new();
        let mut join = 2usize;
        while join <= n / 4 {
            expected_joins.push(join);
            join *= 2;
        }
        prop_assert_eq!(b_bands.len(), expected_joins.len());
        prop_assert_eq!(b_bands.len(), (n as f64).log2() as usize - 2);

        for (band, &b) in b_bands.iter().zip(expected_joins.iter()) {
            // symmetric about b: [b/2, 3b/2), width b, peak/centre on b.
            prop_assert_eq!(band.src_lo, b / 2);
            prop_assert_eq!(band.src_hi, 3 * b / 2);
            prop_assert_eq!(band.width(), b);
            prop_assert!(is_pow2(band.width()));
            // window centre is the join itself
            let w = &s.windows[a.len() + (b_bands.iter().position(|x| x == band).unwrap())];
            prop_assert_eq!(w.fcentre, b);
        }

        // No B band reaches Nyquist.
        for band in b_bands {
            prop_assert!(band.src_hi < n / 2 + 1);
        }
        let top_b = b_bands.last().unwrap();
        prop_assert_eq!((top_b.src_lo, top_b.src_hi), (n / 8, 3 * n / 8));

        // All widths >= 1; IFFT lengths in A's octaves and all of B are powers of two.
        for band in &s.bands {
            prop_assert!(band.width() >= 1);
        }

        // Output offsets are contiguous and cover 0..output_len exactly once.
        let mut off = 0;
        for band in &s.bands {
            prop_assert_eq!(band.out_off, off);
            off += band.width();
        }
        prop_assert_eq!(off, n - 1);
    }
}

#[test]
fn real_scheme_length_is_half_plus_one() {
    for k in 2..=12 {
        let n = 1usize << k;
        let s = dyadic_real(n).unwrap();
        assert_eq!(s.output_len, n / 2 + 1, "N={n}");
        assert!(s.invertible);
    }
}

#[test]
fn complex_scheme_length_is_n() {
    for k in 3..=10 {
        let n = 1usize << k;
        let s = dyadic_complex(n).unwrap();
        assert_eq!(s.output_len, n);
        assert!(s.complex_input);
    }
}

#[test]
fn rejects_non_power_of_two() {
    assert_eq!(dyadic_dual_real(12).unwrap_err(), GaussogramError::PowerOfTwo(12));
    assert_eq!(dyadic_real(6).unwrap_err(), GaussogramError::PowerOfTwo(6));
    assert_eq!(dyadic_complex(100).unwrap_err(), GaussogramError::PowerOfTwo(100));
}

#[test]
fn rejects_too_small_dual() {
    match dyadic_dual_real(4) {
        Err(GaussogramError::TooSmall { scheme, min, got }) => {
            assert_eq!(scheme, "dyadic_dual_real");
            assert_eq!(min, 8);
            assert_eq!(got, 4);
        }
        other => panic!("expected TooSmall, got {other:?}"),
    }
}

#[test]
fn legacy_partition_parity() {
    assert_eq!(legacy_real_partitions(8), vec![1, 2, 4, 6, 8]);
    assert_eq!(legacy_real_partitions(16), vec![1, 2, 4, 8, 11, 14, 16]);
    assert_eq!(complex_partitions(8), vec![1, 2, 4, 6, 7, 8]);
}
