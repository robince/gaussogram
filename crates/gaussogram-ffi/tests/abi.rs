//! Exercise the C ABI from Rust: persistent handles must reuse their scratch
//! without leaking state between calls and match the core engine exactly.

use std::ptr;

use gaussogram_core::{
    build, dyadic_complex, dyadic_dual_real, dyadic_dual_real_with, dyadic_real, dyadic_real_with,
    WindowKind, STATUS_OK,
};
use gaussogram_ffi::*;
use num_complex::Complex;

fn signal(n: usize, f: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / n as f64).sin() + 0.1 * i as f64)
        .collect()
}

#[test]
fn handle_forward_and_inverse_match_core_across_calls() {
    let n = 256;
    for (scheme_id, scheme) in [(0, dyadic_dual_real(n)), (1, dyadic_real(n))] {
        let engine = build(scheme.unwrap());
        let olen = engine.output_len();
        let mut h: *mut GaussogramHandle = ptr::null_mut();
        assert_eq!(
            unsafe { gaussogram_create(scheme_id, n, 0, 0, &mut h) },
            STATUS_OK
        );
        assert_eq!(unsafe { gaussogram_output_len(h) }, olen);

        for f in [3.0, 17.0, 60.0] {
            let x = signal(n, f);
            let mut expected = vec![Complex::new(0.0, 0.0); olen];
            engine.forward(&x, &mut expected).unwrap();
            let mut out = vec![0.0f64; 2 * olen];
            let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n, out.as_mut_ptr(), olen) };
            assert_eq!(st, STATUS_OK);
            for (k, e) in expected.iter().enumerate() {
                assert_eq!((out[2 * k], out[2 * k + 1]), (e.re, e.im));
            }
            // Both real schemes invert (dyadic_dual_real via the frame inverse).
            let mut recon = vec![0.0f64; n];
            let st =
                unsafe { gaussogram_inverse_real(h, out.as_ptr(), olen, recon.as_mut_ptr(), n) };
            assert_eq!(st, STATUS_OK);
            for (a, b) in x.iter().zip(&recon) {
                assert!((a - b).abs() < 1e-10);
            }
        }

        // A wrong-length call reports an error and leaves the handle usable.
        let x = signal(n, 5.0);
        let mut out = vec![0.0f64; 2 * olen];
        let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n - 1, out.as_mut_ptr(), olen) };
        assert_eq!(st, STATUS_INPUT_LEN_MISMATCH);
        let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n, out.as_mut_ptr(), olen) };
        assert_eq!(st, STATUS_OK);
        unsafe { gaussogram_destroy(h) };
    }
}

#[test]
fn partitions_report_status_for_invalid_n() {
    let mut buf = [0i32; 64];
    let real = |n: usize, cap: usize, buf: &mut [i32]| unsafe {
        gaussogram_real_partitions(n, buf.as_mut_ptr(), cap)
    };
    let complex = |n: usize, cap: usize, buf: &mut [i32]| unsafe {
        gaussogram_complex_partitions(n, buf.as_mut_ptr(), cap)
    };

    assert_eq!(real(8, 64, &mut buf), 5);
    assert_eq!(&buf[..5], &[1, 2, 4, 6, 8]);
    assert_eq!(complex(8, 64, &mut buf), 6);
    assert_eq!(&buf[..6], &[1, 2, 4, 6, 7, 8]);

    assert_eq!(real(12, 64, &mut buf), -STATUS_POWER_OF_TWO);
    assert_eq!(complex(12, 64, &mut buf), -STATUS_POWER_OF_TWO);
    assert_eq!(real(0, 64, &mut buf), -STATUS_POWER_OF_TWO);
    assert_eq!(real(2, 64, &mut buf), -STATUS_TOO_SMALL);
    assert_eq!(complex(4, 64, &mut buf), -STATUS_TOO_SMALL);
    assert_eq!(real(16, 3, &mut buf), -STATUS_OUTPUT_LEN_MISMATCH);
    assert_eq!(
        unsafe { gaussogram_complex_partitions(8, ptr::null_mut(), 64) },
        -STATUS_OUTPUT_LEN_MISMATCH
    );
}

#[test]
fn create_ex_bands_per_octave_matches_core() {
    let n = 256;
    for bpo in [1usize, 2, 4, 8] {
        for (scheme_id, scheme) in [
            (
                0,
                dyadic_dual_real_with(n, WindowKind::Gaussian, false, bpo),
            ),
            (1, dyadic_real_with(n, WindowKind::Gaussian, bpo)),
        ] {
            let engine = build(scheme.unwrap());
            let olen = engine.output_len();
            let mut h: *mut GaussogramHandle = ptr::null_mut();
            let st = unsafe { gaussogram_create_ex(scheme_id, n, 0, 0, bpo, &mut h) };
            assert_eq!(st, STATUS_OK);
            // bands_per_octave reshapes the tiling but not the packed length.
            assert_eq!(unsafe { gaussogram_output_len(h) }, olen);
            let nb = engine.scheme().bands.len();
            assert_eq!(unsafe { gaussogram_num_bands(h) }, nb as i32);
            let mut lo = vec![0i64; nb];
            let mut width = vec![0i64; nb];
            let mut fc = vec![0i64; nb];
            let mut off = vec![0i64; nb];
            let got = unsafe {
                gaussogram_scheme_bands(
                    h,
                    lo.as_mut_ptr(),
                    width.as_mut_ptr(),
                    fc.as_mut_ptr(),
                    off.as_mut_ptr(),
                    nb,
                )
            };
            assert_eq!(got, nb as i32);
            for (i, b) in engine.scheme().bands.iter().enumerate() {
                assert_eq!(
                    (lo[i], width[i], off[i]),
                    (b.src_lo as i64, b.width() as i64, b.out_off as i64)
                );
            }

            let x = signal(n, 9.0);
            let mut expected = vec![Complex::new(0.0, 0.0); olen];
            engine.forward(&x, &mut expected).unwrap();
            let mut out = vec![0.0f64; 2 * olen];
            let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n, out.as_mut_ptr(), olen) };
            assert_eq!(st, STATUS_OK);
            for (k, e) in expected.iter().enumerate() {
                assert_eq!((out[2 * k], out[2 * k + 1]), (e.re, e.im));
            }
            let mut recon = vec![0.0f64; n];
            let st =
                unsafe { gaussogram_inverse_real(h, out.as_ptr(), olen, recon.as_mut_ptr(), n) };
            assert_eq!(st, STATUS_OK);
            for (a, b) in x.iter().zip(&recon) {
                assert!((a - b).abs() < 1e-10);
            }
            unsafe { gaussogram_destroy(h) };
        }
    }
}

#[test]
fn create_ex_rejects_bad_options() {
    let mut h: *mut GaussogramHandle = ptr::null_mut();
    // An invalid value is STATUS_BANDS_PER_OCTAVE for every scheme, including
    // dyadic_complex (which supports only 1), and with an unsupported
    // nyquist_flat_top too: validity is checked before scheme support.
    for scheme_id in [0, 1, 2] {
        for bpo in [0usize, 3, 6] {
            assert_eq!(
                unsafe { gaussogram_create_ex(scheme_id, 64, 0, 0, bpo, &mut h) },
                STATUS_BANDS_PER_OCTAVE,
                "scheme_id={scheme_id} bpo={bpo}"
            );
        }
    }
    assert_eq!(
        unsafe { gaussogram_create_ex(2, 64, 0, 1, 3, &mut h) },
        STATUS_BANDS_PER_OCTAVE
    );
    // A valid value dyadic_complex does not support: it has no sub-octave
    // variant.
    assert_eq!(
        unsafe { gaussogram_create_ex(2, 64, 0, 0, 2, &mut h) },
        gaussogram_core::STATUS_UNSUPPORTED_OPTION
    );
    assert_eq!(
        unsafe { gaussogram_create_ex(1, 64, 0, 0, 1, ptr::null_mut()) },
        gaussogram_core::STATUS_INTERNAL
    );
    assert!(h.is_null());
}

#[test]
fn forward_complex_matches_core() {
    let n = 64;
    let engine = build(dyadic_complex(n).unwrap());
    let olen = engine.output_len();
    let z: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
        .collect();
    let mut expected = vec![Complex::new(0.0, 0.0); olen];
    engine.forward_complex(&z, &mut expected).unwrap();

    let mut h: *mut GaussogramHandle = ptr::null_mut();
    assert_eq!(unsafe { gaussogram_create(2, n, 0, 0, &mut h) }, STATUS_OK);
    let mut out = vec![0.0f64; 2 * olen];
    let st = unsafe {
        gaussogram_forward_complex(h, z.as_ptr() as *const f64, n, out.as_mut_ptr(), olen)
    };
    assert_eq!(st, STATUS_OK);
    for (k, e) in expected.iter().enumerate() {
        assert_eq!((out[2 * k], out[2 * k + 1]), (e.re, e.im));
    }
    // Real input on a complex handle, and complex input on a real handle, are
    // reported as the wrong input kind.
    let x = signal(n, 3.0);
    let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n, out.as_mut_ptr(), olen) };
    assert_eq!(st, STATUS_WRONG_INPUT_KIND);
    unsafe { gaussogram_destroy(h) };

    let mut h: *mut GaussogramHandle = ptr::null_mut();
    assert_eq!(unsafe { gaussogram_create(1, n, 0, 0, &mut h) }, STATUS_OK);
    let olen = unsafe { gaussogram_output_len(h) };
    let mut out = vec![0.0f64; 2 * olen];
    let st = unsafe {
        gaussogram_forward_complex(h, z.as_ptr() as *const f64, n, out.as_mut_ptr(), olen)
    };
    assert_eq!(st, STATUS_WRONG_INPUT_KIND);
    unsafe { gaussogram_destroy(h) };
}
