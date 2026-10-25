//! Exercise the C ABI from Rust: persistent handles must reuse their scratch
//! without leaking state between calls and match the core engine exactly.

use std::ptr;

use gaussogram_core::{
    build, dyadic_dual_real, dyadic_dual_real_with, dyadic_real, TopWindow, WindowKind, STATUS_OK,
    STATUS_UNSUPPORTED_OPTION,
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
        assert_eq!(unsafe { gaussogram_create(scheme_id, n, 0, 0, &mut h) }, STATUS_OK);
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
fn top_window_ids_build_the_expected_schemes() {
    let n = 256;
    // Equal schemes give equal transforms of an impulse.
    let impulse_forward = |h: *mut GaussogramHandle| {
        let mut x = vec![0.0f64; n];
        x[3] = 1.0;
        let olen = unsafe { gaussogram_output_len(h) };
        let mut out = vec![0.0f64; 2 * olen];
        let st = unsafe { gaussogram_forward_real(h, x.as_ptr(), n, out.as_mut_ptr(), olen) };
        assert_eq!(st, STATUS_OK);
        (x, out)
    };
    for (id, top) in [(0, TopWindow::Wide), (1, TopWindow::Flat), (2, TopWindow::Gaussian)] {
        let engine = build(dyadic_dual_real_with(n, WindowKind::Gaussian, top, 1).unwrap());
        let mut h: *mut GaussogramHandle = ptr::null_mut();
        assert_eq!(unsafe { gaussogram_create(0, n, 0, id, &mut h) }, STATUS_OK);
        let (x, out) = impulse_forward(h);
        let mut expected = vec![Complex::new(0.0, 0.0); engine.output_len()];
        engine.forward(&x, &mut expected).unwrap();
        for (k, e) in expected.iter().enumerate() {
            assert_eq!((out[2 * k], out[2 * k + 1]), (e.re, e.im), "id={id}");
        }
        unsafe { gaussogram_destroy(h) };
    }
    let mut h: *mut GaussogramHandle = ptr::null_mut();
    // Unknown id, and a non-default id on a scheme without the option.
    assert_eq!(unsafe { gaussogram_create(0, n, 0, 3, &mut h) }, STATUS_UNSUPPORTED_OPTION);
    assert_eq!(unsafe { gaussogram_create(0, n, 0, -1, &mut h) }, STATUS_UNSUPPORTED_OPTION);
    for id in [1, 2] {
        assert_eq!(unsafe { gaussogram_create(1, n, 0, id, &mut h) }, STATUS_UNSUPPORTED_OPTION);
    }
    assert_eq!(unsafe { gaussogram_create(1, n, 0, 0, &mut h) }, STATUS_OK);
    unsafe { gaussogram_destroy(h) };
}
