//! Exercise the C ABI from Rust: persistent handles must reuse their scratch
//! without leaking state between calls and match the core engine exactly.

use std::ptr;

use gaussogram_core::{build, dyadic_dual_real, dyadic_real, STATUS_OK};
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
            if scheme_id == 1 {
                let mut recon = vec![0.0f64; n];
                let st = unsafe {
                    gaussogram_inverse_real(h, out.as_ptr(), olen, recon.as_mut_ptr(), n)
                };
                assert_eq!(st, STATUS_OK);
                for (a, b) in x.iter().zip(&recon) {
                    assert!((a - b).abs() < 1e-10);
                }
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
