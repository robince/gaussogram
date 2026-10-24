#!/usr/bin/env python
"""Generate golden regression vectors for the REAL schemes.

The complex scheme has C-reference parity vectors (complex_golden.txt, from
gen_golden.c against fst-uofc). The real schemes (`dyadic_real`,
`dyadic_dual_real`) are new in gaussogram, so no external reference exists;
their forward values are instead validated against an independent numpy
implementation of the mathematical definition
(python/tests/test_reference_impl.py). This script freezes those validated
outputs as regression vectors so any future numerical drift in the Rust core
is caught by `cargo test` (tests/transform.rs::real_golden_regression).

Run from the repo root with the built extension installed:

    python golden/gen_real_golden.py

The input signal MUST match `real_golden_input` in tests/transform.rs.
"""

import numpy as np

import gaussogram as g

CASES = [
    # (scheme, n, bands_per_octave)
    ("dyadic_real", 8, 1),
    ("dyadic_real", 32, 1),
    ("dyadic_real", 32, 2),
    ("dyadic_real", 256, 1),
    ("dyadic_real", 256, 4),
    ("dyadic_dual_real", 8, 1),
    ("dyadic_dual_real", 32, 1),
    ("dyadic_dual_real", 32, 2),
    ("dyadic_dual_real", 256, 1),
    ("dyadic_dual_real", 256, 4),
]


def make_input(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (
        np.sin(2 * np.pi * 2 * i / n)
        + 0.5 * np.cos(2 * np.pi * 5 * i / n)
        + 0.05 * i / n
    )


def main() -> None:
    lines = []
    for scheme, n, bpo in CASES:
        c = g.gft1d_real(make_input(n), scheme=scheme, bands_per_octave=bpo)
        lines.append(f"SCHEME {scheme} N {n} BPO {bpo} LEN {len(c)}")
        lines.extend(f"{float(v.real):.17g} {float(v.imag):.17g}" for v in c)
    out = "\n".join(lines) + "\n"
    with open("golden/real_golden.txt", "w") as f:
        f.write(out)
    print(f"wrote golden/real_golden.txt ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
