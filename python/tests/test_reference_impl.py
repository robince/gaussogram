"""Independent numpy reference implementation of the forward GFT.

Round-trip tests alone cannot catch a consistent forward-side error: a
mis-scaled window screen, for example, would cancel exactly in the inverse and
every round-trip test would still pass. This file therefore recomputes the
forward coefficients *from the mathematical definition* — the window built
directly from the spec formula (RUST_PORT_SPEC.md / window.rs docstrings), the
spectrum and per-band inversions via numpy's FFTs — and requires the native
output to match to near machine precision, for both real schemes, several
signal classes, all bands_per_octave values, and both window types.
"""

import numpy as np
import pytest

import gaussogram as g


def freq_gaussian(n: int, fcentre: int) -> np.ndarray:
    """Real frequency-domain Gaussian: DFT of the unit-sum time Gaussian
    ``g[s] ∝ exp(-((s/N - 1/2)^2 fc^2 / 2))`` circularly shifted so its peak
    (sample N/2) lands on sample 0."""
    s = np.arange(n, dtype=np.float64)
    fc = float(abs(fcentre))
    gt = fc / np.sqrt(2.0 * np.pi) * np.exp(-((s / n - 0.5) ** 2) * fc * fc / 2.0)
    total = gt.sum()
    if total != 0.0:
        gt = gt / total
    rolled = np.roll(gt, -(n // 2))  # peak to sample 0
    return np.fft.fft(rolled).real


def band_screen(n: int, fcentre: int, w: int, window_type: str) -> np.ndarray:
    """Window screen for a band of width ``w``: the frequency Gaussian sampled
    so its peak sits at local index ``w//2`` (source bin ``lo + w//2``).
    Width-1 bands are exact (screen 1.0); box windows are all-ones."""
    if window_type == "box":
        return np.ones(w)
    if w == 1:
        return np.array([1.0])
    f = freq_gaussian(n, fcentre)
    k = np.arange(w)
    return f[(k - w // 2) % n]


def reference_forward(
    x: np.ndarray, scheme: str, window_type: str, bands_per_octave: int
) -> np.ndarray:
    """Forward GFT from the definition: one rfft, then per band multiply the
    spectrum slice by the screen and inverse-DFT it (numpy ifft carries the
    1/w normalisation the fast algorithm applies explicitly)."""
    n = len(x)
    spectrum = np.fft.rfft(x)  # length N/2+1; all real-scheme bands fit inside
    lay = g.scheme_bands(
        n, scheme=scheme, window_type=window_type, bands_per_octave=bands_per_octave
    )
    total = int((lay.out_off + lay.width).max())
    out = np.zeros(total, dtype=np.complex128)
    for lo, w, fc, off in zip(lay.src_lo, lay.width, lay.fcentre, lay.out_off):
        lo, w, fc, off = int(lo), int(w), int(fc), int(off)
        windowed = spectrum[lo : lo + w] * band_screen(n, fc, w, window_type)
        out[off : off + w] = np.fft.ifft(windowed)
    return out


def signals(n: int) -> dict:
    rng = np.random.default_rng(12345)
    i = np.arange(n, dtype=np.float64)
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    return {
        "impulse": impulse,
        "tone_on_join": np.cos(2 * np.pi * (n // 8) * i / n),
        "mixture": (
            np.sin(2 * np.pi * 2 * i / n)
            + 0.5 * np.cos(2 * np.pi * 5 * i / n)
            + 0.05 * i / n
        ),
        "noise": rng.standard_normal(n),
    }


@pytest.mark.parametrize("scheme", ["dyadic_real", "dyadic_dual_real"])
@pytest.mark.parametrize("window_type", ["gaussian", "box"])
@pytest.mark.parametrize("bands_per_octave", [1, 2, 4])
@pytest.mark.parametrize("n", [8, 32, 256])
def test_forward_matches_reference(scheme, window_type, bands_per_octave, n):
    for name, x in signals(n).items():
        got = g.gft1d_real(
            np.ascontiguousarray(x),
            scheme=scheme,
            window_type=window_type,
            bands_per_octave=bands_per_octave,
        )
        want = reference_forward(x, scheme, window_type, bands_per_octave)
        assert got.shape == want.shape
        err = np.max(np.abs(got - want))
        assert err < 1e-12, (
            f"{scheme}/{window_type}/bpo={bands_per_octave}/N={n}/{name}: "
            f"max |native - reference| = {err:.3e}"
        )
