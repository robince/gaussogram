"""NumPy-friendly wrappers over the `gaussogram._native` Rust extension.

Keeps the legacy `pygft` function names (`gft1d`, `gft1d_real`, `partitions`,
`real_partitions`) so porting is mechanical, plus new helpers for the
`dyadic_dual_real` default scheme.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import _native

#: Reusable transform handle (builds FFT plans + scratch once, reuses across
#: calls). Prefer this over the free functions for tight loops. See the native
#: docstring for threading caveats (one instance per thread).
Gaussogram1d = _native.Gaussogram1d

__all__ = [
    "gft1d",
    "gft1d_real",
    "inverse_real",
    "partitions",
    "real_partitions",
    "scheme_bands",
    "output_len",
    "BandLayout",
    "Gaussogram1d",
    "to_grid",
]


def _require_1d(a, dtype, name: str) -> NDArray:
    """Validate that ``a`` is a 1-D C-contiguous ndarray of exactly ``dtype``.

    This layer intentionally does **not** copy or cast: a silent
    ``np.ascontiguousarray`` would hide allocations from the caller and break
    the zero-copy contract. If the array does not already match, the caller is
    told what is wrong and asked to convert explicitly.
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(a).__name__}")
    if a.dtype != dtype:
        raise TypeError(
            f"{name} must have dtype {np.dtype(dtype)}, got {a.dtype}; "
            f"convert explicitly with .astype({np.dtype(dtype)!r})"
        )
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got {a.ndim}D")
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{name} must be C-contiguous; call np.ascontiguousarray() explicitly"
        )
    return a


class BandLayout(NamedTuple):
    """Per-band geometry of a scheme (parallel arrays)."""

    src_lo: NDArray[np.int64]
    width: NDArray[np.int64]
    fcentre: NDArray[np.int64]
    out_off: NDArray[np.int64]

    @property
    def src_hi(self) -> NDArray[np.int64]:
        return self.src_lo + self.width


def gft1d_real(
    x,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    nyquist_flat_top: bool = False,
) -> NDArray[np.complex128]:
    """Forward GFT of a real signal.

    Default scheme `dyadic_dual_real` returns **N-1** complex coefficients
    (a deliberate breaking change from the legacy length-N contract). Pass
    ``scheme="dyadic_real"`` for the invertible length-``N/2+1`` baseline.
    """
    x = _require_1d(x, np.float64, "x")
    return _native.gft1d_real(x, scheme, window_type, nyquist_flat_top)


def gft1d(z, window_type: str = "gaussian") -> NDArray[np.complex128]:
    """Forward GFT of a complex signal (symmetric `dyadic_complex` scheme,
    output length N). Port of the legacy `pygft.gft1d`."""
    z = _require_1d(z, np.complex128, "z")
    return _native.gft1d(z, window_type)


def inverse_real(coeffs, window_type: str = "gaussian") -> NDArray[np.float64]:
    """Inverse of the invertible `dyadic_real` scheme (length-``N/2+1`` input)."""
    coeffs = _require_1d(coeffs, np.complex128, "coeffs")
    return _native.inverse_real(coeffs, window_type)


def partitions(n: int) -> NDArray[np.int32]:
    """Complex-scheme partition boundaries (legacy `gft_1dPartitions`)."""
    return _native.partitions(int(n))


def real_partitions(n: int) -> NDArray[np.int32]:
    """Legacy real-scheme partition boundaries (`gft_1dRealPartitions`)."""
    return _native.real_partitions(int(n))


def scheme_bands(
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    nyquist_flat_top: bool = False,
) -> BandLayout:
    """Band layout (src_lo, width, fcentre, out_off) for a scheme."""
    lo, width, fcentre, out_off = _native.scheme_bands(
        int(n), scheme, window_type, nyquist_flat_top
    )
    return BandLayout(lo, width, fcentre, out_off)


def output_len(
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    nyquist_flat_top: bool = False,
) -> int:
    """Packed output length for a scheme."""
    return int(_native.output_len(int(n), scheme, window_type, nyquist_flat_top))


def _blur_axis(a: NDArray[np.float64], sigma: float, axis: int) -> NDArray[np.float64]:
    """Edge-padded 1-D Gaussian convolution along one axis (numpy-only)."""
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    def conv1d(m: NDArray[np.float64]) -> NDArray[np.float64]:
        padded = np.pad(m, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    return np.apply_along_axis(conv1d, axis, a)


def _gaussian_blur(grid: NDArray[np.float64], sigma) -> NDArray[np.float64]:
    """Separable Gaussian blur. ``sigma`` is a scalar (both axes) or a
    ``(sigma_freq, sigma_time)`` pair, in grid cells."""
    if np.isscalar(sigma):
        sf = st = float(sigma)
    else:
        sf, st = float(sigma[0]), float(sigma[1])
    out = grid
    if sf > 0:
        out = _blur_axis(out, sf, axis=0)
    if st > 0:
        out = _blur_axis(out, st, axis=1)
    return out


def _interp_freq_grid(
    node_freqs: NDArray[np.float64],
    node_profiles: NDArray[np.float64],
    nfreq: int,
) -> NDArray[np.float64]:
    """Linearly interpolate band-centre profiles across the frequency axis.

    ``node_profiles`` is ``(M, N)``: one time-row per band, anchored at the
    band's centre frequency ``node_freqs`` (length ``M``, sorted ascending). For
    each output row ``r`` in ``[0, nfreq)`` we interpolate, column by column,
    between the two band centres that bracket ``r``. Rows outside the centre
    range are clamped to the nearest band (the np.interp endpoint convention).
    """
    x = np.arange(nfreq, dtype=np.float64)
    xp = node_freqs.astype(np.float64)
    m = xp.shape[0]
    if m == 1:
        return np.repeat(node_profiles, nfreq, axis=0)
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, m - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    denom = x1 - x0
    w = np.where(denom > 0, (x - x0) / denom, 0.0)
    w = np.clip(w, 0.0, 1.0)  # clamp rows below/above the centre range
    lo_prof = node_profiles[idx]
    hi_prof = node_profiles[idx + 1]
    return (1.0 - w)[:, None] * lo_prof + w[:, None] * hi_prof


def to_grid(
    coeffs,
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    nyquist_flat_top: bool = False,
    interp: str = "linear",
    interp_freq: str = "block",
    normalize: str = "none",
    smooth=None,
) -> NDArray[np.float64]:
    """Render packed GFT coefficients onto a ``(N/2+1, N)`` magnitude grid for
    display: frequency on the vertical axis, time on the horizontal.

    Each band carries ``width`` complex samples spanning the full signal
    duration. The band magnitude is first resampled to ``N`` time points
    (``interp``), then placed on the frequency axis (``interp_freq``).

    Parameters
    ----------
    interp:
        Time-axis resampling of each band's magnitude:
        ``"linear"`` (default) piecewise-linear interpolation between the band's
        ``width`` samples — smooth; ``"nearest"`` block/step sampling — shows
        the raw coefficient cells.
    interp_freq:
        Frequency-axis placement of each band's resampled time-row:
        ``"block"`` (default) copies the row into every frequency row the band
        covers (a flat block), and overlapping bands are combined by keeping the
        larger magnitude per cell — this matches the band footprints exactly.
        ``"linear"`` instead anchors each band's row at its **centre frequency**
        and linearly interpolates between adjacent band centres, giving a smooth
        frequency axis with no block seams. With ``dyadic_dual_real`` the offset
        B bands add extra centre nodes, so the interpolation is finer.
        ``"gauss"`` weights each band's row by its **actual Gaussian window**
        across frequency (peak at ``fcentre``, ``σ_f = fcentre / 2π``) and
        normalises across overlapping bands (partition of unity). Unlike
        ``"block"`` (which smears a band over its full ``±3σ`` storage support ≈
        1.58 octaves) and ``"linear"`` (which spreads it linearly to the
        neighbouring centres), ``"gauss"`` confines each band to its true
        effective bandwidth (FWHM ≈ 1/3 octave), so a tone renders as a compact
        blob at its frequency. Best matches the transform's real resolution for
        the overlapping dual scheme.
    normalize:
        Per-band brightness normalisation. A spectral line of fixed amplitude
        produces coefficients of magnitude ``∝ 1/width``, so wider (higher-
        frequency) bands render dimmer for the same energy — the contrast drifts
        across the frequency axis. ``"none"`` (default) shows raw ``|coeff|``;
        ``"width"`` multiplies each band's row by its width (L1): an equal-
        amplitude **tone** is equally bright in every band, but a flat-spectrum
        event (impulse / white noise) then tilts *up* ∝ width toward high
        frequency. ``"energy"`` multiplies by √width (L2): the energy-preserving
        compromise — neither tones nor broadband are perfectly flat, both tilt
        gently. There is no single choice flat for both, because a tone scales
        as 1/width and a flat spectrum as a constant (the classic L1-vs-L2
        wavelet/CQT normalisation trade).
    smooth:
        Optional Gaussian blur applied to the assembled grid, in grid cells.
        A scalar blurs both axes equally; a ``(sigma_freq, sigma_time)`` pair
        blurs per-axis. ``None`` (default) disables smoothing. This further
        softens whatever edges remain after ``interp``/``interp_freq``.
    """
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    layout = scheme_bands(n, scheme, window_type, nyquist_flat_top)
    nfreq = n // 2 + 1

    if interp not in ("linear", "nearest"):
        raise ValueError(f"interp must be 'linear' or 'nearest', got {interp!r}")
    if interp_freq not in ("linear", "block", "gauss"):
        raise ValueError(
            f"interp_freq must be 'linear', 'block', or 'gauss', got {interp_freq!r}"
        )
    if normalize not in ("none", "width", "energy"):
        raise ValueError(
            f"normalize must be 'none', 'width', or 'energy', got {normalize!r}"
        )

    target_times = np.arange(n)

    def band_row(off: int, w: int) -> NDArray[np.float64]:
        """Time-resample one band's magnitude to all ``N`` columns."""
        mag = np.abs(coeffs[off : off + w])
        # |coeff| ∝ 1/width for a fixed-amplitude line; ×width (L1) equalises tone
        # brightness across bands, ×√width (L2) is the energy-preserving middle.
        if normalize == "width":
            mag = mag * w
        elif normalize == "energy":
            mag = mag * np.sqrt(w)
        if w == 1:
            return np.full(n, mag[0])
        band_times = np.linspace(0, n - 1, w)
        if interp == "nearest":
            idx = np.abs(target_times[:, None] - band_times[None, :]).argmin(axis=1)
            return mag[idx]
        return np.interp(target_times, band_times, mag)

    if interp_freq == "block":
        grid = np.zeros((nfreq, n), dtype=np.float64)
        for lo, w, off in zip(layout.src_lo, layout.width, layout.out_off):
            lo, w, off = int(lo), int(w), int(off)
            if lo >= nfreq:
                continue
            row = band_row(off, w)
            hi = min(lo + w, nfreq)
            grid[lo:hi, :] = np.maximum(grid[lo:hi, :], row[None, :])
    elif interp_freq == "linear":  # anchor each band at its centre, interpolate
        nodes: dict[int, NDArray[np.float64]] = {}
        for w, off, fc in zip(layout.width, layout.out_off, layout.fcentre):
            w, off, fc = int(w), int(off), int(fc)
            if fc >= nfreq:
                continue
            row = band_row(off, w)
            # Two bands can share a centre (e.g. degenerate widths): keep max.
            nodes[fc] = np.maximum(nodes[fc], row) if fc in nodes else row
        node_freqs = np.array(sorted(nodes), dtype=np.float64)
        node_profiles = np.stack([nodes[int(f)] for f in node_freqs])
        grid = _interp_freq_grid(node_freqs, node_profiles, nfreq)
    else:  # "gauss": weight each band by its window profile, normalise overlaps
        grid = np.zeros((nfreq, n), dtype=np.float64)
        weight_sum = np.zeros(nfreq, dtype=np.float64)
        for lo, w, off, fc in zip(
            layout.src_lo, layout.width, layout.out_off, layout.fcentre
        ):
            lo, w, off, fc = int(lo), int(w), int(off), int(fc)
            if lo >= nfreq:
                continue
            row = band_row(off, w)
            hi = min(lo + w, nfreq)
            rows = np.arange(lo, hi)
            if w == 1 or fc == 0:
                # DC / width-1 bands carry no taper: flat over their support.
                wj = np.ones(hi - lo, dtype=np.float64)
            else:
                sigma = fc / (2.0 * np.pi)  # freq-domain σ of the band Gaussian
                wj = np.exp(-((rows - fc) ** 2) / (2.0 * sigma**2))
                # Mirror the forward transform's Nyquist flat-top on the top band.
                if nyquist_flat_top and lo + w >= nfreq:
                    wj[rows >= fc] = 1.0
            grid[lo:hi, :] += wj[:, None] * row[None, :]
            weight_sum[lo:hi] += wj
        weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        grid /= weight_sum[:, None]

    if smooth is not None:
        grid = _gaussian_blur(grid, smooth)
    return grid
