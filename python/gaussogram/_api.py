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

__all__ = [
    "gft1d",
    "gft1d_real",
    "inverse_real",
    "partitions",
    "real_partitions",
    "scheme_bands",
    "output_len",
    "BandLayout",
    "to_grid",
]


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
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("input must be a 1D vector")
    return _native.gft1d_real(x, scheme, window_type, nyquist_flat_top)


def gft1d(z, window_type: str = "gaussian") -> NDArray[np.complex128]:
    """Forward GFT of a complex signal (symmetric `dyadic_complex` scheme,
    output length N). Port of the legacy `pygft.gft1d`."""
    z = np.ascontiguousarray(z, dtype=np.complex128)
    if z.ndim != 1:
        raise ValueError("input must be a 1D vector")
    return _native.gft1d(z, window_type)


def inverse_real(coeffs, window_type: str = "gaussian") -> NDArray[np.float64]:
    """Inverse of the invertible `dyadic_real` scheme (length-``N/2+1`` input)."""
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError("input must be a 1D vector")
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


def to_grid(
    coeffs,
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    nyquist_flat_top: bool = False,
) -> NDArray[np.float64]:
    """Render packed GFT coefficients onto a ``(N/2+1, N)`` magnitude grid for
    display: frequency on the vertical axis, time on the horizontal.

    Each band carries ``width`` complex samples spanning the full signal
    duration; they are nearest-neighbour interpolated to ``N`` time points and
    written to the rows the band covers. For the overlapping dual scheme, the
    larger magnitude wins per (freq, time) cell.
    """
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    layout = scheme_bands(n, scheme, window_type, nyquist_flat_top)
    nfreq = n // 2 + 1
    grid = np.zeros((nfreq, n), dtype=np.float64)

    target_times = np.arange(n)
    for lo, w, off in zip(layout.src_lo, layout.width, layout.out_off):
        lo = int(lo)
        w = int(w)
        off = int(off)
        band = coeffs[off : off + w]
        if w == 1:
            row = np.full(n, np.abs(band[0]))
        else:
            band_times = np.linspace(0, n - 1, w)
            nearest = np.abs(target_times[:, None] - band_times[None, :]).argmin(axis=1)
            row = np.abs(band[nearest])
        hi = min(lo + w, nfreq)
        if lo >= nfreq:
            continue
        grid[lo:hi, :] = np.maximum(grid[lo:hi, :], row[None, :])
    return grid
