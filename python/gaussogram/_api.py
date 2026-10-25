"""NumPy-friendly wrappers over the `gaussogram._native` Rust extension.

Keeps the legacy `pygft` function names (`gft1d`, `gft1d_real`, `partitions`,
`real_partitions`) so porting is mechanical, plus new helpers for the
`dyadic_dual_real` default scheme.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import _native

#: Gain of the ``"wide"`` top-band Gaussian at its band edges. Mirrors
#: ``TOP_EDGE_GAIN`` in the Rust core (``window.rs``).
_TOP_EDGE_GAIN = 0.63

_TOP_WINDOWS = ("wide", "gaussian", "flat")


def _resolve_top_window(top_window, nyquist_flat_top) -> str:
    """Resolve ``top_window`` and the deprecated ``nyquist_flat_top`` alias to
    one of ``"wide"``, ``"gaussian"``, ``"flat"``.

    ``top_window=None`` means the default, ``"wide"``. A bool in the
    ``top_window`` slot is the old positional ``nyquist_flat_top``.
    """
    if isinstance(top_window, (bool, np.bool_)):
        if nyquist_flat_top is not None:
            raise TypeError("nyquist_flat_top given twice (positionally and by keyword)")
        top_window, nyquist_flat_top = None, bool(top_window)
    if nyquist_flat_top is not None:
        # False means "no flat top" and so the default, as before; but the
        # default top window changed, so say how to get the old output.
        if nyquist_flat_top:
            msg = "nyquist_flat_top is deprecated; use top_window='flat'"
        else:
            msg = (
                "nyquist_flat_top is deprecated. nyquist_flat_top=False now "
                "gives the default top_window='wide'; pass top_window='gaussian' "
                "for the previous output"
            )
        # Called from a public wrapper, so the user's frame is two up.
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if nyquist_flat_top:
            if top_window not in (None, "flat"):
                raise ValueError(
                    f"nyquist_flat_top=True conflicts with top_window={top_window!r}"
                )
            top_window = "flat"
        elif top_window == "flat":
            raise ValueError("nyquist_flat_top=False conflicts with top_window='flat'")
    if top_window is None:
        return "wide"
    if top_window not in _TOP_WINDOWS:
        raise ValueError(
            f"top_window must be 'wide', 'gaussian' or 'flat' (got {top_window!r})"
        )
    return top_window


class Gaussogram1d(_native.Gaussogram1d):
    """Reusable transform handle (builds FFT plans + scratch once, reuses
    across calls). Prefer this over the free functions for tight loops.

    ``Gaussogram1d(n, scheme="dyadic_dual_real", window_type="gaussian",
    top_window=None, bands_per_octave=1)``; ``top_window`` is as in
    :func:`gft1d_real`. One instance is **not** safe to call concurrently from
    several Python threads: create one per thread.
    """

    def __new__(
        cls,
        n: int,
        scheme: str = "dyadic_dual_real",
        window_type: str = "gaussian",
        top_window: str | None = None,
        bands_per_octave: int = 1,
        *,
        nyquist_flat_top: bool | None = None,
    ):
        top = _resolve_top_window(top_window, nyquist_flat_top)
        return super().__new__(cls, int(n), scheme, window_type, top, int(bands_per_octave))

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
    "coefficient_geometry",
    "coefficient_adjacency",
    "to_sparse_adjacency",
    "CoeffGeometry",
    "CoeffAdjacency",
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
    top_window: str | None = None,
    bands_per_octave: int = 1,
    *,
    nyquist_flat_top: bool | None = None,
) -> NDArray[np.complex128]:
    """Forward GFT of a real signal.

    Default scheme `dyadic_dual_real` returns **N-1** complex coefficients
    (a deliberate breaking change from the legacy length-N contract). Pass
    ``scheme="dyadic_real"`` for the critically sampled length-``N/2+1`` baseline.

    ``bands_per_octave`` (a power of two) subdivides each dyadic octave into that
    many frequency sub-bands, trading time resolution for frequency resolution.
    It does **not** change the output length — ``dyadic_real`` stays ``N/2+1`` and
    ``dyadic_dual_real`` stays ``N-1`` for every value — it only reshapes the
    tiling. Running a few values (e.g. 1, 2, 4) gives complementary
    multiresolution views for far less data than a dense spectrogram.

    ``top_window`` shapes the top band of ``dyadic_dual_real`` (tiling A's band
    just below Nyquist, whose upper half no tiling-B band covers). ``None``
    means the default:

    * ``"wide"`` (default): a Gaussian with the usual peak, widened so its gain
      at both band edges is 0.63. This removes the dead zone below Nyquist
      (frame condition number ~3.5 instead of ~84 at ``bands_per_octave=1``)
      with almost no ringing in the band's time response. At
      ``bands_per_octave >= 2`` the plain Gaussian is already as wide, so
      nothing changes.
    * ``"gaussian"``: the plain Gaussian every other band uses.
    * ``"flat"``: the plain Gaussian held at its peak value up to Nyquist.
      Conditions the frame too, but the step where the band's FFT wraps makes
      its time response ring.

    Other schemes accept only the default. Box windows ignore it.
    ``nyquist_flat_top=True`` is a deprecated alias for ``top_window="flat"``.
    """
    x = _require_1d(x, np.float64, "x")
    top = _resolve_top_window(top_window, nyquist_flat_top)
    return _native.gft1d_real(x, scheme, window_type, top, int(bands_per_octave))


def gft1d(z, window_type: str = "gaussian") -> NDArray[np.complex128]:
    """Forward GFT of a complex signal (symmetric `dyadic_complex` scheme,
    output length N). Port of the legacy `pygft.gft1d`."""
    z = _require_1d(z, np.complex128, "z")
    return _native.gft1d(z, window_type)


def inverse_real(
    coeffs,
    window_type: str = "gaussian",
    bands_per_octave: int = 1,
    scheme: str | None = None,
    top_window: str | None = None,
    *,
    nyquist_flat_top: bool | None = None,
) -> NDArray[np.float64]:
    """Inverse of a real-input scheme.

    * ``dyadic_real`` (length ``N/2+1``): exact per-band inverse.
    * ``dyadic_dual_real`` (length ``N-1``): least-squares frame inverse.
      Exact for unmodified coefficients; for modified ones (thresholded,
      masked, denoised) it returns the signal whose coefficients are closest
      to them in signal energy (each band's coefficients weighted by
      ``sqrt(width)``, undoing the stored ``1/width`` scaling), using both
      tilings, so it is far less sensitive to edits near octave joins than
      inverting tiling A alone.

    ``scheme=None`` infers the scheme from the length: ``N/2+1`` is always
    ``2^k+1`` and ``N-1`` is always ``2^k-1``, which never coincide.

    .. warning::
        ``bands_per_octave``, ``window_type`` and ``top_window`` must match
        the values passed to the forward transform. A mismatch **cannot be
        detected** — the packed length is the same for every
        ``bands_per_octave`` and ``top_window`` — and silently reconstructs the
        wrong signal.
    """
    coeffs = _require_1d(coeffs, np.complex128, "coeffs")
    top = _resolve_top_window(top_window, nyquist_flat_top)
    return _native.inverse_real(coeffs, window_type, int(bands_per_octave), scheme, top)


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
    top_window: str | None = None,
    bands_per_octave: int = 1,
    *,
    nyquist_flat_top: bool | None = None,
) -> BandLayout:
    """Band layout (src_lo, width, fcentre, out_off) for a scheme."""
    top = _resolve_top_window(top_window, nyquist_flat_top)
    lo, width, fcentre, out_off = _native.scheme_bands(
        int(n), scheme, window_type, top, int(bands_per_octave)
    )
    return BandLayout(lo, width, fcentre, out_off)


def output_len(
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    top_window: str | None = None,
    bands_per_octave: int = 1,
    *,
    nyquist_flat_top: bool | None = None,
) -> int:
    """Packed output length for a scheme (invariant to ``bands_per_octave``)."""
    top = _resolve_top_window(top_window, nyquist_flat_top)
    return int(
        _native.output_len(int(n), scheme, window_type, top, int(bands_per_octave))
    )


#: Target number of float64 elements in a working block of ``to_grid`` / the
#: blur (~8 MB), bounding temporaries independently of the grid size.
_BLOCK_ELEMS = 1 << 20


def _blur_axis_(a: NDArray[np.float64], sigma: float, axis: int) -> NDArray[np.float64]:
    """In place: edge-padded 1-D Gaussian convolution along one axis of a
    writable 2-D float64 array.

    Vectorised as a weighted sum of shifted slices, processed in blocks along
    the other axis: each block is copied (padded) before being overwritten, so
    only block-sized temporaries are needed. The kernel is symmetric, so this
    equals ``np.convolve(..., mode="valid")`` on the edge-padded data.
    """
    # Round half up (not Python's half-to-even) to match MATLAB's round().
    radius = max(1, int(np.floor(3.0 * sigma + 0.5)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    m = np.moveaxis(a, axis, 0)          # blur along axis 0 of this view
    length, other = m.shape
    step = max(1, _BLOCK_ELEMS // max(1, length + 2 * radius))
    for c0 in range(0, other, step):
        c1 = min(c0 + step, other)
        padded = np.pad(m[:, c0:c1], ((radius, radius), (0, 0)), mode="edge")
        acc = m[:, c0:c1]
        np.multiply(padded[0:length], kernel[0], out=acc)
        tmp = np.empty_like(acc)
        for k in range(1, len(kernel)):
            np.multiply(padded[k : k + length], kernel[k], out=tmp)
            acc += tmp
    return a


def _gaussian_blur_(grid: NDArray[np.float64], sigma) -> NDArray[np.float64]:
    """Separable Gaussian blur, in place. ``sigma`` is a scalar (both axes) or
    a ``(sigma_freq, sigma_time)`` pair, in grid cells."""
    if np.isscalar(sigma):
        sf = st = float(sigma)
    else:
        sf, st = float(sigma[0]), float(sigma[1])
    if sf > 0:
        _blur_axis_(grid, sf, axis=0)
    if st > 0:
        _blur_axis_(grid, st, axis=1)
    return grid


def _interp_freq_grid(
    node_freqs: NDArray[np.float64],
    node_profiles: NDArray[np.float64],
    row_freqs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linearly interpolate band-centre profiles across the frequency axis.

    ``node_profiles`` is ``(M, T)``: one time-row per band, anchored at the
    band's centre frequency ``node_freqs`` (length ``M``, sorted ascending). For
    each output row (at frequency ``row_freqs[r]``, in bins) we interpolate,
    column by column, between the two band centres that bracket it. Rows
    outside the centre range are clamped to the nearest band (the np.interp
    endpoint convention). Rows are filled in blocks to bound temporaries.
    """
    x = np.asarray(row_freqs, dtype=np.float64)
    nrows = x.shape[0]
    xp = node_freqs.astype(np.float64)
    m = xp.shape[0]
    if m == 1:
        return np.repeat(node_profiles, nrows, axis=0)
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, m - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    denom = x1 - x0
    w = np.where(denom > 0, (x - x0) / np.where(denom > 0, denom, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)  # clamp rows below/above the centre range
    ncols = node_profiles.shape[1]
    grid = np.empty((nrows, ncols), dtype=np.float64)
    step = max(1, _BLOCK_ELEMS // max(1, ncols))
    for r0 in range(0, nrows, step):
        r1 = min(r0 + step, nrows)
        blk = grid[r0:r1]
        np.multiply(node_profiles[idx[r0:r1]], (1.0 - w[r0:r1])[:, None], out=blk)
        blk += w[r0:r1, None] * node_profiles[idx[r0:r1] + 1]
    return grid


def _nearest_sample_index(ntime: int, w: int) -> NDArray[np.int64]:
    """Index of the circularly nearest of ``w`` band samples (at times
    ``j*N/w``) for each of ``ntime`` columns (at times ``c*N/ntime``).

    Column ``c`` sits at ``k = c*w/ntime`` in sample units, so this is done in
    exact integer arithmetic: float rounding of ``k`` would otherwise break
    exact ties inconsistently. Ties go to the lower index, so the wrap-around
    tie between samples ``w-1`` and ``0`` goes to 0. Computed directly rather
    than via an ``(ntime, w)`` distance matrix.
    """
    c = np.arange(ntime, dtype=np.int64)
    num = 2 * c * w - ntime            # 2*ntime*(k - 1/2)
    idx = -((-num) // (2 * ntime))     # ceil(k - 1/2): ties round down
    idx[2 * c * w == (2 * w - 1) * ntime] = 0
    return idx % w


def to_grid(
    coeffs,
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    top_window: str | None = None,
    interp: str = "linear",
    interp_freq: str = "block",
    normalize: str = "none",
    smooth=None,
    bands_per_octave: int = 1,
    n_freq: int | None = None,
    n_time: int | None = None,
    *,
    nyquist_flat_top: bool | None = None,
) -> NDArray[np.float64]:
    """Render packed GFT coefficients onto a ``(n_freq, n_time)`` magnitude
    grid for display: frequency on the vertical axis, time on the horizontal.
    The default is the full ``(N/2+1, N)`` resolution. Real schemes only: the
    grid is the positive half-spectrum, so ``dyadic_complex`` is rejected.

    Each band carries ``width`` complex samples spanning the full signal
    duration. The band magnitude is first resampled to ``n_time`` time points
    (``interp``), then placed on the frequency axis (``interp_freq``).

    Parameters
    ----------
    n_freq, n_time:
        Output size. Rows sit at ``n_freq`` evenly spaced frequencies from 0 to
        ``N/2`` bins (inclusive) and columns at times ``k * N / n_time``
        samples. The full-resolution grid is ``(N/2+1) * N`` float64 values
        (16 GiB at N = 2^16), so pass a screen-sized shape for long signals.
        With fewer rows than bins, a band narrower than the row spacing may
        cover no row in ``"block"`` / ``"gauss"`` mode.
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
        the overlapping dual scheme. The top band of ``dyadic_dual_real``
        follows ``top_window`` (wider for ``"wide"``, flat above its peak for
        ``"flat"``), as in the forward transform.
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
    if scheme == "dyadic_complex":
        # The grid is the positive half-spectrum 0..N/2; the complex scheme's
        # negative-frequency bands have no rows there.
        raise ValueError(
            "to_grid renders the positive half-spectrum of a real scheme; "
            "'dyadic_complex' is not supported"
        )
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    top = _resolve_top_window(top_window, nyquist_flat_top)
    layout = scheme_bands(n, scheme, window_type, top, bands_per_octave)
    expected = int((layout.out_off + layout.width).max())
    if coeffs.ndim != 1 or len(coeffs) != expected:
        raise ValueError(
            f"coeffs has length {coeffs.size} but scheme {scheme!r} with n={n} "
            f"packs {expected} coefficients; check n and scheme match the "
            f"forward transform"
        )
    nfreq = n // 2 + 1 if n_freq is None else int(n_freq)
    ntime = n if n_time is None else int(n_time)
    if nfreq < 2 or ntime < 1:
        raise ValueError(f"need n_freq >= 2 and n_time >= 1, got ({nfreq}, {ntime})")
    # Row frequencies in bins; the default is exactly 0, 1, ..., N/2.
    row_freqs = np.linspace(0.0, n / 2, nfreq)

    def band_rows(lo: int, hi: int) -> tuple[int, int]:
        """Rows whose frequency lies in the band's support ``[lo, hi)``."""
        return (int(np.searchsorted(row_freqs, lo, side="left")),
                int(np.searchsorted(row_freqs, hi, side="left")))

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

    target_times = np.arange(ntime) * (n / ntime)

    def band_row(off: int, w: int) -> NDArray[np.float64]:
        """Time-resample one band's magnitude to the ``n_time`` columns."""
        mag = np.abs(coeffs[off : off + w])
        # |coeff| ∝ 1/width for a fixed-amplitude line; ×width (L1) equalises tone
        # brightness across bands, ×√width (L2) is the energy-preserving middle.
        if normalize == "width":
            mag = mag * w
        elif normalize == "energy":
            mag = mag * np.sqrt(w)
        if w == 1:
            return np.full(ntime, mag[0])
        # The unshifted per-band inverse FFT puts sample j at time ``j·n/w``
        # (matching ``coefficient_geometry``). The transform is circular, so
        # resample periodically: the last sample wraps back to sample 0 at t=n.
        if interp == "nearest":
            return mag[_nearest_sample_index(ntime, w)]
        band_times = np.arange(w) * (n / w)
        return np.interp(target_times, band_times, mag, period=n)

    if interp_freq == "block":
        grid = np.zeros((nfreq, ntime), dtype=np.float64)
        for lo, w, off in zip(layout.src_lo, layout.width, layout.out_off):
            lo, w, off = int(lo), int(w), int(off)
            r0, r1 = band_rows(lo, lo + w)
            if r0 >= r1:
                continue
            row = band_row(off, w)
            blk = grid[r0:r1]
            np.maximum(blk, row[None, :], out=blk)
    elif interp_freq == "linear":  # anchor each band at its centre, interpolate
        nodes: dict[int, NDArray[np.float64]] = {}
        for w, off, fc in zip(layout.width, layout.out_off, layout.fcentre):
            w, off, fc = int(w), int(off), int(fc)
            if fc > n // 2:
                continue
            row = band_row(off, w)
            # Two bands can share a centre (e.g. degenerate widths): keep max.
            nodes[fc] = np.maximum(nodes[fc], row) if fc in nodes else row
        node_freqs = np.array(sorted(nodes), dtype=np.float64)
        node_profiles = np.stack([nodes[int(f)] for f in node_freqs])
        grid = _interp_freq_grid(node_freqs, node_profiles, row_freqs)
    else:  # "gauss": weight each band by its window profile, normalise overlaps
        grid = np.zeros((nfreq, ntime), dtype=np.float64)
        weight_sum = np.zeros(nfreq, dtype=np.float64)
        step = max(1, _BLOCK_ELEMS // ntime)
        for lo, w, off, fc in zip(
            layout.src_lo, layout.width, layout.out_off, layout.fcentre
        ):
            lo, w, off, fc = int(lo), int(w), int(off), int(fc)
            r0, r1 = band_rows(lo, lo + w)
            if r0 >= r1:
                continue
            row = band_row(off, w)
            rows = row_freqs[r0:r1]
            if w == 1 or fc == 0:
                # DC / width-1 bands carry no taper: flat over their support.
                wj = np.ones(r1 - r0, dtype=np.float64)
            else:
                sigma = fc / (2.0 * np.pi)  # freq-domain σ of the band Gaussian
                # Mirror the forward transform's top window on the dual scheme's
                # top band (the one ending just below the width-1 Nyquist band).
                # Box windows ignore top_window in the transform, so here too.
                is_top = (
                    scheme == "dyadic_dual_real"
                    and window_type == "gaussian"
                    and lo + w == n // 2
                )
                if is_top and top == "wide":
                    sigma_wide = (w / 2.0) / np.sqrt(2.0 * np.log(1.0 / _TOP_EDGE_GAIN))
                    sigma = max(sigma, sigma_wide)
                wj = np.exp(-((rows - fc) ** 2) / (2.0 * sigma**2))
                if is_top and top == "flat":
                    wj[rows >= fc] = 1.0
            # Accumulate in row blocks so the outer product stays small.
            for b0 in range(r0, r1, step):
                b1 = min(b0 + step, r1)
                grid[b0:b1] += wj[b0 - r0 : b1 - r0, None] * row[None, :]
            weight_sum[r0:r1] += wj
        weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        grid /= weight_sum[:, None]

    if smooth is not None:
        grid = _gaussian_blur_(grid, smooth)  # grid is ours: blur in place
    return grid


# ---- coefficient geometry & adjacency (for downstream models) -------------
#
# These describe *where each coefficient lives* in the time-frequency plane and
# *how coefficients relate*, so a model can operate natively on the packed ~N
# vector (transformer positional features / continuous convolutions / GNNs /
# graph-Laplacian smoothness / structured-sparsity groups) WITHOUT densifying it
# to an N x N/2 grid. Both are signal-INDEPENDENT — they depend only on
# (n, scheme, window_type, top_window) — so compute once and reuse. This is
# the complement of `to_grid`, which resamples values for display; here the
# values are untouched and only the basis geometry is exposed.


def _band_freq_and_tiling(lay: BandLayout, n: int, scheme: str):
    """Per-band signed centre frequency and tiling id.

    Real schemes pack tiling A first (``N/2+1`` coefficients), then tiling B.
    ``dyadic_complex`` is a single tiling over the full spectrum: its bands from
    bin ``N/2`` up are negative frequencies, whose ``fcentre`` is stored as a
    magnitude.
    """
    if scheme == "dyadic_complex":
        freqs = [-int(fc) if int(lo) >= n // 2 else int(fc)
                 for lo, fc in zip(lay.src_lo, lay.fcentre)]
        return freqs, [0] * len(freqs)
    a_len = n // 2 + 1
    return ([int(fc) for fc in lay.fcentre],
            [0 if int(off) < a_len else 1 for off in lay.out_off])


class CoeffGeometry(NamedTuple):
    """Per-coefficient time-frequency geometry (parallel arrays, length =
    output_len). Each coefficient owns a tile of size ``dt`` (time) x ``df``
    (frequency); ``dt * df == n`` for every tile."""

    time: NDArray[np.float64]   #: tile centre time (samples) = j·n/w, in [0, n)
    freq: NDArray[np.float64]   #: tile centre frequency (bins; < 0 for the
                                #: negative-frequency bands of dyadic_complex)
    dt: NDArray[np.float64]     #: time extent (samples) = n / width
    df: NDArray[np.float64]     #: frequency extent (bins) = width
    logf: NDArray[np.float64]   #: log2(max(|freq|, 0.5)); DC = -1
    level: NDArray[np.int64]    #: octave index = floor(log2(max(|centre freq|, 1)))
    sub: NDArray[np.int64]      #: within-octave sub-band rank (low->high frequency)
    tiling: NDArray[np.int64]   #: 0 = tiling A, 1 = tiling B (dual scheme)
    band: NDArray[np.int64]     #: index of the band each coefficient belongs to


def coefficient_geometry(
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    top_window: str | None = None,
    bands_per_octave: int = 1,
    *,
    nyquist_flat_top: bool | None = None,
) -> CoeffGeometry:
    """Geometry of every packed coefficient as points in the time-frequency
    plane (see :class:`CoeffGeometry`). Use as positional features for a
    transformer, or as the coordinate space for a continuous-kernel convolution.

    With ``bands_per_octave > 1`` each octave holds several sub-bands; ``level``
    is the octave index and ``sub`` is the rank within the octave, so the two
    together locate a coefficient on the (log-frequency, sub-band) lattice.
    """
    top = _resolve_top_window(top_window, nyquist_flat_top)
    lay = scheme_bands(n, scheme, window_type, top, bands_per_octave)
    total = int((lay.out_off + lay.width).max())
    time = np.zeros(total)
    freq = np.zeros(total)
    dt = np.zeros(total)
    df = np.zeros(total)
    level = np.zeros(total, dtype=np.int64)
    sub = np.zeros(total, dtype=np.int64)
    tiling = np.zeros(total, dtype=np.int64)
    band = np.zeros(total, dtype=np.int64)

    # Per-band octave index and tiling, then the within-octave rank: bands that
    # share a (tiling, sign, octave) are ranked low->high by source frequency.
    nb = len(lay.width)
    b_freq, b_tiling = _band_freq_and_tiling(lay, n, scheme)
    b_octave = [int(np.floor(np.log2(max(abs(f), 1)))) for f in b_freq]
    b_sub = [0] * nb
    groups: dict[tuple[int, bool, int], list[int]] = {}
    for bi in range(nb):
        key = (b_tiling[bi], b_freq[bi] < 0, b_octave[bi])
        groups.setdefault(key, []).append(bi)
    for members in groups.values():
        for rank, bi in enumerate(sorted(members, key=lambda j: int(lay.src_lo[j]))):
            b_sub[bi] = rank

    for bi, (w, off) in enumerate(zip(lay.width, lay.out_off)):
        w, off = int(w), int(off)
        j = np.arange(w)
        idx = off + j
        time[idx] = j * (n / w)
        freq[idx] = b_freq[bi]
        dt[idx] = n / w
        df[idx] = w
        level[idx] = b_octave[bi]
        sub[idx] = b_sub[bi]
        tiling[idx] = b_tiling[bi]
        band[idx] = bi
    logf = np.log2(np.maximum(np.abs(freq), 0.5))
    return CoeffGeometry(time, freq, dt, df, logf, level, sub, tiling, band)


class CoeffAdjacency(NamedTuple):
    """Sparse neighbourhood graph over the packed coefficients (undirected;
    each edge stored once with ``i < j``). Symmetrise for message passing; the
    unweighted graph Laplacian encodes a smoothness prior over the irregular
    tiling, and ``edge_type`` lets you weight relations differently."""

    edge_index: NDArray[np.int64]    #: (2, E) endpoint indices, i < j
    edge_type: NDArray[np.int64]     #: 0 = time, 1 = band, 2 = dual (see type_names)
    edge_weight: NDArray[np.float64] #: (E,) structural weights (1.0 by default)
    type_names: tuple = ("time", "band", "dual")


def _connect_by_time(b1: dict, b2: dict, circular: bool):
    """Time-aligned cross-band edges: link each coefficient of the finer-time
    band to every coefficient of the coarser-time band whose time tile it
    overlaps (one, or two when it straddles a coarse tile boundary).

    Sample ``j`` of a width-``w`` band is the centre of the tile
    ``[(j - 1/2)·n/w, (j + 1/2)·n/w)``. In units of ``n / (2·wf·wc)`` the fine
    tile ``j`` spans ``((2j-1)·wc, (2j+1)·wc)`` and the coarse tile ``k`` spans
    ``((2k-1)·wf, (2k+1)·wf)``, so the overlapping ``k`` follow in exact integer
    arithmetic. A fine tile is never wider than a coarse one, so at most two
    match. Past the last coarse tile, ``circular`` wraps to coarse tile 0 (the
    transform is periodic); otherwise the end stays on the last coarse tile.
    """
    fine, coarse = (b1, b2) if b1["w"] >= b2["w"] else (b2, b1)
    wf, wc = fine["w"], coarse["w"]
    j = np.arange(wf, dtype=np.int64)
    # Smallest k with (2k+1)·wf > (2j-1)·wc; largest k with (2k-1)·wf < (2j+1)·wc.
    k_lo = ((2 * j - 1) * wc - wf) // (2 * wf) + 1
    k_hi = -(-((2 * j + 1) * wc + wf) // (2 * wf)) - 1
    two = k_hi > k_lo
    jj = np.concatenate([j, j[two]])
    kk = np.concatenate([k_lo, k_hi[two]])
    kk = kk % wc if circular else np.clip(kk, 0, wc - 1)
    return coarse["gidx"][kk], fine["gidx"][jj]


def coefficient_adjacency(
    n: int,
    scheme: str = "dyadic_dual_real",
    window_type: str = "gaussian",
    top_window: str | None = None,
    bands_per_octave: int = 1,
    circular: bool = False,
    *,
    nyquist_flat_top: bool | None = None,
) -> CoeffAdjacency:
    """Neighbourhood graph over the packed coefficients (see
    :class:`CoeffAdjacency`). Three edge types:

    - ``time`` — consecutive coefficients within a band (intra-band time axis);
    - ``band`` — frequency-adjacent bands within a tiling, time-aligned
      (the cross-octave / cross-band "scale" relation);
    - ``dual`` — tiling-A ↔ tiling-B coefficients with overlapping support
      (the redundancy of the ``dyadic_dual_real`` scheme; empty otherwise).

    Cross-band (``band`` / ``dual``) edges join coefficients whose time tiles
    overlap: in a 2:1 octave step every interior coarse coefficient links to
    three fine ones, and a fine coefficient on a coarse tile boundary links to
    both coarse neighbours.

    The transform is circular, but by default the graph is not: no edge joins
    the end of the record to its start, so clusters (e.g. permutation cluster
    statistics) cannot wrap from ``t ≈ n`` to ``t = 0``. Pass
    ``circular=True`` for the periodic graph, which adds the wrap-around edge
    within every band of width >= 3 and wraps cross-band edges too.
    """
    top = _resolve_top_window(top_window, nyquist_flat_top)
    lay = scheme_bands(n, scheme, window_type, top, bands_per_octave)
    b_freq, b_tiling = _band_freq_and_tiling(lay, n, scheme)
    bands = []
    for bi, (lo, w, off) in enumerate(zip(lay.src_lo, lay.width, lay.out_off)):
        lo, w, off = int(lo), int(w), int(off)
        bands.append(
            dict(bi=bi, lo=lo, hi=lo + w, w=w, fc=b_freq[bi], off=off,
                 gidx=off + np.arange(w),
                 tiling=b_tiling[bi])
        )

    src, dst, typ = [], [], []

    def add(s, d, t):
        s, d = np.atleast_1d(s), np.atleast_1d(d)
        src.append(s)
        dst.append(d)
        typ.append(np.full(len(d), t, dtype=np.int64))

    # time: consecutive coefficients inside each band
    for b in bands:
        if b["w"] >= 2:
            add(b["gidx"][:-1], b["gidx"][1:], 0)
        if circular and b["w"] >= 3:
            add(b["gidx"][-1], b["gidx"][0], 0)
    # band: frequency-adjacent bands within each tiling, time-aligned
    for tl in (0, 1):
        tb = sorted((b for b in bands if b["tiling"] == tl), key=lambda b: b["fc"])
        for a, c in zip(tb[:-1], tb[1:]):
            add(*_connect_by_time(a, c, circular), 1)
    # dual: tiling A <-> tiling B with overlapping frequency support
    a_bands = [b for b in bands if b["tiling"] == 0]
    b_bands = [b for b in bands if b["tiling"] == 1]
    for a in a_bands:
        for bb in b_bands:
            if a["lo"] < bb["hi"] and bb["lo"] < a["hi"]:
                add(*_connect_by_time(a, bb, circular), 2)

    if not src:
        return CoeffAdjacency(np.zeros((2, 0), np.int64), np.zeros(0, np.int64), np.zeros(0))

    s = np.concatenate(src)
    d = np.concatenate(dst)
    t = np.concatenate(typ)
    lo_i = np.minimum(s, d)
    hi_i = np.maximum(s, d)
    keep = lo_i != hi_i  # drop any self-loops
    lo_i, hi_i, t = lo_i[keep], hi_i[keep], t[keep]
    # de-duplicate identical (i, j, type) triples via a packed 1-D key
    total = int(hi_i.max()) + 1 if len(hi_i) else 1
    key = (lo_i * total + hi_i) * 3 + t
    _, uniq = np.unique(key, return_index=True)
    uniq = np.sort(uniq)
    edge_index = np.stack([lo_i[uniq], hi_i[uniq]]).astype(np.int64)
    return CoeffAdjacency(edge_index, t[uniq].astype(np.int64), np.ones(len(uniq)))


def to_sparse_adjacency(
    adj: "CoeffAdjacency",
    n_nodes: int,
    types=("time", "band"),
    symmetric: bool = True,
    weighted: bool = False,
    fmt: str = "csr",
):
    """Build a sparse ``(n_nodes, n_nodes)`` adjacency matrix from a
    :class:`CoeffAdjacency`, for cluster-based statistics (MNE / FieldTrip
    permutation clustering and TFCE, or ``scipy.sparse.csgraph`` connected
    components).

    Parameters
    ----------
    n_nodes:
        Number of coefficients (= ``output_len``). Pass it explicitly; the edge
        list alone can under-count nodes if the last coefficients are isolated.
    types:
        Which edge relations to include, by name or id. Default ``("time",
        "band")`` — the genuine time-frequency neighbourhood. ``"dual"`` (the
        tiling-A↔B redundancy edges) is excluded by default: those link
        *redundant copies* of the same location, which would merge them in a
        cluster. Pass ``types=("time", "band", "dual")`` to include them.
    symmetric:
        If True (default), add both directions so the matrix is symmetric.
    weighted:
        If True, use ``adj.edge_weight``; otherwise binary (1.0) connectivity.

    Notes
    -----
    Caveat for multiresolution clustering: a low-frequency coefficient spans the
    whole time axis (``dt = n/width``), so ``band`` edges can bridge temporally
    distant activity through coarse tiles and inflate clusters. To avoid that,
    restrict ``types=("time",)``, or weight/prune edges using
    :func:`coefficient_geometry` before building the matrix.
    """
    try:
        import scipy.sparse as sp
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_sparse_adjacency requires scipy (pip install scipy)") from e

    name_to_id = {name: i for i, name in enumerate(adj.type_names)}
    sel = {name_to_id[t] if isinstance(t, str) else int(t) for t in types}
    mask = np.isin(adj.edge_type, list(sel))
    ei = adj.edge_index[:, mask]
    w = adj.edge_weight[mask].astype(np.float64) if weighted else np.ones(int(mask.sum()))
    rows, cols = ei[0], ei[1]
    if symmetric:
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
        w = np.concatenate([w, w])
    mat = sp.coo_matrix((w, (rows, cols)), shape=(n_nodes, n_nodes))
    return mat.asformat(fmt)
