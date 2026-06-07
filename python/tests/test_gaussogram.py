import numpy as np
import pytest

import gaussogram as g


# ---- partition parity (legacy vectors still reproduce) --------------------


def test_real_partitions_match_matlab_core_examples():
    np.testing.assert_array_equal(g.real_partitions(8), np.array([1, 2, 4, 6, 8], dtype=np.int32))
    np.testing.assert_array_equal(
        g.real_partitions(16),
        np.array([1, 2, 4, 8, 11, 14, 16], dtype=np.int32),
    )


def test_partitions_return_int32_positive_boundaries():
    pars = g.partitions(16)
    assert pars.dtype == np.int32
    assert pars.ndim == 1
    assert np.all(pars > 0)


# ---- new default contract: gft1d_real returns N-1 -------------------------


def test_gft1d_real_default_returns_n_minus_one():
    x = np.linspace(0.0, 1.0, 64)
    out = g.gft1d_real(x)
    assert out.shape == (63,)
    assert out.dtype == np.complex128
    assert np.all(np.isfinite(out))


def test_gft1d_real_dyadic_real_returns_half_plus_one():
    x = np.linspace(0.0, 1.0, 64)
    out = g.gft1d_real(x, scheme="dyadic_real")
    assert out.shape == (33,)
    assert np.all(np.isfinite(out))


def test_gft1d_complex_returns_length_n():
    x = np.linspace(0.0, 1.0, 16).astype(np.complex128)
    out = g.gft1d(x)
    assert out.shape == (16,)
    assert out.dtype == np.complex128
    assert np.all(np.isfinite(out))


def test_real_transform_accepts_box_window():
    out = g.gft1d_real(np.linspace(0.0, 1.0, 16), window_type="box")
    assert out.shape == (15,)
    assert np.all(np.isfinite(out))


# ---- round-trip (invertible dyadic_real) ----------------------------------


@pytest.mark.parametrize("n", [8, 16, 64, 256])
def test_dyadic_real_round_trip(n):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    coeffs = g.gft1d_real(x, scheme="dyadic_real")
    recon = g.inverse_real(coeffs)
    np.testing.assert_allclose(recon, x, atol=1e-9)


# ---- functional win: band-edge tone rescued by tiling B -------------------


def test_band_edge_tone_rescued_by_dual_scheme():
    n = 256
    join = 32  # an A octave boundary (dead zone)
    x = np.cos(2 * np.pi * join * np.arange(n) / n)
    out = g.gft1d_real(x, scheme="dyadic_dual_real")
    layout = g.scheme_bands(n, scheme="dyadic_dual_real")

    # Identify A vs B bands: A occupies the first N/2+1 coefficients.
    a_len = n // 2 + 1
    a_energy = 0.0
    b_energy = 0.0
    for lo, w, off in zip(layout.src_lo, layout.width, layout.out_off):
        lo, w, off = int(lo), int(w), int(off)
        if lo <= join < lo + w:
            e = float(np.sum(np.abs(out[off : off + w]) ** 2))
            if off < a_len:
                a_energy += e
            else:
                b_energy += e
    assert b_energy > a_energy


# ---- display grid ---------------------------------------------------------


def test_to_grid_shape_and_finite():
    n = 128
    x = np.cos(2 * np.pi * 20 * np.arange(n) / n)
    grid = g.to_grid(g.gft1d_real(x), n)
    assert grid.shape == (n // 2 + 1, n)
    assert np.all(np.isfinite(grid))


def test_to_grid_interp_and_smooth_modes():
    n = 128
    x = np.cos(2 * np.pi * 20 * np.arange(n) / n)
    coeffs = g.gft1d_real(x)
    near = g.to_grid(coeffs, n, interp="nearest")
    lin = g.to_grid(coeffs, n, interp="linear")
    sm = g.to_grid(coeffs, n, interp="linear", smooth=(2.0, 4.0))
    sm_scalar = g.to_grid(coeffs, n, smooth=1.5)
    for grid in (near, lin, sm, sm_scalar):
        assert grid.shape == (n // 2 + 1, n)
        assert np.all(np.isfinite(grid))
    # The modes must actually differ.
    assert not np.array_equal(near, lin)
    assert not np.array_equal(lin, sm)


def test_to_grid_rejects_unknown_interp():
    n = 64
    with pytest.raises(ValueError, match="interp"):
        g.to_grid(g.gft1d_real(np.ones(n)), n, interp="cubic")


def test_to_grid_interp_freq_modes():
    n = 128
    x = np.cos(2 * np.pi * 20 * np.arange(n) / n)
    coeffs = g.gft1d_real(x)
    block = g.to_grid(coeffs, n, interp="linear", interp_freq="block")
    linf = g.to_grid(coeffs, n, interp="linear", interp_freq="linear")
    linf_sm = g.to_grid(coeffs, n, interp="linear", interp_freq="linear", smooth=(2.0, 4.0))
    gauss = g.to_grid(coeffs, n, interp="linear", interp_freq="gauss")
    for grid in (block, linf, linf_sm, gauss):
        assert grid.shape == (n // 2 + 1, n)
        assert np.all(np.isfinite(grid))
    # Frequency interpolation must change the result vs block fill.
    assert not np.array_equal(block, linf)
    assert not np.array_equal(linf, linf_sm)
    assert not np.array_equal(block, gauss)
    assert not np.array_equal(linf, gauss)


def test_to_grid_rejects_unknown_normalize():
    n = 64
    with pytest.raises(ValueError, match="normalize"):
        g.to_grid(g.gft1d_real(np.ones(n)), n, normalize="zscore")


def test_to_grid_width_normalize_equalises_brightness_across_octaves():
    # |coeff| ∝ 1/width, so without normalisation an equal-amplitude tone is
    # dimmer in higher (wider) bands. normalize="width" should flatten that.
    n = 512
    t = np.arange(n)
    centres = (24, 48, 96, 192)  # successive octave-band centres
    peaks_none, peaks_width = [], []
    for f in centres:
        c = g.gft1d_real(np.ascontiguousarray(np.cos(2 * np.pi * f * t / n)),
                         nyquist_flat_top=True)
        kw = dict(nyquist_flat_top=True, interp_freq="gauss")
        peaks_none.append(g.to_grid(c, n, normalize="none", **kw).max())
        peaks_width.append(g.to_grid(c, n, normalize="width", **kw).max())
    peaks_none = np.array(peaks_none)
    peaks_width = np.array(peaks_width)
    # raw peaks fall steeply across octaves; width-normalised peaks are ~flat.
    assert peaks_none[0] > 3 * peaks_none[-1]
    assert peaks_width.max() / peaks_width.min() < 1.3


def test_to_grid_gauss_confines_join_tone_to_effective_bandwidth():
    # A tone exactly on an octave join is captured by a B band centred there.
    # `block` smears it across the band's full ±3σ support; `gauss` confines it
    # to the effective bandwidth and peaks at the true frequency.
    n = 512
    join = 64
    x = np.ascontiguousarray(np.cos(2 * np.pi * join * np.arange(n) / n))
    coeffs = g.gft1d_real(x, scheme="dyadic_dual_real", nyquist_flat_top=True)
    kw = dict(scheme="dyadic_dual_real", nyquist_flat_top=True)
    block = g.to_grid(coeffs, n, interp_freq="block", **kw)
    gauss = g.to_grid(coeffs, n, interp_freq="gauss", **kw)

    col_b = block[:, n // 2]
    col_g = gauss[:, n // 2]

    def half_max_span(col):
        rows = np.flatnonzero(col > 0.5 * col.max())
        return int(rows.min()), int(rows.max())

    lo_b, hi_b = half_max_span(col_b)
    lo_g, hi_g = half_max_span(col_g)
    # gauss peaks at the true frequency; block does not (it fills the support).
    assert abs(int(col_g.argmax()) - join) <= 2
    # gauss is markedly narrower in frequency than the full-support block fill.
    assert (hi_g - lo_g) < 0.5 * (hi_b - lo_b)


def test_to_grid_rejects_unknown_interp_freq():
    n = 64
    with pytest.raises(ValueError, match="interp_freq"):
        g.to_grid(g.gft1d_real(np.ones(n)), n, interp_freq="quadratic")


# ---- error cases ----------------------------------------------------------


@pytest.mark.parametrize("size", [12, 100])
def test_non_power_of_two_rejected(size):
    with pytest.raises(ValueError, match="power of two"):
        g.gft1d_real(np.ones(size))


def test_dual_scheme_rejects_n_less_than_8():
    with pytest.raises(ValueError):
        g.gft1d_real(np.ones(4), scheme="dyadic_dual_real")


def test_invalid_window_type_is_rejected():
    with pytest.raises(ValueError, match="window_type"):
        g.gft1d_real(np.ones(16), window_type="hann")


@pytest.mark.parametrize("size", [1, 2, 6, 12])
def test_real_partitions_reject_invalid_sizes(size):
    with pytest.raises(ValueError, match="power of two"):
        g.real_partitions(size)


@pytest.mark.parametrize("size", [1, 2, 4, 12])
def test_complex_partitions_reject_invalid_sizes(size):
    with pytest.raises(ValueError, match="power of two"):
        g.partitions(size)


# ---- strict input validation (no silent copy/cast in the wrapper) ---------


def test_wrong_dtype_rejected_not_silently_cast():
    x = np.cos(np.arange(64, dtype=np.float32))  # float32, not float64
    with pytest.raises(TypeError, match="dtype"):
        g.gft1d_real(x)


def test_non_contiguous_input_rejected():
    x = np.cos(2 * np.pi * 10 * np.arange(128) / 128)
    with pytest.raises(ValueError, match="C-contiguous"):
        g.gft1d_real(x[::2])  # strided view, length 64


def test_non_ndarray_input_rejected():
    with pytest.raises(TypeError, match="ndarray"):
        g.gft1d_real([0.0] * 64)


def test_higher_dim_input_rejected():
    with pytest.raises(ValueError, match="1D"):
        g.gft1d_real(np.zeros((8, 8)))


def test_input_not_mutated_and_writeable_flag_restored():
    x = np.ascontiguousarray(np.cos(2 * np.pi * 12 * np.arange(64) / 64))
    before = x.copy()
    assert x.flags.writeable
    g.gft1d_real(x)
    # The transform borrows the buffer; it must neither mutate it nor leave it
    # locked read-only after returning.
    np.testing.assert_array_equal(x, before)
    assert x.flags.writeable


# ---- reusable Gaussogram1d handle -----------------------------------------


def test_handle_matches_free_function():
    n = 256
    x = np.ascontiguousarray(np.cos(2 * np.pi * 40 * np.arange(n) / n))
    h = g.Gaussogram1d(n, scheme="dyadic_dual_real", nyquist_flat_top=True)
    assert h.n == n
    assert h.output_len == n - 1
    free = g.gft1d_real(x, scheme="dyadic_dual_real", nyquist_flat_top=True)
    handle = h.forward(x)
    np.testing.assert_array_equal(handle, free)


def test_handle_reused_across_calls_is_stable():
    n = 128
    h = g.Gaussogram1d(n, scheme="dyadic_dual_real")
    first = None
    out = None
    for f in (8, 16, 32):
        x = np.ascontiguousarray(np.cos(2 * np.pi * f * np.arange(n) / n))
        out = h.forward(x)
        # Re-running the same signal must reproduce exactly (scratch reuse must
        # not leak state between calls).
        np.testing.assert_array_equal(out, h.forward(x))
        if f == 8:
            first = out
    # Sanity: a different frequency gives a different result.
    assert not np.array_equal(first, out)


def test_handle_round_trip_dyadic_real():
    n = 256
    rng = np.random.default_rng(1)
    x = np.ascontiguousarray(rng.standard_normal(n))
    h = g.Gaussogram1d(n, scheme="dyadic_real")
    assert h.invertible
    coeffs = h.forward(x)
    recon = h.inverse(coeffs)
    np.testing.assert_allclose(recon, x, atol=1e-9)


def test_handle_rejects_wrong_length():
    h = g.Gaussogram1d(64)
    with pytest.raises(ValueError):
        h.forward(np.ascontiguousarray(np.zeros(32)))


# ---- coefficient geometry & adjacency (downstream-model metadata) ---------


def test_coefficient_geometry_matches_packing():
    n = 128
    geo = g.coefficient_geometry(n)  # dyadic_dual_real
    assert len(geo.time) == g.output_len(n)
    # every tile has area n (dt = n/width, df = width)
    np.testing.assert_allclose(geo.dt * geo.df, n)
    # centre times lie inside the record; the dual scheme has both tilings
    assert geo.time.min() >= 0 and geo.time.max() < n
    assert set(geo.tiling.tolist()) == {0, 1}
    # per-coefficient freq equals the owning band's centre frequency
    lay = g.scheme_bands(n)
    for lo, w, fc, off in zip(lay.src_lo, lay.width, lay.fcentre, lay.out_off):
        off, w, fc = int(off), int(w), int(fc)
        assert np.all(geo.freq[off : off + w] == fc)


def test_coefficient_adjacency_is_valid_graph():
    n = 128
    total = g.output_len(n)
    adj = g.coefficient_adjacency(n)
    ei, et = adj.edge_index, adj.edge_type
    assert ei.shape[0] == 2 and ei.shape[1] == et.shape[0]
    # valid, undirected (i < j), no self-loops
    assert ei.min() >= 0 and ei.max() < total
    assert np.all(ei[0] < ei[1])
    # edge types are a subset of the three named relations
    assert set(et.tolist()).issubset({0, 1, 2})
    # time edges = consecutive coefficients within every band of width >= 2
    expected_time = sum(max(int(w) - 1, 0) for w in g.scheme_bands(n).width if int(w) >= 2)
    assert int((et == 0).sum()) == expected_time
    # the dual scheme has A<->B (type 2) edges
    assert int((et == 2).sum()) > 0


def test_coefficient_adjacency_single_tiling_has_no_dual_edges():
    adj = g.coefficient_adjacency(128, scheme="dyadic_real")
    assert int((adj.edge_type == 2).sum()) == 0  # no tiling B, so no dual edges


# ---- bands_per_octave (sub-octave subdivision) ----------------------------


@pytest.mark.parametrize("bpo", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [16, 64, 256, 512])
def test_bands_per_octave_preserves_output_length(n, bpo):
    # Subdivision conserves the coefficient count: real stays N/2+1, dual N-1.
    assert g.output_len(n, "dyadic_real", bands_per_octave=bpo) == n // 2 + 1
    assert g.output_len(n, "dyadic_dual_real", bands_per_octave=bpo) == n - 1


@pytest.mark.parametrize("bpo", [2, 4, 8])
@pytest.mark.parametrize("n", [16, 64, 256])
def test_bands_per_octave_round_trip(n, bpo):
    # dyadic_real stays exactly invertible for every power-of-two bands_per_octave.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    coeffs = g.gft1d_real(x, scheme="dyadic_real", bands_per_octave=bpo)
    assert len(coeffs) == n // 2 + 1
    recon = g.inverse_real(coeffs, bands_per_octave=bpo)
    np.testing.assert_allclose(recon, x, atol=1e-9)


def test_bands_per_octave_subdivides_octaves():
    # bpo=2 splits each splittable octave into two equal sub-bands.
    base = g.scheme_bands(256, "dyadic_real", bands_per_octave=1)
    sub = g.scheme_bands(256, "dyadic_real", bands_per_octave=2)
    assert len(sub.width) > len(base.width)
    # Tiling A remains a gap-free, overlap-free cover of 0..=N/2.
    assert sub.src_lo[0] == 0
    np.testing.assert_array_equal(sub.src_hi[:-1], sub.src_lo[1:])
    assert sub.src_hi[-1] == 256 // 2 + 1


@pytest.mark.parametrize("bpo", [0, 3, 6, 5])
def test_bands_per_octave_rejects_non_power_of_two(bpo):
    x = np.zeros(256)
    with pytest.raises(ValueError, match="bands_per_octave"):
        g.gft1d_real(x, scheme="dyadic_real", bands_per_octave=bpo)


def test_bands_per_octave_geometry_level_and_sub():
    n = 256
    geo = g.coefficient_geometry(n, "dyadic_real", bands_per_octave=2)
    # level is the octave index; sub ranks within the octave (0 or 1 for bpo=2).
    assert set(geo.sub.tolist()).issubset({0, 1})
    assert (geo.sub == 1).sum() > 0  # some octaves actually split
    # tile area is still conserved
    np.testing.assert_allclose(geo.dt * geo.df, n)
    # within a band, (level, sub) is constant and matches log-frequency ordering
    lay = g.scheme_bands(n, "dyadic_real", bands_per_octave=2)
    for off, w in zip(lay.out_off, lay.width):
        off, w = int(off), int(w)
        assert len(set(geo.level[off : off + w].tolist())) == 1
        assert len(set(geo.sub[off : off + w].tolist())) == 1


def test_bands_per_octave_one_matches_default_layout():
    # bpo=1 must be identical to the implicit single-band-per-octave cover.
    for scheme in ("dyadic_real", "dyadic_dual_real"):
        a = g.scheme_bands(256, scheme, bands_per_octave=1)
        b = g.scheme_bands(256, scheme)
        np.testing.assert_array_equal(a.src_lo, b.src_lo)
        np.testing.assert_array_equal(a.width, b.width)
        np.testing.assert_array_equal(a.fcentre, b.fcentre)
