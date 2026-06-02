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
