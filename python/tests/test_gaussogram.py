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
