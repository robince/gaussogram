# %% [markdown]
# # gaussogram — tone at band centre vs band edge
#
# The `dyadic_dual_real` scheme tiles the positive half-spectrum with two
# interleaved octave covers:
#
# - **Tiling A** (the "primary" bands): `[1,2) [2,4) [4,8) ... [N/4, N/2]`,
#   centred at 1, 3, 6, 12, 24, **48**, 96, 192 for N=512.
# - **Tiling B** (half-octave offset): bands centred on A's octave *joins*
#   (2, 4, 8, 16, 32, **64**, 128) to rescue tones that land in A's Gaussian
#   "dead zones" at the joins.
#
# This script compares a tone sitting at a **band centre** (f=48, the centre of
# A's `[32,64)` band) against one at the **band edge / octave join** (f=64, the
# A-band boundary). At the edge, the single tiling (`dyadic_real`) loses energy
# into the dead zone; the dual scheme's B band centred at 64 recovers it.
#
# Build the extension first: `maturin develop --release`

# %%
import numpy as np
import matplotlib.pyplot as plt

import gaussogram as g

N = 512
t = np.arange(N)

CENTRE_F = 48   # centre of tiling-A band [32, 64)
EDGE_F = 64     # octave join between A bands [32,64) and [64,128)


def tone(freq_bin):
    """Full-length (stationary) cosine — isolates the frequency-placement effect."""
    return np.ascontiguousarray(np.cos(2 * np.pi * freq_bin * t / N))


def band_energy(coeffs, layout):
    """Per-band (centre_freq, energy, is_tiling_B) for stem plots."""
    a_len = N // 2 + 1  # tiling A occupies the first N/2+1 coefficients
    fc, energy, is_b = [], [], []
    for lo, w, off, c in zip(layout.src_lo, layout.width, layout.out_off, layout.fcentre):
        off, w, c = int(off), int(w), int(c)
        seg = coeffs[off : off + w]
        fc.append(c)
        energy.append(float(np.sum(np.abs(seg) ** 2)))
        is_b.append(off >= a_len)
    return np.array(fc), np.array(energy), np.array(is_b)


def plot_tf(ax, coeffs, scheme, title):
    grid = g.to_grid(coeffs, N, scheme=scheme, nyquist_flat_top=True, interp="linear")
    im = ax.imshow(
        grid, origin="lower", aspect="auto",
        extent=[0, N, 0, grid.shape[0]], cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("time (samples)")
    ax.set_ylabel("freq bin")
    return im


# %% [markdown]
# ## Time-frequency images (2x2): {centre, edge} x {single tiling, dual scheme}
#
# At the centre frequency both schemes localise cleanly. At the edge, the single
# tiling smears the tone across the two neighbouring A bands (a vertical spread),
# while the dual scheme keeps it tight in the B band centred on the join.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for i, (label, f) in enumerate([("centre f=48", CENTRE_F), ("edge/join f=64", EDGE_F)]):
    c = g.gft1d_real(tone(f), scheme="dyadic_dual_real", nyquist_flat_top=True)
    c_real = g.gft1d_real(tone(f), scheme="dyadic_real", nyquist_flat_top=True)
    plot_tf(axes[i, 0], c_real, "dyadic_real", f"{label} — dyadic_real (tiling A only)")
    im = plot_tf(axes[i, 1], c, "dyadic_dual_real", f"{label} — dyadic_dual_real")
fig.colorbar(im, ax=axes, label="|coeff|", pad=0.01)
fig.suptitle("Tone at band centre vs band edge")
plt.show()

# %% [markdown]
# ## Per-band energy (where does the tone's energy land?)
#
# Stems at each band's centre frequency, log-scaled. A/B bands marked
# separately. At f=48 a single A band dominates. At f=64 the A bands on either
# side (centres 48 and 96) catch the spillover with a dip in between — the dead
# zone — while tiling B places a band exactly at 64 that captures the tone.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (label, f) in zip(axes, [("centre f=48", CENTRE_F), ("edge/join f=64", EDGE_F)]):
    layout = g.scheme_bands(N, scheme="dyadic_dual_real", nyquist_flat_top=True)
    coeffs = g.gft1d_real(tone(f), scheme="dyadic_dual_real", nyquist_flat_top=True)
    fc, energy, is_b = band_energy(coeffs, layout)
    floor = 1e-6 * energy.max()
    e = np.maximum(energy, floor)

    a = ~is_b
    ax.vlines(fc[a], floor, e[a], color="tab:blue", lw=1)
    ax.scatter(fc[a], e[a], color="tab:blue", label="tiling A (primary)", zorder=3)
    ax.vlines(fc[is_b], floor, e[is_b], color="tab:orange", lw=1)
    ax.scatter(fc[is_b], e[is_b], color="tab:orange", marker="s",
               label="tiling B (offset)", zorder=3)
    ax.axvline(f, color="k", ls="--", lw=0.8, alpha=0.6, label=f"tone f={f}")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("band centre frequency (bin)")
    ax.set_title(label)
    ax.legend(fontsize=8)
axes[0].set_ylabel("band energy")
fig.suptitle("Per-band energy distribution: centre vs edge tone (dyadic_dual_real)")
fig.tight_layout()
plt.show()
