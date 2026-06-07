# %% [markdown]
# # gaussogram — the time-frequency tiling (how `to_grid` assembles the image)
#
# This script draws **one outline rectangle per stored coefficient** so you can
# see exactly where each number in the packed transform lives in the
# time-frequency plane, and how `to_grid` turns ~N coefficients into an
# `(N/2+1, N)` image.
#
# ## The model
#
# The transform stores only ~N complex numbers. They are grouped into **bands**.
# A band covering frequency rows `[lo, lo+w)` carries `w` complex coefficients,
# and those `w` samples are spread evenly across the **whole** time record. So a
# single coefficient owns a time-frequency **tile** of size:
#
# ```
#   (N / w) samples in time   ×   w rows in frequency      →  area = N
# ```
#
# Narrow low-frequency bands (small `w`) → tall, thin tiles: good frequency
# resolution, poor time resolution. Wide high-frequency bands (large `w`) →
# short, fat tiles: the constant-Q trade-off, drawn explicitly.
#
# ## How `to_grid` fills the image from these tiles
#
# 1. **Time axis — interpolated.** Each band has `w` magnitude samples placed at
#    `np.linspace(0, N-1, w)`. `to_grid` resamples them to all `N` time columns
#    (`interp="nearest"` = pick the nearest sample → visible steps at tile
#    boundaries; `interp="linear"` = straight lines between sample centres).
# 2. **Frequency axis — block fill, NOT interpolated.** The resampled row is
#    copied unchanged into *every* frequency row the band spans (`grid[lo:hi]`).
#    A band is one flat colour vertically — the hard horizontal edges you see in
#    the gaussograms are exactly these block boundaries.
# 3. **Overlap — resolved by max.** In `dyadic_dual_real` the offset B bands sit
#    on top of the A bands. Where an A tile and a B tile cover the same
#    `(freq, time)` cell, `to_grid` keeps the **larger magnitude**
#    (`np.maximum`). There is no averaging or blending across the overlap.
#
# (The optional `smooth=(sigma_f, sigma_t)` is a cosmetic Gaussian blur applied
# *after* assembly — it softens the block edges but is not part of the transform.)
#
# Build the extension first: `maturin develop --release`

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import gaussogram as g

N = 64  # small, so individual tiles are countable
NFREQ = N // 2 + 1


def band_tiles(layout):
    """Yield (lo, hi, t_start, t_width, col_index, n_cols) for every coefficient
    tile. A band of width ``w`` spanning rows [lo, hi) is split into ``w`` tiles
    along time, each (N/w) wide."""
    for lo, w in zip(layout.src_lo, layout.width):
        lo, w = int(lo), int(w)
        hi = lo + w
        tile_t = N / w  # time-width of one coefficient's tile
        for k in range(w):
            yield lo, hi, k * tile_t, tile_t, k, w


def draw_tiles(ax, layout, color, label, hatch=None, lw=1.4,
               face_alpha=0.0, inset=(0.0, 0.0)):
    """Draw one outline rectangle per coefficient tile.

    ``face_alpha`` adds a translucent fill so two overlapping tilings blend
    visibly instead of hiding each other. ``inset=(it, ifr)`` shrinks each
    rectangle by ``it`` samples in time and ``ifr`` rows in frequency, so tiles
    that share an exact boundary with another tiling no longer draw coincident
    edges (otherwise the second-drawn colour paints over the first)."""
    it, ifr = inset
    face = mcolors.to_rgba(color, face_alpha) if face_alpha > 0 else "none"
    first = True
    for lo, hi, t0, tw, *_ in band_tiles(layout):
        hi_clip = min(hi, NFREQ)  # the displayed grid only has N/2+1 rows
        if lo >= NFREQ:
            continue
        x, y = t0 + it, lo + ifr
        w_, h_ = tw - 2 * it, (hi_clip - lo) - 2 * ifr
        if w_ <= 0 or h_ <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (x, y), w_, h_,
                facecolor=face, edgecolor=color, linewidth=lw,
                hatch=hatch, label=label if first else None,
            )
        )
        first = False


# %% [markdown]
# ## 1. `dyadic_real` — a clean, non-overlapping dyadic tiling
#
# Every (freq, time) cell is covered by exactly one tile. Count them: the widths
# are `[1, 1, 2, 4, 8, 17]`, summing to the 33 = N/2+1 stored coefficients. Note
# how each octave doubles the time resolution (more, narrower columns) while
# halving the frequency resolution (taller rows).

# %%
layout_real = g.scheme_bands(N, scheme="dyadic_real", nyquist_flat_top=True)

fig, ax = plt.subplots(figsize=(9, 5))
draw_tiles(ax, layout_real, color="tab:blue", label="coefficient tile")
ax.set_xlim(0, N)
ax.set_ylim(0, NFREQ)
ax.set_xlabel("time (samples)")
ax.set_ylabel("frequency (bin)")
ax.set_title("dyadic_real tiling (N=64): each rectangle = one stored coefficient")
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. `dyadic_dual_real` — A tiles (blue) with offset B tiles (orange) on top
#
# The B bands are centred on tiling A's octave *joins* (the horizontal dotted
# lines). Each B band is built (see `scheme.rs`) as `[b/2, 3b/2)` — **linearly
# symmetric** about a join `b`, with its Gaussian peak exactly on `b`.
#
# **Why the orange bands overlap each other** (this looks wrong but is correct):
# the B joins `2,4,8,16` are spaced one octave apart, but each B band spans
# `[b/2, 3b/2)` = a ratio of 3:1 ≈ 1.58 octaves — *wider* than the spacing — so
# consecutive B bands necessarily overlap, by `b/2` bins. Unlike tiling A (a
# critically-sampled partition: each frequency covered exactly once), tiling B is
# a deliberately **redundant overlay** — that oversampling is what rescues
# dead-zone tones. The rectangles show each band's *support*; the Gaussian inside
# tapers from the centre (dotted line) into the edges, so the overlaps are window
# *tails*, not equal-weight double counting. In the rendered image `to_grid`
# keeps whichever magnitude is larger per cell (`np.maximum`).

# %%
layout_dual = g.scheme_bands(N, scheme="dyadic_dual_real", nyquist_flat_top=True)
# Split A (first N/2+1 coeffs) from B by out_off, matching to_grid / band_energy.
a_len = NFREQ
a_mask = layout_dual.out_off < a_len
layout_A = g.BandLayout(*(arr[a_mask] for arr in layout_dual))
layout_B = g.BandLayout(*(arr[~a_mask] for arr in layout_dual))

# A and B bands share many exact frequency boundaries (e.g. A [2,4) and B [2,6)
# both start at row 2), so drawing both as bare outlines makes the
# second-drawn colour paint over the first. Translucent fills let the overlap
# blend, and a small *frequency* inset on B lifts its horizontal edges off A's
# coincident edges. (No time inset: within a band the tiles abut in time, so
# insetting in time would draw misleading gaps between adjacent coefficients.)
fig, ax = plt.subplots(figsize=(9, 5))
draw_tiles(ax, layout_A, color="tab:blue", label="tiling A (primary)",
           face_alpha=0.12)
draw_tiles(ax, layout_B, color="tab:orange", label="tiling B (offset)",
           face_alpha=0.12, inset=(0.0, 0.35))
# Mark each B band's centre/peak frequency (the A-joins): the Gaussian peaks
# here and tapers into the overlapping support drawn around it.
for fc in layout_B.fcentre:
    ax.axhline(int(fc), color="tab:orange", ls=":", lw=1.0, alpha=0.8)
ax.axhline(int(layout_B.fcentre[0]), color="tab:orange", ls=":", lw=1.0,
           alpha=0.8, label="B peak (octave join)")
ax.set_xlim(0, N)
ax.set_ylim(0, NFREQ)
ax.set_xlabel("time (samples)")
ax.set_ylabel("frequency (bin)")
ax.set_title("dyadic_dual_real tiling (N=64): A (blue) + B (orange, inset) overlap")
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Tiling next to the rendered image
#
# Left: the B-band tile outlines. Right: what `to_grid` actually draws for a tone
# sitting on the f=16 octave join — the B tile centred there lights up, block-
# filled across its frequency rows and (linearly) interpolated across time. The
# outline on the right marks the dominant B tile so you can match drawing to
# render.

# %%
JOIN = 16  # an A octave boundary (dead zone) that a B band is centred on
t = np.arange(N)
sig = np.ascontiguousarray(np.cos(2 * np.pi * JOIN * t / N))
coeffs = g.gft1d_real(sig, scheme="dyadic_dual_real", nyquist_flat_top=True)
grid = g.to_grid(coeffs, N, scheme="dyadic_dual_real",
                 nyquist_flat_top=True, interp="linear", normalize="width")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

draw_tiles(axL, layout_A, color="tab:blue", label="tiling A", face_alpha=0.12)
draw_tiles(axL, layout_B, color="tab:orange", label="tiling B",
           face_alpha=0.12, inset=(0.0, 0.35))
axL.set_xlim(0, N)
axL.set_ylim(0, NFREQ)
axL.set_title("tile outlines (A + B)")
axL.set_xlabel("time (samples)")
axL.set_ylabel("frequency (bin)")
axL.legend(loc="upper right", fontsize=8)

axR.imshow(grid, origin="lower", aspect="auto",
           extent=[0, N, 0, NFREQ], cmap="magma")
# Outline the B band centred on the join (rows [8,24), clipped to NFREQ).
for lo, w in zip(layout_B.src_lo, layout_B.width):
    lo, w = int(lo), int(w)
    if lo <= JOIN < lo + w:
        axR.add_patch(Rectangle((0, lo), N, min(lo + w, NFREQ) - lo,
                                fill=False, edgecolor="cyan", lw=1.6,
                                label=f"B band covering f={JOIN}"))
axR.set_title(f"to_grid render, tone at join f={JOIN}")
axR.set_xlabel("time (samples)")
axR.legend(loc="upper right", fontsize=8)
fig.suptitle("Tiling outlines vs the assembled image (block fill in f, interp in t)")
fig.tight_layout()
plt.show()
