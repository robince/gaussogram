# %% [markdown]
# # gaussogram — display interpolation comparison
#
# The transform stores only ~N coefficients: each band carries `width` complex
# samples spanning the whole record. To draw an `(N/2+1, N)` image, `to_grid`
# resamples each band's magnitude along **time** (`interp`) and then places it on
# the **frequency** axis (`interp_freq`). This script compares the four useful
# combinations:
#
# 1. `interp="nearest", interp_freq="block"` — rawest: step cells in time,
#    flat blocks in frequency (you can see the individual coefficient footprints)
# 2. `interp="linear",  interp_freq="block"` — piecewise-linear in time, still
#    block-filled in frequency (the previous default)
# 3. `interp="linear",  interp_freq="linear"` — also interpolate across
#    frequency: each band is anchored at its centre and linearly blended with
#    its neighbours, removing the horizontal block seams
# 4. `interp="linear",  interp_freq="gauss"` — weight each band by its **actual
#    Gaussian window** (σ_f = fcentre/2π) and normalise overlaps. `"block"`
#    smears a band over its full ±3σ storage support (≈1.58 octaves for the dual
#    scheme's B bands); `"gauss"` instead confines it to its true effective
#    bandwidth (FWHM ≈ 1/3 octave), so a tone renders as a compact blob at its
#    frequency rather than a tall block — the honest frequency resolution.
#
# Build the extension first: `maturin develop --release`

# %%
import numpy as np
import matplotlib.pyplot as plt

import gaussogram as g

N = 512
t = np.arange(N)


def tone(freq_bin, extent=0.5):
    sig = np.cos(2 * np.pi * freq_bin * t / N)
    if extent < 1.0:
        lo = int(round(N * (1.0 - extent) / 2.0))
        hi = int(round(N * (1.0 + extent) / 2.0))
        mask = np.zeros(N)
        mask[lo:hi] = 1.0
        sig = sig * mask
    return sig


def chirp(f0, f1):
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N)) / N
    return np.cos(phase)


# A mix of stationary (gated tones) and sweeping (chirp) content shows the
# interpolation differences on both axes.
signal = tone(36) + 0.8 * tone(140) + 0.9 * chirp(40, 110)
coeffs = g.gft1d_real(signal, nyquist_flat_top=True)

MODES = [
    ("nearest time / block freq", dict(interp="nearest", interp_freq="block")),
    ("linear time / block freq", dict(interp="linear", interp_freq="block")),
    ("linear time + linear freq", dict(interp="linear", interp_freq="linear")),
    ("linear time + gauss freq", dict(interp="linear", interp_freq="gauss")),
]

# %% [markdown]
# ## Side-by-side time-frequency images
#
# Same signal, same (shared) colour scale — only the display resampling differs.

# %%
grids = [g.to_grid(coeffs, N, nyquist_flat_top=True, normalize="width", **kw)
         for _, kw in MODES]
vmax = max(gd.max() for gd in grids)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True, sharey=True)
im = None
for ax, (name, _), grid in zip(axes, MODES, grids):
    im = ax.imshow(
        grid, origin="lower", aspect="auto",
        extent=[0, N, 0, grid.shape[0]], cmap="magma", vmin=0, vmax=vmax,
    )
    ax.set_title(name)
    ax.set_xlabel("time (samples)")
axes[0].set_ylabel("frequency (bin)")
fig.colorbar(im, ax=axes, label="|coeff|", pad=0.01)
fig.suptitle("Display interpolation: tones (36, 140) + chirp (40->110)")
plt.show()

# %% [markdown]
# ## 1-D time cross-section through a single frequency row
#
# Slicing one frequency row makes the resampling explicit. The chosen row sits
# inside a band with only a handful of time samples, so `nearest` shows discrete
# steps and `linear` connects them with straight segments. The `linear freq`
# mode can shift the level too: instead of the single covering band, the row's
# value is interpolated between the band centres above and below it.

# %%
row = 20  # inside the fc~24 band (width 16): ~16 samples across N
fig, ax = plt.subplots(figsize=(9, 4))
for name, kw in MODES:
    grid = g.to_grid(coeffs, N, nyquist_flat_top=True, normalize="width", **kw)
    ax.plot(np.arange(N), grid[row], lw=1.3, label=name)
ax.set_title(f"Magnitude along time at frequency bin {row}")
ax.set_xlabel("time (samples)")
ax.set_ylabel("|coeff|")
ax.margins(x=0)
ax.legend()
fig.tight_layout()
plt.show()
