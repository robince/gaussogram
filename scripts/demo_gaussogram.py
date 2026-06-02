# %% [markdown]
# # gaussogram demo
#
# Jupyter "cell mode" script (run with the VS Code / Jupyter `# %%` cell
# delimiters, or `jupytext`). Plots the `dyadic_dual_real` GFT of windowed sine
# waves, chirps, impulses, boxcars, and multi-component signals.
#
# Build the extension first:
#
# ```
# maturin develop --release
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt

import gaussogram as g

N = 512
t = np.arange(N)


def _gate(sig: np.ndarray, extent: float) -> np.ndarray:
    """Zero everything outside the central ``extent`` fraction of time."""
    if extent >= 1.0:
        return sig
    lo = int(round(N * (1.0 - extent) / 2.0))
    hi = int(round(N * (1.0 + extent) / 2.0))
    mask = np.zeros(N)
    mask[lo:hi] = 1.0
    return sig * mask


def tone(freq_bin: float, extent: float = 0.5) -> np.ndarray:
    """Cosine at ``freq_bin``, present only over the central ``extent`` of time
    (default: middle 50%). Gating localises it in time so the time-frequency
    plot shows a finite horizontal segment rather than a band spanning all t."""
    return _gate(np.cos(2 * np.pi * freq_bin * t / N), extent)


def chirp(f0: float, f1: float) -> np.ndarray:
    # linear instantaneous-frequency sweep from f0 to f1 (in bins)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N)) / N
    return np.cos(phase)


def impulse(center: float = 0.5, amp: float = 1.0) -> np.ndarray:
    """Single-sample spike at fractional position ``center``."""
    sig = np.zeros(N)
    sig[int(round(N * center))] = amp
    return sig


def boxcar(start: float = 0.4, stop: float = 0.6, amp: float = 1.0) -> np.ndarray:
    """Rectangular pulse over the fractional interval ``[start, stop)``."""
    sig = np.zeros(N)
    sig[int(round(N * start)) : int(round(N * stop))] = amp
    return sig


def plot_gaussogram(
    signal,
    title,
    scheme="dyadic_dual_real",
    interp="linear",
    smooth=None,
):
    coeffs = g.gft1d_real(signal, scheme=scheme, nyquist_flat_top=True)
    grid = g.to_grid(
        coeffs, len(signal), scheme=scheme, nyquist_flat_top=True,
        interp=interp, smooth=smooth,
    )

    fig, (ax_sig, ax_tf) = plt.subplots(
        2, 1, figsize=(9, 5), gridspec_kw={"height_ratios": [1, 3]}, sharex=True
    )
    ax_sig.plot(np.arange(len(signal)), signal, lw=0.8)
    ax_sig.set_title(title)
    ax_sig.set_ylabel("amplitude")
    ax_sig.margins(x=0)

    im = ax_tf.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[0, len(signal), 0, grid.shape[0]],
        cmap="magma",
    )
    ax_tf.set_xlabel("time (samples)")
    ax_tf.set_ylabel("frequency (bin)")
    fig.colorbar(im, ax=ax_tf, label="|coeff|", pad=0.01)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Single tones at different frequencies (middle 50% of time)
#
# Each tone is gated to the central half of the record, so it should localise as
# a horizontal segment at its frequency, switching on near t=N/4 and off near
# t=3N/4. Higher frequencies sit in wider bands (better time resolution, so the
# on/off edges look sharper); low frequencies in narrow bands (better frequency
# resolution, blurrier edges) — the constant-Q tradeoff.

# %%
for f in (16, 64, 200):
    plot_gaussogram(tone(f), f"Sine wave, f = {f} bins (central 50%)")
plt.show()

# %% [markdown]
# ## Band-edge tones — the point of the dual scheme
#
# Tones placed exactly on tiling A's octave joins (e.g. 32, 64, 128) fall in
# A's Gaussian "dead zones". The half-octave-offset tiling B is centred on those
# joins, so it rescues them. Compare the dual scheme against the single tiling.

# %%
join = 64
sig = tone(join)
for scheme in ("dyadic_real", "dyadic_dual_real"):
    plot_gaussogram(sig, f"Tone at octave join f={join}  —  scheme={scheme}", scheme=scheme)
plt.show()

# %% [markdown]
# ## Impulse
#
# A single-sample spike is maximally localised in time and flat in frequency, so
# it should trace a vertical ridge at its time index spanning all bands. The
# ridge widens at low frequency (narrow-band = poor time resolution) and narrows
# at high frequency — the time-domain mirror of the constant-Q tradeoff.

# %%
plot_gaussogram(impulse(0.5), "Unit impulse at t = N/2")
plt.show()

# %% [markdown]
# ## Boxcar
#
# A rectangular pulse: a low-frequency lobe spanning its duration, plus
# broadband energy concentrated at the two switching edges.

# %%
plot_gaussogram(boxcar(0.4, 0.6), "Boxcar over t in [0.4*N, 0.6*N)")
plt.show()

# %% [markdown]
# ## Linear chirp
#
# A frequency sweep should trace a diagonal ridge in the time-frequency plane.

# %%
plot_gaussogram(chirp(10, 230), "Linear chirp, 10 -> 230 bins")
plt.show()

# %% [markdown]
# ## Multi-component signal
#
# Two gated tones (central 50%) plus a full-length chirp, demonstrating
# simultaneous resolution of stationary and non-stationary components.

# %%
multi = tone(24) + 0.8 * tone(150) + 0.9 * chirp(40, 110)
plot_gaussogram(multi, "Two tones (24, 150 bins, central 50%) + chirp (40->110)")
plt.show()

# %% [markdown]
# ## Transient burst
#
# A short Gaussian-windowed high-frequency burst — well localised in time by the
# wide high-frequency bands. (The carrier here is ungated; the Gaussian envelope
# does the localising.)

# %%
burst = np.exp(-((t - N * 0.6) ** 2) / (2 * (N * 0.03) ** 2)) * tone(180, extent=1.0)
plot_gaussogram(burst, "Gaussian burst at f=180 bins, centred at t=0.6*N")
plt.show()

# %% [markdown]
# ## Interpolation / smoothing for display
#
# The transform stores only ~N coefficients total: each band carries `width`
# complex samples spanning the whole record (a few samples for narrow
# low-frequency bands, up to N/2 for the widest high-frequency band). To draw an
# (N/2+1, N) image we resample each band's magnitude along time and block-fill it
# across the frequency rows the band covers. So the apparent resolution is set by
# the band widths, not by an N x N/2 dense Stockwell grid — that sparsity is the
# whole point of the representation, but it shows up as blockiness here.
#
# `to_grid` exposes the resampling:
# - `interp="nearest"`: raw step/block sampling — you can see the individual
#   coefficient cells.
# - `interp="linear"` (default): piecewise-linear in time — smooths the time axis.
# - `smooth=(sf, st)`: an extra separable Gaussian blur (sigma in cells) that
#   also softens the hard block edges across frequency.

# %%
demo_sig = tone(24) + 0.8 * tone(150) + 0.9 * chirp(40, 110)
for interp, smooth, label in [
    ("nearest", None, "nearest (raw cells)"),
    ("linear", None, "linear in time"),
    ("linear", (2.0, 4.0), "linear + Gaussian smooth (sigma_f=2, sigma_t=4)"),
]:
    plot_gaussogram(demo_sig, f"Multi-component — {label}", interp=interp, smooth=smooth)
plt.show()

# %% [markdown]
# ## Output size and band layout

# %%
print("dyadic_dual_real output length for N=512:", g.output_len(N))
print("  (= N - 1 =", N - 1, ")")
layout = g.scheme_bands(N)
print("band widths:", layout.width.tolist())
print("band centres:", layout.fcentre.tolist())
