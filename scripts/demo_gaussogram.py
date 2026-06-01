# %% [markdown]
# # gaussogram demo
#
# Jupyter "cell mode" script (run with the VS Code / Jupyter `# %%` cell
# delimiters, or `jupytext`). Plots the `dyadic_dual_real` GFT of sine waves at
# different frequencies, chirps, and multi-component signals.
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


def tone(freq_bin: float) -> np.ndarray:
    return np.cos(2 * np.pi * freq_bin * t / N)


def chirp(f0: float, f1: float) -> np.ndarray:
    # linear instantaneous-frequency sweep from f0 to f1 (in bins)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N)) / N
    return np.cos(phase)


def plot_gaussogram(signal, title, scheme="dyadic_dual_real"):
    coeffs = g.gft1d_real(signal, scheme=scheme, nyquist_flat_top=True)
    grid = g.to_grid(coeffs, len(signal), scheme=scheme, nyquist_flat_top=True)

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
# ## Single tones at different frequencies
#
# Each tone should localise as a horizontal band at its frequency. Higher
# frequencies sit in wider bands (better time resolution); low frequencies in
# narrow bands (better frequency resolution) — the constant-Q tradeoff.

# %%
for f in (16, 64, 200):
    plot_gaussogram(tone(f), f"Sine wave, f = {f} bins")
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
# ## Linear chirp
#
# A frequency sweep should trace a diagonal ridge in the time-frequency plane.

# %%
plot_gaussogram(chirp(10, 230), "Linear chirp, 10 → 230 bins")
plt.show()

# %% [markdown]
# ## Multi-component signal
#
# Two steady tones plus a chirp, demonstrating simultaneous resolution of
# stationary and non-stationary components.

# %%
multi = tone(24) + 0.8 * tone(150) + 0.9 * chirp(40, 110)
plot_gaussogram(multi, "Two tones (24, 150 bins) + chirp (40→110)")
plt.show()

# %% [markdown]
# ## Transient burst
#
# A short Gaussian-windowed high-frequency burst — well localised in time by the
# wide high-frequency bands.

# %%
burst = np.exp(-((t - N * 0.6) ** 2) / (2 * (N * 0.03) ** 2)) * tone(180)
plot_gaussogram(burst, "Gaussian burst at f=180 bins, centred at t=0.6·N")
plt.show()

# %% [markdown]
# ## Output size and band layout

# %%
print("dyadic_dual_real output length for N=512:", g.output_len(N))
print("  (= N - 1 =", N - 1, ")")
layout = g.scheme_bands(N)
print("band widths:", layout.width.tolist())
print("band centres:", layout.fcentre.tolist())
