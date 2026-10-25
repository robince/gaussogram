# %% [markdown]
# # Dead-zone sweep: tone response vs frequency
#
# Cell-mode script (`# %%`). The headline figure for the dual-tiling scheme:
# sweep a unit-amplitude tone across every integer frequency bin and record the
# **best single-band response** — the peak coefficient magnitude of the band
# that captures the tone best (scaled by band width so a captured tone scores
# its window gain, independent of the band it lands in).
#
# For a tone at integer bin `f`, the spectrum is a single spike `|X[f]| = N/2`,
# so within a band the coefficients are uniform with magnitude
# `|X[f]| * W(f) / w`; the reported response `w * max|c| / (N/2)` is exactly the
# window gain `W(f)` of the best band at that frequency:
#
# - a **single dyadic cover** (`dyadic_real`) collapses at every octave join
#   `f = 2, 4, 8, ...` where both neighbouring Gaussians are in their tails
#   (the "dead zones" — response ~0.6% of the band-centre response);
# - the **dual tiling** (`dyadic_dual_real`) places an offset band's *peak* on
#   every join, so the response is bounded away from zero at every frequency,
#   at every `bands_per_octave`.
#
# Near Nyquist, B has no band (it would need bins above N/2), so the upper half
# of A's top band is covered by that band alone. The dual scheme's default
# `top_window="wide"` widens that band's Gaussian so the response there stays
# at about 0.63 or higher instead of rolling off to ~0.11 as the plain Gaussian
# (`top_window="gaussian"`, drawn dashed at bands_per_octave = 1) does. The
# lowest bins are exact (width-1/width-2 bands barely taper).
#
# Writes `reports/figures/dead_zone_sweep.png` and prints a join/centre table
# and the response at the last bin before Nyquist.
# Build the extension first: `uv sync --all-extras --reinstall-package gaussogram`.

# %%
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gaussogram as g

N = 512
t = np.arange(N)
OUTDIR = "reports/figures"
os.makedirs(OUTDIR, exist_ok=True)


# %% response sweep
def best_band_response(
    scheme: str, bands_per_octave: int, top_window: str | None = None
) -> np.ndarray:
    """Peak width-scaled coefficient magnitude per integer tone frequency.

    Returns ``resp[f]`` for ``f = 1 .. N/2 - 1``: the max over bands of
    ``w * max_j |c_band[j]|``, normalised by the spike magnitude ``N/2`` so a
    tone exactly on a band peak scores ~1 (its window gain).
    """
    kw = dict(scheme=scheme, bands_per_octave=bands_per_octave, top_window=top_window)
    lay = g.scheme_bands(N, **kw)
    h = g.Gaussogram1d(N, **kw)
    freqs = np.arange(1, N // 2)
    resp = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        x = np.cos(2 * np.pi * f * t / N)
        c = h.forward(np.ascontiguousarray(x))
        best = 0.0
        for w, off in zip(lay.width, lay.out_off):
            w, off = int(w), int(off)
            best = max(best, w * np.abs(c[off : off + w]).max())
        resp[i] = best / (N / 2)
    return resp


freqs = np.arange(1, N // 2)
sweeps = {
    ("dyadic_real", 1): best_band_response("dyadic_real", 1),
    ("dyadic_dual_real", 1): best_band_response("dyadic_dual_real", 1),
    ("dyadic_real", 2): best_band_response("dyadic_real", 2),
    ("dyadic_dual_real", 2): best_band_response("dyadic_dual_real", 2),
}
plain_top = best_band_response("dyadic_dual_real", 1, top_window="gaussian")

# %% figure
joins = [2 ** k for k in range(1, int(np.log2(N)) - 1)]  # 2 .. N/4
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, bpo in zip(axes, (1, 2)):
    ax.semilogx(
        freqs, sweeps[("dyadic_real", bpo)],
        color="tab:red", lw=1.2, label="dyadic_real (single cover)",
    )
    ax.semilogx(
        freqs, sweeps[("dyadic_dual_real", bpo)],
        color="tab:blue", lw=1.4, label="dyadic_dual_real (dual tiling)",
    )
    if bpo == 1:
        ax.semilogx(
            freqs, plain_top, color="tab:blue", lw=1.0, ls="--", alpha=0.7,
            label='dyadic_dual_real, top_window="gaussian"',
        )
    for j in joins:
        ax.axvline(j, color="0.8", lw=0.6, zorder=0)
    ax.set_yscale("log")
    ax.set_ylabel("best-band window gain")
    ax.set_title(f"bands_per_octave = {bpo} (joins marked)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
axes[-1].set_xlabel("tone frequency (bin)")
fig.suptitle("Dead-zone sweep: unit tone response vs frequency (N = 512)")
fig.tight_layout()
path = os.path.join(OUTDIR, "dead_zone_sweep.png")
fig.savefig(path, dpi=150)
print(f"wrote {path}")

# %% join/centre table
print("\nResponse at octave joins vs band centres (bpo = 1):")
print(f"{'bin':>6} {'kind':>8} {'single':>10} {'dual':>10} {'ratio':>8}")
single, dual = sweeps[("dyadic_real", 1)], sweeps[("dyadic_dual_real", 1)]
worst_single, worst_dual = 1.0, 1.0
for j in joins:
    if j < 8:  # narrow low bands barely taper; joins only interesting from 8 up
        continue
    s, d = single[j - 1], dual[j - 1]
    centre = j + j // 2  # centre of the octave [j, 2j)
    sc, dc = single[centre - 1], dual[centre - 1]
    print(f"{j:>6} {'join':>8} {s:>10.4f} {d:>10.4f} {d / s:>8.1f}x")
    print(f"{centre:>6} {'centre':>8} {sc:>10.4f} {dc:>10.4f} {'':>8}")
    worst_single = min(worst_single, s / sc)
    worst_dual = min(worst_dual, d / dc)
print(
    f"\nworst join/centre ratio: single cover {worst_single:.4f}, "
    f"dual tiling {worst_dual:.4f}"
)
print(
    f"last bin before Nyquist (f = {N // 2 - 1}, bpo = 1): dual {dual[-1]:.4f}, "
    f'dual with top_window="gaussian" {plain_top[-1]:.4f}'
)
