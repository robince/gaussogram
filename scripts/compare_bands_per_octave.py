# %% [markdown]
# # bands_per_octave comparison (1 vs 2 vs 4)
#
# Cell-mode script (`# %%`). Renders the gaussogram of three canonical signals —
# an impulse, three staggered tones, and a chirp — at `bands_per_octave = 1, 2, 4`
# for both real schemes, to show the time/frequency trade-off the option rotates
# through at a *constant* coefficient count.
#
# Subdividing octaves never adds coefficients: a band's width is both its
# frequency extent and its number of time samples, so splitting an octave
# conserves the total. `dyadic_real` stays `N/2+1` and `dyadic_dual_real` stays
# `N-1` for every setting — only the tiling shape changes.
#
# Writes `reports/figures/bpo_<scheme>.png`. Build the extension first:
# `maturin develop --release`.

# %%
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gaussogram as g

N = 512
t = np.arange(N)
BPOS = (1, 2, 4)
YLIM = 200                       # focus the frequency axis
OUTDIR = "reports/figures"
os.makedirs(OUTDIR, exist_ok=True)


# --- signals (shared with the other demo / comparison scripts) -------------
def impulse(center: float) -> np.ndarray:
    s = np.zeros(N)
    s[int(round(N * center))] = 1.0
    return s


def gated_tone(f, t0, t1, taper=0.2) -> np.ndarray:
    lo, hi = int(t0 * N), int(t1 * N)
    env = np.zeros(N)
    seg = hi - lo
    w = np.ones(seg)
    r = max(int(taper * seg), 1)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
    w[:r] *= ramp
    w[-r:] *= ramp[::-1]
    env[lo:hi] = w
    return env * np.cos(2 * np.pi * f * t / N)


def chirp(f0, f1) -> np.ndarray:
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N)) / N
    return np.cos(phase)


three = (
    gated_tone(30, 0.10, 0.50)
    + gated_tone(70, 0.35, 0.75)
    + gated_tone(140, 0.60, 0.95)
)
chirp_f0, chirp_f1 = 15, 170
inst = chirp_f0 + (chirp_f1 - chirp_f0) * t / N

# Impulse placed slightly off-centre (0.45 N) so it does not sit exactly on a
# coarse time-cell boundary (which would split its energy between two cells).
imp_pos = 0.45
SIGNALS = {
    f"impulse @ {imp_pos:g}N": (impulse(imp_pos), dict(vlines=[int(imp_pos * N)])),
    "tones 30/70/140 (staggered)": (three, dict(hlines=[30, 70, 140])),
    f"chirp {chirp_f0}->{chirp_f1}": (chirp(chirp_f0, chirp_f1), dict(inst=inst)),
}


def grid(sig, scheme, nyq, bpo):
    c = g.gft1d_real(sig, scheme=scheme, nyquist_flat_top=nyq, bands_per_octave=bpo)
    return g.to_grid(
        c, N, scheme=scheme, nyquist_flat_top=nyq, bands_per_octave=bpo,
        interp="linear", interp_freq="gauss", normalize="width",
    )


def make_figure(scheme: str, nyq: bool) -> str:
    nrow, ncol = len(SIGNALS), len(BPOS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow),
                             sharex=True, sharey=True)
    for r, (title, (sig, marks)) in enumerate(SIGNALS.items()):
        for c, bpo in enumerate(BPOS):
            ax = axes[r, c]
            ax.imshow(grid(sig, scheme, nyq, bpo), origin="lower", aspect="auto",
                      cmap="magma", extent=[0, N, 0, N // 2 + 1])
            for hl in marks.get("hlines", ()):
                ax.axhline(hl, color="lime", ls=":", lw=0.6)
            for vl in marks.get("vlines", ()):
                ax.axvline(vl, color="cyan", ls=":", lw=0.6)
            if "inst" in marks:
                ax.plot(t, marks["inst"], "c--", lw=0.7)
            ax.set_ylim(0, YLIM)
            if r == 0:
                ax.set_title(f"bands_per_octave = {bpo}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{title}\nfreq (bin)", fontsize=8)
            if r == nrow - 1:
                ax.set_xlabel("time (samples)")
    coeffs = g.output_len(N, scheme)
    fig.suptitle(f"{scheme} — bands_per_octave 1/2/4 (N={N}, {coeffs} coeffs each)")
    fig.tight_layout()
    out = f"{OUTDIR}/bpo_{scheme}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# %% [markdown]
# ## Coefficient budget and the time/frequency trade
# Subdivision repartitions the axis, it never adds coefficients. Each band carries
# `width` samples => frequency extent `df = width` bins and time extent
# `dt = N/width` samples; higher `bpo` shrinks `df` and grows `dt` in lockstep.

# %%
for scheme in ("dyadic_real", "dyadic_dual_real"):
    print(f"\n=== {scheme} (output_len = {g.output_len(N, scheme)}) ===")
    print(f"{'bpo':>4s} {'n_bands':>8s}   band @bin96: df(bins)  dt(samples)")
    for bpo in BPOS:
        lay = g.scheme_bands(N, scheme, bands_per_octave=bpo)
        bi = int(np.argmax((lay.src_lo <= 96) & (96 < lay.src_lo + lay.width)))
        w = int(lay.width[bi])
        print(f"{bpo:4d} {len(lay.width):8d}              {w:8d}     {N / w:8.1f}")

# %% [markdown]
# ## The pictures (both schemes)

# %%
for scheme, nyq in (("dyadic_real", False), ("dyadic_dual_real", True)):
    print("saved", make_figure(scheme, nyq))
