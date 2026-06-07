# %% [markdown]
# # Situating gaussogram against off-the-shelf transforms
#
# Runs the same set of transforms on several signals and renders them all on a
# common frequency-BIN axis for a fair visual comparison:
#
# - **gaussogram** `dyadic_dual_real` (the current Rust transform)
# - **two_channel_ramp**, **pou_A** (our prototypes)
# - **pywt `wavedec`** — critically-sampled dyadic wavelet (the parsimony option)
# - **dtcwt** — dual-tree complex wavelet (~2x, shift-invariant)
# - **nsgt** — nonstationary Gabor / invertible constant-Q (the principled match)
# - **ssqueezepy `cwt`/`ssq_cwt`** (GMW wavelet) and **`stft`/`ssq_stft`**
#
# Octave methods (pywt, dtcwt) are rendered by anchoring each octave band at its
# geometric-centre frequency and interpolating between centres (a standard
# log-frequency scalogram) — fairer than a hard block-fill. They are still
# octave-coarse (one band per octave); that is the transform, not the rendering.

# %%
import warnings
import numpy as np
import runpy

# --- numpy 2.x compat shims for the older nsgt / dtcwt packages ---
warnings.filterwarnings("ignore")
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
_np_clip = np.clip
def _clip_compat(a, amin, amax, out=None, **k):
    if out is not None and getattr(out, "dtype", None) is not None and out.dtype.kind in "iu":
        out[...] = np.asarray(_np_clip(a, amin, amax)).astype(out.dtype)
        return out
    return _np_clip(a, amin, amax, out=out, **k)
np.clip = _clip_compat

import gaussogram as g
import pywt
from ssqueezepy import cwt, ssq_cwt, stft, ssq_stft
from nsgt import NSGT, OctScale
import dtcwt

N = 512
t = np.arange(N)
nfreq = N // 2 + 1
proto = runpy.run_path("scripts/prototype_edge_schemes.py")


def node_interp(node_freqs, profiles):
    """Place per-band time-rows at their centre frequencies and linearly
    interpolate across the bin axis (rows outside the centre range are clamped)."""
    order = np.argsort(node_freqs)
    xf = np.asarray(node_freqs, float)[order]
    P = np.stack(profiles)[order]
    x = np.arange(nfreq)
    idx = np.clip(np.searchsorted(xf, x, side="right") - 1, 0, len(xf) - 2)
    x0, x1 = xf[idx], xf[idx + 1]
    w = np.clip(np.where(x1 > x0, (x - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0), 0, 1)
    return (1 - w)[:, None] * P[idx] + w[:, None] * P[idx + 1]


def time_to_N(mag):
    return (np.full(N, mag[0]) if mag.size == 1
            else np.interp(np.arange(N), np.linspace(0, N - 1, mag.size), mag))


# ---------------------------------------------------------------------------
# Grid builders (signal -> (rows, time) magnitude grid).  row_to_bin=None means
# the grid is already on the 0..N/2 bin axis; otherwise it maps rows->bins.
# ---------------------------------------------------------------------------
# All methods use RAW |coeff| (no width normalisation) so the comparison is
# consistent: a flat-spectrum impulse stays flat across every panel. (Tones then
# dim toward high frequency equally for all — the honest, consistent trade.)
def grid_gaussogram(s):
    c = g.gft1d_real(s, scheme="dyadic_dual_real", nyquist_flat_top=True)
    return g.to_grid(c, N, nyquist_flat_top=True, interp_freq="gauss", normalize="none")


pywt_level = int(np.log2(N)) - 1


def grid_pywt(s):
    coeffs = pywt.wavedec(s, "sym8", level=pywt_level, mode="periodization")
    nodef, profs = [], []
    for k, cD in enumerate(coeffs[1:]):            # cD_L ... cD_1
        j = pywt_level - k
        lo, hi = N / 2 ** (j + 1), N / 2 ** j
        nodef.append(np.sqrt(lo * hi))             # geometric octave centre
        profs.append(time_to_N(np.abs(cD)))
    hiA = N / 2 ** (pywt_level + 1)
    nodef.append(hiA / 2)
    profs.append(time_to_N(np.abs(coeffs[0])))
    return node_interp(nodef, profs)


_dt = dtcwt.Transform1d()


def grid_dtcwt(s):
    pyr = _dt.forward(np.ascontiguousarray(s).reshape(-1, 1), nlevels=8)
    nodef, profs = [], []
    for j, h in enumerate(pyr.highpasses, start=1):
        lo, hi = N / 2 ** (j + 1), N / 2 ** j
        nodef.append(np.sqrt(lo * hi))
        profs.append(time_to_N(np.abs(h).ravel()))
    return node_interp(nodef, profs)


_scl = OctScale(fmin=8, fmax=N / 2, bpo=4)
_nsg = NSGT(_scl, fs=N, Ls=N, real=True)
_nsg_nodes = np.concatenate([[0.0], np.asarray(_scl.F()), [N / 2]])


def grid_nsgt(s):
    c = list(_nsg.forward(np.ascontiguousarray(s)))
    nf = _nsg_nodes[: len(c)]
    return node_interp(nf, [time_to_N(np.abs(np.asarray(ci))) for ci in c])


_, _scales = cwt(x := np.cos(2 * np.pi * 50 * t / N).astype(np.float64), "gmw", fs=N)
try:
    from ssqueezepy.experimental import scale_to_freq
    _cwt_freqs = np.asarray(scale_to_freq(_scales, "gmw", N, fs=N))
except Exception:
    _cwt_freqs = None


# registry: name -> (gridfn, row_to_bin, real_dof)
def _count(s, complex_):
    return None  # filled below


METHODS = {
    "gaussogram_dual":   (grid_gaussogram, None, 2 * (N - 1)),
    "two_channel_ramp":  (lambda s: proto["grid_ramp"](s, width_norm=False), None, 2 * 514),
    "pou_A":             (lambda s: proto["grid_pou"](s, width_norm=False), None, 2 * 383),
    "pywt_wavedec":      (grid_pywt, None, sum(len(c) for c in
                          pywt.wavedec(t * 0.0, "sym8", level=pywt_level, mode="periodization"))),
    "dtcwt":             (grid_dtcwt, None, 2 * sum(h.size for h in
                          _dt.forward((t * 0.0).reshape(-1, 1), nlevels=8).highpasses)),
    "nsgt(CQ,4bpo)":     (grid_nsgt, None, 2 * sum(np.asarray(ci).size
                          for ci in _nsg.forward(t * 0.0))),
    "cwt:GMW":           (lambda s: np.abs(cwt(s, "gmw", fs=N)[0]), _cwt_freqs, 2 * len(_scales) * N),
    "ssq_cwt":           (lambda s: np.abs(ssq_cwt(s, "gmw", fs=N)[0]), _cwt_freqs, 2 * len(_scales) * N),
    "stft(absphase)":    (lambda s: np.abs(stft(s)), None, 2 * stft(t * 0.0).size),
    "ssq_stft":          (lambda s: np.abs(ssq_stft(s)[0]), None, 2 * stft(t * 0.0).size),
}


def to_bin_axis(grid, row_to_bin):
    if row_to_bin is None:
        return grid
    order = np.argsort(row_to_bin)
    xb = np.asarray(row_to_bin, float)[order]
    G = grid[order]
    bins = np.arange(nfreq)
    return np.stack([np.interp(bins, xb, G[:, c]) for c in range(grid.shape[1])], axis=1)


def run_comparison(signal, title, fname, inst=None, hlines=(), vlines=()):
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = {name: to_bin_axis(gf(signal), r2b) for name, (gf, r2b, _) in METHODS.items()}
    if inst is not None:
        mid = slice(N // 8, 7 * N // 8)
        print(f"\n{title}: chirp ridge MAE (bins, lower=better)")
        for name, gr in grids.items():
            mae = np.mean(np.abs(gr[:, mid].argmax(axis=0) - inst[mid]))
            print(f"  {name:18s} {mae:6.2f}")

    ncol = 5
    nrow = int(np.ceil(len(grids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow),
                             sharex=True, sharey=True)
    for ax, (name, gr) in zip(axes.ravel(), grids.items()):
        dof = METHODS[name][2]
        ax.imshow(gr, origin="lower", aspect="auto", cmap="magma", extent=[0, N, 0, nfreq])
        if inst is not None:
            ax.plot(t, inst, "c--", lw=0.7)
        for hl in hlines:
            ax.axhline(hl, color="lime", ls=":", lw=0.6)
        for vl in vlines:
            ax.axvline(vl, color="cyan", ls=":", lw=0.6)
        ax.set_ylim(0, 200)
        ax.set_title(f"{name}  ({dof/N:.0f}x DOF)", fontsize=8)
    for ax in axes.ravel()[len(grids):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=95)
    plt.close(fig)
    print(f"  saved {fname}")


# %% [markdown]
# ## Static parsimony table (signal-independent)

# %%
print(f"\n{'method':18s} {'real DOF':>9s} {'redundancy':>11s}")
for name, (_, _, dof) in METHODS.items():
    print(f"{name:18s} {dof:9d} {dof / N:9.1f}x")

# %% [markdown]
# ## Signal 1 — impulse at centre (shows TIME resolution at each frequency)
# A spike is flat in frequency; each transform's vertical ridge widens where its
# time resolution is poor. Constant-Q => narrow at high freq, wide at low freq.

# %%
imp = np.zeros(N); imp[N // 2] = 1.0
run_comparison(imp, "Signal 1: impulse at t=N/2 (temporal resolution per freq)",
               "/tmp/cmp_impulse.png", vlines=[N // 2])

# %% [markdown]
# ## Signal 2 — up-chirp starting low (15 -> 85), no tone

# %%
g0, g1 = 15, 85
inst2 = g0 + (g1 - g0) * t / N
chirp2 = np.cos(2 * np.pi * (g0 * t + (g1 - g0) * t**2 / (2 * N)) / N)
run_comparison(chirp2, f"Signal 2: up-chirp {g0}->{g1} bins",
               "/tmp/cmp_chirp_low.png", inst=inst2)

# %% [markdown]
# ## Signal 3 — three short tones at different times, partially overlapping

# %%
def gated_tone(f, t0, t1, taper=0.2):
    lo, hi = int(t0 * N), int(t1 * N)
    env = np.zeros(N)
    seg = hi - lo
    w = np.ones(seg)
    r = max(int(taper * seg), 1)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
    w[:r] *= ramp; w[-r:] *= ramp[::-1]
    env[lo:hi] = w
    return env * np.cos(2 * np.pi * f * t / N)


three = (gated_tone(30, 0.10, 0.50)
         + gated_tone(70, 0.35, 0.75)
         + gated_tone(140, 0.60, 0.95))
run_comparison(three, "Signal 3: tones f=30,70,140 at staggered, overlapping times",
               "/tmp/cmp_three_tones.png", hlines=[30, 70, 140])

print("\ndone")
