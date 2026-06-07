# %% [markdown]
# # Prototype: fixing tiling-A's octave-join dead zone
#
# Numpy-only exploration (no Rust changes yet). Compares three ways to keep
# tiling A's clean, regular dyadic grid while not losing energy at the octave
# joins:
#
# - **baseline_A**: current `dyadic_real` — one centred Gaussian per band,
#   inverted by division. Invertible, but a tone on a join produces tiny
#   coefficients (the "dead zone").
# - **two_channel_A**: each band keeps its grid cell but stores TWO coefficients
#   — a *centre* window `sin(θ)` and an *edge* window `cos(θ)` (θ: 0→π across the
#   band). `sin²+cos²=1` ⇒ power-complementary ⇒ tight frame ⇒ exact inverse by
#   matched synthesis (no division, robust to window zeros). This is the user's
#   "capture what the centred Gaussian misses, in the same window" idea, made
#   exact.
# - **pou_A**: one channel per band, but neighbouring bands OVERLAP at the joins
#   with a `cos/sin` cross-fade (`Σ window² = 1`), so a join tone is shared
#   smoothly between the two adjacent cells. Partition-of-unity, tight frame.
#
# Metrics: perfect-reconstruction error, coefficient count, join-vs-centre tone
# concentration, and how well a chirp's ridge tracks the true instantaneous
# frequency (the artefact that motivated this).

# %%
import numpy as np

N = 512
t = np.arange(N)


def a_bands(n):
    """Tiling-A dyadic band edges: [0,1) [1,2) [2,4) ... [n/4, n/2+1)."""
    bands = []
    prev, edge = 0, 1
    while edge <= n // 4:
        bands.append((prev, edge))
        prev, edge = edge, edge * 2
    bands.append((prev, n // 2 + 1))
    return bands


BANDS = a_bands(N)


def gauss_screen(lo, hi):
    """Centred Gaussian screen over band bins [lo,hi), σ_f = fcentre/2π."""
    w = hi - lo
    if w == 1:
        return np.ones(1)
    fc = lo + w // 2
    sigma = fc / (2.0 * np.pi)
    k = np.arange(lo, hi)
    return np.exp(-((k - fc) ** 2) / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Scheme 1: baseline A (one centred Gaussian per band, invert by division)
# ---------------------------------------------------------------------------
def baseline_fwd(x):
    X = np.fft.rfft(x)
    chans = []
    for lo, hi in BANDS:
        s = gauss_screen(lo, hi)
        chans.append(np.fft.ifft(X[lo:hi] * s))
    return chans


def baseline_inv(chans):
    X = np.zeros(N // 2 + 1, dtype=complex)
    for (lo, hi), c in zip(BANDS, chans):
        s = gauss_screen(lo, hi)
        X[lo:hi] = np.fft.fft(c) / s  # division inverse (s>0 required)
    return np.fft.irfft(X, N)


# ---------------------------------------------------------------------------
# Scheme 2: two-channel A (centre sin(θ) + edge cos(θ), power-complementary)
# ---------------------------------------------------------------------------
def two_channel_windows(lo, hi):
    w = hi - lo
    if w == 1:
        return np.ones(1), np.zeros(1)
    theta = np.pi * (np.arange(w) + 0.5) / w  # 0→π across the band
    return np.sin(theta), np.cos(theta)  # centre peaks mid, edge peaks at ends


def two_channel_fwd(x):
    X = np.fft.rfft(x)
    out = []
    for lo, hi in BANDS:
        wc, we = two_channel_windows(lo, hi)
        out.append((np.fft.ifft(X[lo:hi] * wc), np.fft.ifft(X[lo:hi] * we)))
    return out


def two_channel_inv(out):
    X = np.zeros(N // 2 + 1, dtype=complex)
    for (lo, hi), (cc, ce) in zip(BANDS, out):
        wc, we = two_channel_windows(lo, hi)
        # matched synthesis: fft(ifft(X*w))*w = X*w²; Σ = X(wc²+we²) = X
        X[lo:hi] += np.fft.fft(cc) * wc + np.fft.fft(ce) * we
    return np.fft.irfft(X, N)


# ---------------------------------------------------------------------------
# Scheme 3: partition-of-unity A (overlapping bands, cos/sin cross-fade)
# ---------------------------------------------------------------------------
def pou_layout(n):
    """Bands extended to overlap at each interior join with a cos/sin cross-fade.
    Returns list of (lo, hi, window) with Σ window² = 1 across all bands."""
    base = a_bands(n)
    nf = n // 2 + 1
    # interior joins are the shared boundaries between consecutive bands
    joins = [hi for (_, hi) in base[:-1]]
    # overlap half-width per join: constant-Q (∝ frequency), but never wider than
    # half the smaller adjacent band, and 0 for the tiny low-frequency bands.
    spans = {}
    for j_idx, J in enumerate(joins):
        lo_l, hi_l = base[j_idx]
        lo_r, hi_r = base[j_idx + 1]
        wl, wr = hi_l - lo_l, hi_r - lo_r
        delta = int(round(0.25 * J))
        delta = min(delta, wl // 2, wr // 2)
        spans[J] = max(delta, 0)

    out = []
    for bi, (lo, hi) in enumerate(base):
        e_lo = lo - spans.get(lo, 0)  # extend down into the lower join overlap
        e_hi = hi + spans.get(hi, 0)  # extend up into the upper join overlap
        e_lo = max(e_lo, 0)
        e_hi = min(e_hi, nf)
        win = np.ones(e_hi - e_lo)
        # lower overlap [lo-δ, lo+δ): this band ramps up sin(0→π/2)
        d = spans.get(lo, 0)
        if d > 0:
            seg = np.arange(lo - d, lo + d)
            phi = np.pi / 2 * (seg - (lo - d)) / (2 * d)
            win[(seg - e_lo)] = np.sin(phi)
        # upper overlap [hi-δ, hi+δ): this band ramps down cos(0→π/2)
        d = spans.get(hi, 0)
        if d > 0:
            seg = np.arange(hi - d, hi + d)
            phi = np.pi / 2 * (seg - (hi - d)) / (2 * d)
            win[(seg - e_lo)] = np.cos(phi)
        out.append((e_lo, e_hi, win))
    return out


POU = pou_layout(N)


def pou_fwd(x):
    X = np.fft.rfft(x)
    return [np.fft.ifft(X[lo:hi] * win) for (lo, hi, win) in POU]


def pou_inv(chans):
    X = np.zeros(N // 2 + 1, dtype=complex)
    for (lo, hi, win), c in zip(POU, chans):
        X[lo:hi] += np.fft.fft(c) * win  # Σ win² = 1 ⇒ exact
    return np.fft.irfft(X, N)


# %% [markdown]
# ## 1. Perfect reconstruction + coefficient count
#
# All three must invert; we compare how many coefficients each stores.

# %%
rng = np.random.default_rng(0)
x = rng.standard_normal(N)

schemes = {
    "baseline_A": (baseline_fwd, baseline_inv,
                   sum(hi - lo for lo, hi in BANDS)),
    "two_channel_A": (two_channel_fwd, two_channel_inv,
                      2 * sum(hi - lo for lo, hi in BANDS)),
    "pou_A": (pou_fwd, pou_inv,
              sum(hi - lo for lo, hi, _ in POU)),
}

print(f"signal length N = {N}   (dyadic_real packs {N//2+1}, dyadic_dual packs {N-1})")
print(f"{'scheme':16s} {'coeffs':>7s}  {'recon error':>12s}")
for name, (fwd, inv, ncoef) in schemes.items():
    err = np.max(np.abs(inv(fwd(x)) - x))
    print(f"{name:16s} {ncoef:7d}  {err:12.2e}")


# %% [markdown]
# ## 2. Join tone vs centre tone — is the dead zone gone?
#
# Total represented energy for a tone at a band centre (f=48) vs exactly on the
# octave join (f=64). baseline_A should collapse at the join; the other two
# should not.

# %%
def total_energy(fwd, f):
    sig = np.cos(2 * np.pi * f * t / N)
    chans = fwd(sig)
    e = 0.0
    for c in chans:
        if isinstance(c, tuple):
            e += sum(float(np.sum(np.abs(ci) ** 2)) for ci in c)
        else:
            e += float(np.sum(np.abs(c) ** 2))
    return e


print(f"{'scheme':16s} {'E(f=48 centre)':>15s} {'E(f=64 join)':>14s} {'join/centre':>12s}")
for name, (fwd, _, _) in schemes.items():
    ec, ej = total_energy(fwd, 48), total_energy(fwd, 64)
    print(f"{name:16s} {ec:15.1f} {ej:14.1f} {ej/ec:12.3f}")


# ---------------------------------------------------------------------------
# Scheme 2b: monotonic two-channel A (low cos(θ) + high sin(θ), θ: 0→π/2)
# The centre/edge pair above is symmetric: the edge channel peaks at BOTH joins,
# so it can't tell lower-join from upper-join content (ambiguous within a band).
# A monotonic low/high ramp pair removes that ambiguity and gives clean
# within-octave frequency localisation — better for reading off a ridge.
# ---------------------------------------------------------------------------
def ramp_windows(lo, hi):
    w = hi - lo
    if w == 1:
        return np.ones(1), np.zeros(1)
    theta = (np.pi / 2) * (np.arange(w) + 0.5) / w  # 0→π/2
    return np.cos(theta), np.sin(theta)  # low peaks at lo, high peaks at hi


def ramp_fwd(x):
    X = np.fft.rfft(x)
    out = []
    for lo, hi in BANDS:
        wl, wh = ramp_windows(lo, hi)
        out.append((np.fft.ifft(X[lo:hi] * wl), np.fft.ifft(X[lo:hi] * wh)))
    return out


def ramp_inv(out):
    X = np.zeros(N // 2 + 1, dtype=complex)
    for (lo, hi), (cl, ch) in zip(BANDS, out):
        wl, wh = ramp_windows(lo, hi)
        X[lo:hi] += np.fft.fft(cl) * wl + np.fft.fft(ch) * wh
    return np.fft.irfft(X, N)


schemes["two_channel_ramp"] = (
    ramp_fwd, ramp_inv, 2 * sum(hi - lo for lo, hi in BANDS)
)


# %% [markdown]
# ## 3. Chirp ridge tracking (the artefact that started this)
#
# Build a magnitude grid for each scheme (spreading each band's envelope across
# its bins with the same window it was analysed with) and read off the ridge
# (argmax per time column). Compare to the chirp's true instantaneous frequency.
# Single-channel schemes can't localise *within* an octave (the band is the
# resolution), so their ridge snaps to band structure; the two-channel schemes
# use the channel balance to place the ridge inside the octave.

# %%
def time_interp(mag):
    w = len(mag)
    if w == 1:
        return np.full(N, mag[0])
    return np.interp(np.arange(N), np.linspace(0, N - 1, w), mag)


# Width normalisation: |coeff| ∝ 1/width for a fixed-amplitude line, so scale
# each band's envelope by its width to equalise brightness across the frequency
# axis (otherwise high-frequency = wide bands render dimmer for equal energy).
# ``width_norm`` controls per-band brightness scaling (display only):
#   True  -> x width  (L1: equal-amplitude tones equally bright; impulse tilts up)
#   False -> x1       (raw: flat-spectrum impulse stays flat; tones dim at high f)
def _scale(w, width_norm):
    return float(w) if width_norm else 1.0


def grid_baseline(x, width_norm=True):
    g = np.zeros((N // 2 + 1, N))
    for (lo, hi), c in zip(BANDS, baseline_fwd(x)):
        s = _scale(hi - lo, width_norm)
        g[lo:hi, :] += np.abs(gauss_screen(lo, hi))[:, None] * (s * time_interp(np.abs(c)))
    return g


def grid_two_channel(x, width_norm=True):
    g = np.zeros((N // 2 + 1, N))
    for (lo, hi), (cc, ce) in zip(BANDS, two_channel_fwd(x)):
        s = _scale(hi - lo, width_norm)
        wc, we = two_channel_windows(lo, hi)
        g[lo:hi, :] += (np.abs(wc)[:, None] * (s * time_interp(np.abs(cc)))
                        + np.abs(we)[:, None] * (s * time_interp(np.abs(ce))))
    return g


def grid_ramp(x, width_norm=True):
    g = np.zeros((N // 2 + 1, N))
    for (lo, hi), (cl, ch) in zip(BANDS, ramp_fwd(x)):
        s = _scale(hi - lo, width_norm)
        wl, wh = ramp_windows(lo, hi)
        g[lo:hi, :] += (np.abs(wl)[:, None] * (s * time_interp(np.abs(cl)))
                        + np.abs(wh)[:, None] * (s * time_interp(np.abs(ch))))
    return g


def grid_pou(x, width_norm=True):
    g = np.zeros((N // 2 + 1, N))
    for (lo, hi, win), c in zip(POU, pou_fwd(x)):
        s = _scale(hi - lo, width_norm)
        g[lo:hi, :] += np.abs(win)[:, None] * (s * time_interp(np.abs(c)))
    return g


GRIDDERS = {
    "baseline_A": grid_baseline,
    "two_channel_A": grid_two_channel,
    "two_channel_ramp": grid_ramp,
    "pou_A": grid_pou,
}

f0, f1 = 40, 110
inst = f0 + (f1 - f0) * t / N
phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * N)) / N
chirp = np.cos(phase)

mid = slice(N // 8, 7 * N // 8)  # ignore the very ends
print(f"\nchirp ridge tracking (lower = better)   true f: {f0}->{f1}")
print(f"{'scheme':18s} {'ridge MAE (bins)':>16s}")
grids = {}
for name, gf in GRIDDERS.items():
    g = gf(chirp)
    grids[name] = g
    ridge = g.argmax(axis=0)
    mae = float(np.mean(np.abs(ridge[mid] - inst[mid])))
    print(f"{name:18s} {mae:16.2f}")

# Optional figure (only if matplotlib + a writable backend are available).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(g.max() for g in grids.values())
    fig, axes = plt.subplots(1, len(grids), figsize=(20, 4.2), sharex=True, sharey=True)
    for ax, (name, g) in zip(axes, grids.items()):
        ax.imshow(g, origin="lower", aspect="auto", extent=[0, N, 0, g.shape[0]],
                  cmap="magma", vmin=0, vmax=vmax)
        ax.plot(t, inst, "c--", lw=1.0, label="true inst. freq")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("time")
        ax.set_ylim(0, 140)
    axes[0].set_ylabel("freq bin")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Chirp ridge: dashed = truth. Single-channel snaps to octaves; "
                 "two-channel resolves within-octave.")
    fig.tight_layout()
    fig.savefig("/tmp/edge_schemes_chirp.png", dpi=95)
    print("\nsaved /tmp/edge_schemes_chirp.png")
except Exception as e:  # pragma: no cover
    print(f"(figure skipped: {e})")
