"""Regenerate the figures and metrics for ``reports/`` from the investigation
scripts, and copy them into the repo.

Run from the repo root:  ``python scripts/make_report.py``

Requires the dev dependencies (pywt, ssqueezepy, scipy, nsgt, dtcwt) used by the
cross-transform comparison — install with ``uv sync`` (the ``dev`` group).
"""

import os
import runpy
import shutil

import numpy as np

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

# The bands_per_octave report needs only gaussogram + matplotlib (no heavy deps)
# and writes straight into reports/figures, so run it first and independently.
print("== running compare_bands_per_octave.py ==")
runpy.run_path("scripts/compare_bands_per_octave.py", run_name="__main__")

# Regenerate the scratch figures (these scripts write into /tmp) and capture the
# comparison module's namespace so we can recompute the summary tables.
print("== running prototype_edge_schemes.py ==")
runpy.run_path("scripts/prototype_edge_schemes.py", run_name="__main__")
print("== running compare_existing.py ==")
ce = runpy.run_path("scripts/compare_existing.py", run_name="__main__")

# Copy the generated figures into the repo with stable names.
COPIES = {
    "/tmp/edge_schemes_chirp.png": f"{OUT}/edge_schemes_chirp.png",
    "/tmp/cmp_impulse.png": f"{OUT}/cross_impulse.png",
    "/tmp/cmp_chirp_low.png": f"{OUT}/cross_chirp_low.png",
    "/tmp/cmp_three_tones.png": f"{OUT}/cross_three_tones.png",
}
for src, dst in COPIES.items():
    shutil.copy(src, dst)
    print("wrote", dst)

# Extra table for the report: impulse temporal width (time isolation) per method.
N, nfreq = ce["N"], ce["nfreq"]
METHODS, to_bin = ce["METHODS"], ce["to_bin_axis"]
imp = np.zeros(N)
imp[N // 2] = 1.0


def tfwhm(row):
    pk = row.max()
    if pk <= 0:
        return np.nan
    idx = np.where(row > 0.5 * pk)[0]
    return idx.max() - idx.min() + 1


print("\n== impulse temporal width (FWHM samples; lower = better time isolation) ==")
print(f"{'method':18s} {'@bin40':>7s} {'@bin100':>8s} {'@bin180':>8s}")
for name, (gf, r2b, dof) in METHODS.items():
    G = to_bin(gf(imp), r2b)
    w = [tfwhm(G[b]) for b in (40, 100, 180)]
    print(f"{name:18s} {w[0]:7.0f} {w[1]:8.0f} {w[2]:8.0f}")

print("\ndone — figures in", OUT)
