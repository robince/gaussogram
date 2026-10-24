# Runtime benchmarks

Run from the repository root, using a release-built extension:

```sh
uv sync --extra benchmark --extra test
uv pip install maturin
# If a conda environment is also active, unset CONDA_PREFIX for maturin.
uv run maturin develop --release
uv run python benchmarks/run.py
uv run python benchmarks/report.py reports/benchmarks/latest/results.json
```

`run.py` has no dependency on the plotting code; `report.py` can regenerate plots
from saved results without rerunning any transforms. For an existing environment,
install `.[benchmark,test]` and use that environment's Python throughout.

```sh
# Full Gaussogram parameter sweep and baseline comparisons (can take minutes).
python benchmarks/run.py --preset full --output reports/benchmarks/full
# Larger compact transforms; avoid requesting dense transforms at these lengths.
python benchmarks/run.py --methods gaussogram dwt --sizes 16384 65536 262144
# Display cost, measured on precomputed coefficients.
python benchmarks/run.py --methods grid
# Optional comparisons using the repository's dev dependencies.
python benchmarks/run.py --methods dtcwt nsgt ssq_cwt
python -m pytest benchmarks
```

Quick: N=256, 1024, 4096; two real Gaussogram schemes at 1 and 4 bands/octave.
Full: powers of two from 256 through 16384; both schemes at 1, 2, 4, 8, 16
bands/octave, Gaussian/box windows, and both Nyquist flat-top values for the
dual scheme (the single scheme only supports false).
`--sizes` and `--methods` override the preset's sizes and method selection.
Grid and extended comparisons are opt-in for both presets.

A [sample quick report](../reports/benchmarks/quick/README.md) is included. It uses
`--memory-mib 1024` so the full Stockwell transform at N=4096 is measured too;
the stricter default budget skips that case. This is a local example, not a
portable speed guarantee.

## Timing protocol

Each case has a fresh, sequential subprocess. Imports, input generation, adapter
preparation and validation are outside the timed region. Inputs are contiguous
float64 seeded Gaussian noise; each method receives identical input at a given N.
Two warmups precede calibration. Repetition counts adapt to a 20 ms batch target
(capped at 65536 calls); seven batches yield raw per-call averages, their median,
and quartiles. Shorten these with `--target-seconds` and `--repeats` for smoke runs.
Wall time uses `perf_counter_ns`; it includes Python calls and output allocation/
disposal. Garbage collection is left at its normal setting. No timing thresholds
are used in tests.

Gaussogram reports reusable-handle **forward**, fresh handle **setup**, and
**one_shot** calls separately. DTCWT/NSGT expose separate constructor timings too.
For APIs without reusable plans, library setup remains inside forward calls.
Warm timings exclude lazy initialization/JIT compilation; setup measurements are
also warm, not fresh-process import or first-ever FFT planning measurements.
`grid` measures `to_grid` alone with default linear-time/block-frequency rendering.

All recognized thread environment variables are set before worker imports;
`--threads` defaults to 1. This requests thread limits, but is not a guarantee
that every backend honors them. Results record versions, platform, CPU count,
Git revision/status and settings. Keep other machine workloads quiet; compare
runs on the same machine and build. Build profile is established by the release
build command, not detectable from the installed extension.

The default 60-second **per-case timeout includes imports and warmups**. The
512 MiB memory budget checks conservative array-size estimates before execution;
it is **not an OS-enforced RSS limit**, and Python/JIT/library overhead is extra.
Oversized cases are skipped without reducing frequency resolution. Stockwell
at N=16384 is skipped by default. Configure `--memory-mib` and `--timeout` for
larger runs. Results are checkpointed after every case. Missing libraries,
timeouts, warnings and errors remain visible in JSON; actual errors cause a
nonzero exit. Dependencies missing from a run are marked unavailable.

## What is being compared

- Gaussogram: packed complex coefficients; Python bindings over Rust/rustfft.
- DWT: Haar and sym8, periodization, each wavelet's maximum useful decomposition
  level (`dwt_max_level`). Coefficient counts include the approximation.
- CWT: PyWavelets complex Morlet `cmor1.5-1.0`, FFT method, precision 12 when supported (older PyWavelets uses its fixed
  default, recorded in settings);
  12 voices/octave (full: 4, 12, 32), frequencies 1/N through 1/4 cycles/sample.
  The upper limit avoids the smallest scales' aliasing. This does not cover the
  full Nyquist range used by Stockwell.
- Stockwell: compiled `stockwell.st.st`, Gaussian window, gamma=1, every bin
  from DC through Nyquist and every time sample: `(N/2+1, N)` complex values.
  Small tests compare to an independent direct DFT definition, including the
  package's analytic-signal convention.
- STFT: SciPy Hann windows, 64 samples / 50% overlap (full also 256 samples and
  75% overlap), no boundary extension or tail padding.
- Extended: DTCWT includes lowpass and highpasses; NSGT spans 1 through N/2 with
  an octave scale; ssqueezepy computes GMW synchrosqueezing and returns the
  squeezed coefficients (its internal CWT work is included in time). DTCWT and
  NSGT's legacy NumPy 2 compatibility shims are confined to their workers and
  recorded in settings, matching the existing visual comparison script.

Coefficient counts count one complex number as one coefficient; output bytes
and actual dtypes expose the storage/precision difference. Output bytes are not
peak memory. Frequency coverage, padding, resolution and output cardinality
vary; speed ratios do not establish equivalent accuracy or transform quality.
The comparison plot uses the default Gaussian 1-band/octave Gaussogram settings;
the separate Gaussogram figure includes the full requested sweep. Dense display
costs have their own figure.

Sources: [PyWavelets CWT documentation](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html),
[Stockwell implementation](https://github.com/claudiodsf/stockwell).
