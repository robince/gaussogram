# gaussogram — Rust Implementation Specification

> **gaussogram** — a parsimonious, memory-efficient *adaptive* time-frequency
> representation (a fast Gaussian S-transform / GFT), offered as an alternative to wavelet
> transforms. The name: **Gauss**ian windows + octave bands + **-gram** (a time-frequency
> map). Distributed as the Rust crate(s) `gaussogram-*`, the Python package `gaussogram`,
> and the MATLAB package `+gaussogram`.

**Status:** Design spec, ready for implementation.
**Audience:** An engineer/agent implementing this from scratch, *without* access to the
design conversation that produced it. Everything you need is in this document plus the
existing C source (`gft.c`, `gft.h`) and Cython bindings (`pygft/_core.pyx`) in this repo,
which serve as the numerical reference.

---

## 0. TL;DR / Scope

We are reimplementing the **General Fourier Family Transform (GFT)** — the fast,
frequency-domain S-transform of Brown, Lauzon & Frayne (2010, *IEEE Trans. Signal
Processing* 58(1):281–290) — as a Rust workspace, distributed under the name
**`gaussogram`**, with **first-class Python and MATLAB bindings**. (The underlying
algorithm is still the fast GFT; **gaussogram** is the brand/package for this
implementation and its new default scheme. References below to `gft.c`, `pygft`, `+gft`,
etc. are the *existing* code we port **from**.)

Three things this port delivers over the current C/Cython code:

1. **A new default partitioning scheme** (`dyadic_dual_real`) that fixes the band-edge
   information-loss problem of the current "real partitioning" fork, using a clean
   half-octave-offset dual-tree dyadic tiling. Total output size **N − 1** complex
   coefficients for a length-N real signal.
2. **A clean, flexible core** (`Scheme`/`Band`/`Window` types, an `Fft1d` trait) that
   makes the partitioning scheme data, not hard-coded control flow — so 2D and future
   schemes drop in without rewrites.
3. **Performance**: plan reuse + per-thread scratch + `rayon` batch parallelism over
   independent segments, with the FFT backend behind a trait so we can start on pure-Rust
   `rustfft` and switch to FFTW via a cargo feature when it wins.

**Primary use case:** memory-efficient time-frequency *feature extraction* for training
predictive models on large datasets, run in tight loops. The forward transform and its
packed output layout are the hot path. An inverse transform is desirable for the lossless
scheme but is **not** performance-critical.

We focus on **1D** now. 2D must be *designed for* (see §9) but not implemented yet.

---

## 1. Mathematical background

### 1.1 The S-transform and the GFT

The S-transform is a windowed Fourier transform whose Gaussian window width scales
inversely with frequency (constant-Q, like a wavelet): low frequencies get wide windows
(good frequency resolution, poor time resolution); high frequencies get narrow windows
(good time resolution, poor frequency resolution). It gives a progressive,
multi-resolution time-frequency picture.

The naive S-transform is O(N²) per frequency. The **fast GFT** computes the whole thing in
**O(N log N)**:

1. Take one length-N FFT of the signal → `X[k]`.
2. Multiply `X` by a precomputed **window screen** (the FFTs of the per-frequency Gaussian
   windows, laid end to end — one contiguous screen of length N).
3. Partition the spectrum into **bands** (octaves). For each band, take an **inverse FFT of
   just that band's slice**. The IFFT lengths sum to N, so the total cost is dominated by
   the single length-N forward FFT plus a set of smaller IFFTs.

The output is **one complex coefficient per spectral bin**: a band of width `w` produces a
length-`w` IFFT → `w` complex time samples at that band's centre frequency. Total output =
sum of band widths = number of bins covered.

### 1.2 Width ↔ resolution tradeoff (get this right)

For a band of width `w`:

- The band spans `w` frequency bins → **frequency resolution ∝ 1/w** (wide band = coarse
  frequency).
- Its IFFT yields `w` time samples across the whole signal duration → **time resolution ∝
  w** (wide band = fine time).

So **wide bands give good time resolution**, narrow bands give good frequency resolution.
In a dyadic octave layout the *highest* octave (e.g. `[N/4, N/2)`) is the **widest** band
(width `N/4`) and therefore has the *best* time resolution — this is correct and desirable
for transient/high-frequency content. The only unavoidable width-1 bands are DC and the
single Nyquist coefficient. (A common intuition that "the top frequency has poor time
resolution" is backwards — only the lone Nyquist bin is width-1, not the whole top octave.)

Width-1 bands (DC, the lowest frequency) are **exact**: a width-1 window has no Gaussian
taper and therefore no "dead zone," so a tone landing exactly there is captured fully.

### 1.3 Why the current "real" scheme loses information (the bug we fix)

A Gaussian window per band tapers to near-zero at the **band edges**. A narrowband tone
sitting at a band boundary falls in this taper "dead zone" of *both* adjacent bands and is
attenuated in the representation — information loss exactly at band edges.

The fork's existing `gft_1dRealPartitions` tried to fix this for real signals with an
asymmetric "offset" tiling, but its offset **drifts**: the offset-band edges land at
`|f| = 2, 5, 10, 19, 36, 69, …`, systematically *below* the half-octave-ideal positions.
The consequence is that tiling A's high octave boundaries (e.g. 32, 64 for N=256) end up
*near* the offset band edges and stay inside dead zones — defeating the purpose. It also
produces a width-59 leftover top band (barely offset; centre 98.5 vs A's 96),
**prime-length IFFTs** (17, 59, arising from the `2^k+1` width pattern — slow/awkward for
FFT planners), and folds Nyquist into the leftover band.

### 1.4 Conjugate symmetry → free second view

For a real signal, `X[N−k] = conj(X[k])`, so the negative half-spectrum is redundant. The
current fork scheme packs an asymmetric real tiling into N coefficients but does so by
filling the negative (upper) bins from the Nyquist end with `2^k+1` widths, which makes the
offset *drift* (§1.3). **Our approach instead** spends the same ~N budget as a proper
**dual-tree** decomposition:

- **Tiling A** — a clean lossless dyadic tiling of the positive half-spectrum
  (`N/2 + 1` coefficients). This alone is complete: A captures everything; B only improves
  the band-edge response.
- **Tiling B** — a *second dyadic octave bank offset by half an octave*, with each band
  **centred symmetrically on one of A's band boundaries** (the joins where A's Gaussians
  taper to zero — A's dead zones). Band B for join `b` is `[b/2, 3b/2)`, width `b`.

This is exactly the dual-tree complex wavelet construction (two dyadic banks offset by half
an octave) applied to the GFT, and it has the property the fork was reaching for but missed:
**every one of A's dead zones sits at the exact peak of a B band, at every scale** — no
drift. Conversely B's own band edges fall on A's band centres, where A is strongest. So no
frequency lands in a dead zone of both tilings. A 2× overcomplete dyadic frame of a real
signal naturally costs ~N complex coefficients, so the total of **`N − 1`** (see §2.1) is
the principled size for this representation, not a compromise — and it is spent far better
than the original complex GFT's `[positive half] + [redundant conjugate negative half]`.

---

## 2. The partitioning schemes

All schemes operate on a length-`N` real signal where **`N` is a power of two**. Validate
and reject otherwise with an error mentioning "power of two". **Per-scheme minimum size:**

- `dyadic_real`, `dyadic_complex`: `N ≥ 4`.
- `dyadic_dual_real`: **`N ≥ 8`** (B needs at least one interior octave join above the exact
  low bands). Reject `N < 8` for this scheme with a clear error.

Conventions:
- Bands are **half-open** `[lo, hi)` over spectral bin indices.
- "Boundaries" lists the half-open **right edges** in increasing order.
- "Width" `= hi − lo`. A band of width `w` produces `w` complex output coefficients via a
  length-`w` IFFT.

### 2.1 `dyadic_dual_real` — the DEFAULT scheme

Two tilings, concatenated in the output buffer: **A** then **B**.

**Tiling A — lossless dyadic cover of the positive half-spectrum** (bins `0 .. N/2`,
Nyquist folded into the top band; Nyquist edge handled per §3.2):

- Right edges: `[1, 2, 4, 8, …, N/4, N/2 + 1]`
- Bands / widths:
  - `{0}` width 1 (DC, exact)
  - `[1, 2)` width 1 (exact)
  - `[2, 4)` width 2
  - `[4, 8)` width 4
  - … (octaves) …
  - `[N/8, N/4)` width `N/8`
  - `[N/4, N/2]` width `N/4 + 1` (top octave **including** Nyquist)
- Count: **N/2 + 1** coefficients.

**Tiling B — half-octave-offset dyadic bank, each band centred on an A join.** For each of
A's interior octave boundaries (its dead zones) `b ∈ {2, 4, 8, …, N/4}`, B has one band:

- Band: `[b/2, 3b/2)`, **width `b`**, peak (window centre) **exactly on `b`** — a clean
  *symmetric* Gaussian centred on the join it rescues.
- Right edges: `[3, 6, 12, 24, …, 3N/4]` (= `3b/2` for each `b`).
- Bands / widths:
  - `b=2`:   `[1, 3)`    width 2
  - `b=4`:   `[2, 6)`    width 4
  - `b=8`:   `[4, 12)`   width 8
  - `b=16`:  `[8, 24)`   width 16
  - … (each width = the join `b`) …
  - `b=N/4`: `[N/8, 3N/8)` width `N/4`
- Count: `2 + 4 + … + N/4` = **N/2 − 2** coefficients. Number of B bands: `log2(N) − 2`.

**Total output size:** `(N/2 + 1) + (N/2 − 2)` = **`N − 1`** complex coefficients.
Verified: `N=8 → 7`, `N=16 → 15`, `N=256 → 255`, `N=1024 → 1023`.

**Design invariant (the point of the scheme):**

- Each B band is **symmetric about A's join `b`** (`[b/2, 3b/2)`), so the join sits at the
  window *peak* — at *every* octave, with no drift. This is the key correction over the
  legacy fork scheme (§1.3), whose offset drifted so high-frequency joins fell back into
  dead zones.
- B band width `= b` = A's `[b, 2b)` octave width, so the rescued bin gets exactly the time
  resolution A's own design intends for it (bin `b` is the first bin of A's `[b,2b)` band).
  No time-resolution downgrade at the joins.
- B widths are **powers of two** (2, 4, 8, …, N/4) → radix-2 IFFTs.
- B and A **overlap** in source bins (each bin is read by 1–3 bands; see §5) — that overlap
  is inherent to a half-octave-offset frame and is what buys the no-dead-zone guarantee.

**Why B stops at `b = N/4` and has no band at Nyquist:** the only boundary above `N/4` is
Nyquist itself (`N/2`), which is a spectrum *endpoint*, not an interior join between two
tapering bands — so it needs no offset partner (it is handled within A's top band, §3.2).
The lowest joins (`b=2, 4`) sit between width-1/2 A bands whose Gaussians barely taper, so
their dead zones are negligible; B still covers them for uniformity but they could be
dropped with little effect.

**Worked example, N=256** (total `255 = N − 1`):

| Tiling | Band       | lo  | hi  | width | (B: rescues join) |
|--------|------------|-----|-----|-------|-------------------|
| A      | DC `{0}`   | 0   | 1   | 1     |                   |
| A      | `[1,2)`    | 1   | 2   | 1     |                   |
| A      | `[2,4)`    | 2   | 4   | 2     |                   |
| A      | `[4,8)`    | 4   | 8   | 4     |                   |
| A      | `[8,16)`   | 8   | 16  | 8     |                   |
| A      | `[16,32)`  | 16  | 32  | 16    |                   |
| A      | `[32,64)`  | 32  | 64  | 32    |                   |
| A      | `[64,128]` | 64  | 129 | 65    |                   |
| **A total** |       |     |     | **129** |                 |
| B      | `[1,3)`    | 1   | 3   | 2     | b=2               |
| B      | `[2,6)`    | 2   | 6   | 4     | b=4               |
| B      | `[4,12)`   | 4   | 12  | 8     | b=8               |
| B      | `[8,24)`   | 8   | 24  | 16    | b=16              |
| B      | `[16,48)`  | 16  | 48  | 32    | b=32              |
| B      | `[32,96)`  | 32  | 96  | 64    | b=64              |
| **B total** |       |     |     | **126** |                 |
| **Grand total** |   |     |     | **255 = N − 1** |       |

### 2.2 `dyadic_real` — baseline lossless single tiling

Tiling A alone (`N/2 + 1` coefficients). Lossless cover of the positive half-spectrum.
This scheme **supports an inverse transform** (round-trip to the original real signal up to
floating point). Provided as the simple, invertible baseline and for parity testing.

### 2.3 `dyadic_complex` — port of the original symmetric scheme

Faithful port of the C `gft_1dPartitions` / `gft_1dComplex64` symmetric dyadic scheme for
complex input (output length N). Needed for parity tests against the existing C code and
for users who want the classic behaviour. Behaviour, including the DC/edge special case
(`win[0]=1.0; win[2N−2]=1.0;` in the C window code) and the post-transform `shift` by
`N/2`, must match the C reference numerically (see §8).

### 2.4 Scheme registry

Schemes are **named, data-driven** values, not branches in the transform code. A `Scheme`
is fully described by its ordered list of `Band`s plus per-band window parameters (§3).
Adding a scheme = adding a constructor that returns a `Scheme`; the engine code does not
change. The default is `dyadic_dual_real`.

#### Public-API / output-length decision (resolved)

`dyadic_dual_real` outputs **`N − 1`** coefficients, which differs from the legacy
length-`N` contract of `gft1d_real`. **Decision: break the shape and make the new scheme
the default.**

- `gaussogram.gft1d_real(x)` (and the MATLAB `gaussogram.gft1dReal`) **return the new `N − 1`
  `dyadic_dual_real` result by default.** This is a deliberate breaking change to the public
  return shape.
- Provide an opt-in escape hatch for the old behaviour via a scheme selector, e.g.
  `gft1d_real(x, scheme="dyadic_real")` (length `N/2+1`, invertible) or
  `scheme="legacy")` if a byte-for-byte reproduction of the old `gft_1dRealPartitions`
  output is needed for parity. But the **default** is `dyadic_dual_real`.
- **Tests must be updated to the new contract:** the existing `tests/test_pygft.py`
  assertion that `gft1d_real` returns the same shape as its input (length `N`) is **wrong
  under the new default** and must be replaced with an assertion of length `N − 1` (and a
  separate test that `scheme="dyadic_real"` gives `N/2+1`). See §8.

---

## 3. Windows

### 3.1 Construction (match the C reference)

Per band, the window is a Gaussian in the **time domain** whose width is set by the band's
centre frequency (constant-Q), then normalized to unit sum, circularly shifted so its peak
is at sample 0, and FFT'd to the frequency domain. The frequency-domain Gaussians are laid
end-to-end into a length-N (tiling A) screen and multiplied against the signal spectrum.
See `gaussian`, `windows`, and `windowsFromPars` in `gft.c` for the exact reference
arithmetic — **reproduce their numerics**, not a from-scratch reinterpretation.

Reference centre-frequency formula for a band `[fstart, fstart+fwidth)`:
- positive bands: `fcentre = fstart + floor(fwidth/2)`
- negative bands (complex scheme): `fcentre = abs(-N + fstart) - floor(fwidth/2)`

**Fix the integer-truncation issues** present in the C/Cython code (see §10) — centre and
width math must be done consistently and documented. Where the reference truncates, decide
deliberately and write a test pinning the chosen behaviour.

### 3.2 Nyquist edge handling (tiling A top band)

The top band of tiling A includes the Nyquist bin `N/2`. This is **not** a behaviour of the
C reference: `windowsFromPars` (`gft.c` ~line 301) simply overwrites the top band's window
slots with the generated Gaussian slice, Nyquist included, so the Nyquist-adjacent bins sit
in the Gaussian taper and are attenuated. We must decide deliberately, and the two schemes
differ:

- **`dyadic_real` (invertible baseline): match the C reference exactly — no special edge
  handling.** Round-trip invertibility (§8) depends on the forward window being exactly what
  the inverse expects; do not perturb it. Document Nyquist-edge attenuation as a known,
  intended property of this scheme.

- **`dyadic_dual_real` (lossy feature scheme): optional flat-top at Nyquist**, controlled by
  an explicit `nyquist_flat_top: bool` field on the top `Window` (default **off**, i.e.
  match C). When on, construct the top band's frequency-domain window normally, then for
  every bin from the band **centre** `fcentre` up to and including Nyquist `N/2`, set the
  window value to its peak (the value at `fcentre`, which is the max after the unit-sum
  normalization). Concretely, for band `[lo, N/2]` with screen `w[0..width]` indexed so that
  `w[fcentre-lo]` is the peak: `for i in (fcentre-lo)..width { w[i] = w[fcentre-lo]; }`. This
  makes the upper half-window flat so Nyquist-adjacent content is not attenuated, at the
  cost of (intended) non-invertibility for this scheme. Pin the exact values with a test.

Represent the choice with the explicit `Window` field rather than special-casing inside the
FFT loop. The DC band (width 1) is exact and needs no taper regardless.

### 3.3 Window kinds

Support `gaussian` (default) and `box`, matching the C `gaussian`/`box` function pointers.
`box` must be accepted by the real transforms too (there is a Python test
`test_real_transform_accepts_box_window`). Reject unknown window names with an error
mentioning `window_type`.

---

## 4. Workspace / crate architecture

Mirror the proven layout in `~/code/stglm` (core / ffi / py three-crate split).

```
gaussogram/                   (cargo workspace root)
├── Cargo.toml                (workspace; shared deps & profiles)
├── crates/
│   ├── gaussogram-core/      (pure algorithm, no FFI, no Python)
│   │   └── Cargo.toml        crate-type rlib
│   ├── gaussogram-ffi/       (C ABI shim for MATLAB / other C callers)
│   │   └── Cargo.toml        crate-type ["cdylib", "rlib"]
│   └── gaussogram-py/        (PyO3 extension module)
│       └── Cargo.toml        crate-type ["cdylib", "rlib"], name "_native"
├── python/                   (gaussogram package; thin wrappers over _native)
├── matlab/                   (+gaussogram package + build_mex.m)
└── docs/
```

### 4.1 `gaussogram-core`

Pure Rust, no FFI, no NumPy. Public API (illustrative — adapt names to taste but keep the
shape):

```rust
// A band reads a CONTIGUOUS slice of the (full, conjugate-completed) spectrum
// [src_lo, src_hi) and writes its width complex outputs to a CONTIGUOUS range
// [out_off, out_off + width) of the packed coefficient buffer. src and out ranges
// are tracked separately because overlapping tilings (A and B) reuse source bins
// (see §5 / Finding "overlap").
pub struct Band {
    pub src_lo: usize,   // first spectrum bin (inclusive)
    pub src_hi: usize,   // last spectrum bin (exclusive); width = src_hi - src_lo
    pub out_off: usize,  // offset into the packed output buffer
}
impl Band { pub fn width(&self) -> usize { self.src_hi - self.src_lo } }

pub enum WindowKind { Gaussian, Box }

pub struct Window {
    pub kind: WindowKind,
    pub fcentre: usize,
    pub nyquist_flat_top: bool,   // §3.2; default false (matches C)
    // precomputed frequency-domain taper for THIS band, length = band width.
    // Applied per band against a COPY of the source slice (NOT a single global screen),
    // because A and B overlap in source bins with different windows. See §5.
    pub screen: Vec<f64>,
}

pub struct Scheme {
    pub name: &'static str,
    pub n: usize,
    pub complex_input: bool,   // true only for dyadic_complex
    pub bands: Vec<Band>,      // tiling A then tiling B, in output order
    pub windows: Vec<Window>,  // parallel to bands
    pub output_len: usize,     // sum of band widths (N-1 for dual, N/2+1 for real, N for complex)
    pub invertible: bool,
}

// Scheme constructors (data-driven; no engine branching):
pub fn dyadic_dual_real(n: usize) -> Result<Scheme, GaussogramError>;  // N>=8, output N-1, lossy
pub fn dyadic_real(n: usize)      -> Result<Scheme, GaussogramError>;  // N>=4, output N/2+1, invertible
pub fn dyadic_complex(n: usize)   -> Result<Scheme, GaussogramError>;  // N>=4, output N, complex input

// FFT backend behind a trait (see §6):
pub trait Fft1d {
    fn fft(&self, buf: &mut [Complex<f64>]);
    fn ifft(&self, buf: &mut [Complex<f64>]);
    fn plan(&self, len: usize) -> Arc<dyn Plan>;  // cached, reusable
}

pub struct Gaussogram1d { /* holds Scheme + cached plans + backend */ }
impl Gaussogram1d {
    pub fn new(scheme: Scheme, backend: impl Fft1d + 'static) -> Self;

    // REAL-input schemes (dyadic_real, dyadic_dual_real). Errors if scheme.complex_input.
    pub fn forward(&self, signal: &[f64], out: &mut [Complex<f64>]) -> Result<(), GaussogramError>;
    pub fn forward_batch(&self, signals: &[&[f64]], out: &mut [Complex<f64>]) -> Result<(), GaussogramError>; // rayon

    // COMPLEX-input scheme (dyadic_complex). Errors unless scheme.complex_input.
    pub fn forward_complex(&self, signal: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<(), GaussogramError>;

    // Inverse only where scheme.invertible (currently dyadic_real). Else NotInvertible.
    pub fn inverse(&self, coeffs: &[Complex<f64>], out: &mut [f64]) -> Result<(), GaussogramError>;
}

#[derive(thiserror::Error, Debug)]
pub enum GaussogramError {
    /* PowerOfTwo, TooSmall { scheme, min }, UnknownWindow, NotInvertible,
       WrongInputKind { expected_complex: bool }, OutputLenMismatch, … */
}
```

**Real vs complex input (Finding: contradiction).** The two input kinds take different code
paths and the API makes that explicit:
- `dyadic_real` / `dyadic_dual_real` consume `&[f64]`, start with a **real FFT**
  (`realfft`), and operate on the positive half-spectrum (conjugate-completing only if a
  band needs bins above `N/2`, which the real schemes do not).
- `dyadic_complex` consumes `&[Complex<f64>]`, starts with a **full complex FFT**, and
  covers the whole spectrum including the negative bands (hence its `fcentre` negative-band
  formula in §3.1). Calling `forward` on a complex scheme (or `forward_complex` on a real
  scheme) returns `GaussogramError::WrongInputKind`.

Key principles:
- **The scheme is data.** `forward*` walk `scheme.bands`; for each band they copy the
  source slice `[src_lo, src_hi)`, apply that band's own `screen`, IFFT, and write to
  `[out_off, out_off+width)`. No per-scheme `if`/`match` in the hot loop.
- **Contiguous-slice contract.** Each band's input is a contiguous spectral slice and its
  output is a contiguous range in the packed coefficient buffer. This contract is what
  makes 2D (transform rows, then columns of the transpose) drop in later (§9).
- `f64` complex throughout (matches C `gft_1dComplex64` / `complex128`). Use
  `num_complex::Complex<f64>`.

### 4.2 `gaussogram-ffi`

C ABI for MATLAB and any C caller. Pattern from `~/code/stglm/crates/stglm-ffi/src/lib.rs`:

- `crate-type = ["cdylib", "rlib"]`, depends on `gaussogram-core`.
- `#[repr(C)]` parameter/result structs; `#[no_mangle] pub extern "C"` functions.
- Integer **status codes** for errors (0 = OK); no panics across the boundary — wrap bodies
  in `std::panic::catch_unwind` and translate to an error code.
- **Opaque handle** struct for a constructed `Gaussogram1d` so MATLAB can build once and transform
  many segments (plan reuse across calls). Provide create/destroy/forward/forward_batch.
- Functions to query `output_len` and to fetch the partition vector (for parity with
  `gft_1dRealPartitions`).

### 4.3 `gaussogram-py`

PyO3 extension, mirroring `stglm-py`:

```toml
[lib]
name = "_native"
crate-type = ["cdylib", "rlib"]

[dependencies]
numpy = "0.24"
pyo3  = { version = "0.24", features = ["abi3-py310", "extension-module"] }
gaussogram-core = { path = "../gaussogram-core" }

[build-dependencies]
pyo3-build-config = "0.24"
```

- Built with **maturin**.
- Accept/return `numpy` arrays (`PyReadonlyArray1<f64>` in, `Py<PyArray1<Complex64>>` out).
- The Python-facing API lives in a thin `python/gaussogram/` package wrapping `_native` and
  is published as **`gaussogram`** (`import gaussogram`). It keeps the **same function
  names** as the legacy `pygft` package it replaces (`gaussogram.gft1d`,
  `gaussogram.gft1d_real`, `gaussogram.gft2d`, `gaussogram.partitions`,
  `gaussogram.real_partitions`, `gaussogram.interpolate_nn`, `gaussogram.interpolate_real`)
  so porting is mechanical. Per the §2.4 decision, `gft1d_real`'s **default return shape
  changes to `N − 1`** (with `scheme=` opt-out); update `tests/test_pygft.py` accordingly
  (§8). Other functions whose contract is unchanged keep their existing tests. (A
  `pygft`-named compatibility shim re-exporting `gaussogram` is optional, not required.)

---

## 5. Forward transform pipeline (1D)

For each input segment of length `N`:

1. **Forward FFT** of the signal → spectrum, kept **un-windowed**. For real schemes use a
   real FFT (`realfft`) giving the half-spectrum `0..=N/2`; for `dyadic_complex` use a full
   complex FFT over `0..N`. This spectrum is the **shared, read-only source** for every band.
2. **Per-band: copy → window → IFFT.** For each `Band { src_lo, src_hi, out_off }`:
   - copy the source slice `spectrum[src_lo..src_hi]` into a length-`width` scratch buffer;
   - multiply element-wise by **that band's own** `window.screen` (applying the Nyquist
     flat-top of §3.2 if set);
   - run the cached length-`width` inverse FFT in place;
   - write the `width` complex results to `out[out_off .. out_off + width]`.
3. Output is the concatenation of all bands (tiling A then B for `dyadic_dual_real`).

> **Critical (Finding: overlap).** Do **not** multiply the spectrum once by a single global
> screen and then slice it — that only works when the partitions tile the spectrum exactly
> once, which is true for the C reference (`gft.c` ~line 338) and for the single-tiling
> `dyadic_real`/`dyadic_complex`, but **not** for `dyadic_dual_real`. In the dual scheme A
> and B **reuse the same source bins** with **different windows** (e.g. bin 8 feeds both
> A's `[8,16)` and B's `[4,12)`). The source spectrum must therefore stay un-windowed
> and each band must apply its own window to a private copy. The single-global-screen
> optimisation is permitted only for single-tiling schemes; gate it on
> `scheme` having no overlapping `src` ranges (a property the constructor can flag), and
> keep the per-band-copy path as the correct general implementation.

**Shift convention:** the C complex path calls `shift(cdata, N, N/2)` after the transform;
the MATLAB real path has this **commented out**, while the Python real path does
`np.roll(result, n//2)`. This is an inconsistency in the current code (§10). For the port:
pick **one** documented convention per scheme, apply it identically across Python and
MATLAB, and pin it with a test. Recommended: no implicit roll in the core; expose the shift
as an explicit, documented option if needed for display.

---

## 6. FFT backend (trait + cargo features)

The FFT library is **behind the `Fft1d` trait** and selected by **cargo features**:

- `fft-rustfft` (**default**): pure-Rust `rustfft` + `realfft`. No system dependency, builds
  everywhere, ~80–95% of FFTW throughput. This is the default and what we ship first.
- `fft-fftw`: the `fftw` crate, for when benchmarks show it wins on the target hardware /
  sizes. Note the **MATLAB symbol-clash caveat**: MATLAB ships its own `libmwfftw3`;
  statically linking another FFTW into a MEX file can clash. The stglm MEX build sidesteps
  the analogous OpenBLAS issue by preferring MATLAB's own library via an env var
  (`STGLM_MATLAB_BLAS`); do the equivalent here, or simply keep MEX on the `fft-rustfft`
  backend (recommended) and reserve `fft-fftw` for the Python/standalone builds.

The trait must support **plan creation and caching**: building an FFT/IFFT plan is
expensive relative to executing it, and we run in tight loops over many same-length
segments. Plans are created once per distinct length when a `Gaussogram1d` is built and stored
(e.g. `Arc<dyn Plan>` keyed by length).

---

## 7. Performance & parallelism

The forward transform + packed output is the **hot path** (feature extraction on large
datasets, tight loops). Design accordingly:

- **Plan reuse:** all FFT/IFFT plans for a given `Scheme` (one length-N forward + the set of
  band-width inverses) are built once at `Gaussogram1d` construction and shared via `Arc`. Never
  plan inside the per-segment loop.
- **Per-thread scratch:** each band IFFT needs a scratch buffer; allocate per-thread (not
  per-call) to avoid churn.
- **Batch parallelism is the primary multicore strategy.** Parallelise the **outer loop
  over independent segments** with `rayon` (`par_iter` / `for_each_init` with per-thread
  scratch + shared `Arc` plans). This scales cleanly and is the realistic workload.
- **Do not** rely on FFTW's internal threading or try to parallelise the per-band IFFTs of a
  single transform — the bands are small and the outer batch loop gives far better
  utilisation. (This was explicitly considered and rejected: per-band parallelism has poor
  granularity; FFTW multithreading helps single large transforms, not our many-small-IFFT
  pattern.)
- Keep everything `f64`-contiguous; avoid allocations in `forward`. Provide
  `forward_into(out: &mut [Complex<f64>])` so callers can reuse output buffers.

Benchmark with `criterion`; track throughput vs the C reference and vs the FFTW backend.

---

## 8. Testing strategy

Use `proptest` for invariants and standard `#[test]` for parity/numerics.

**Scheme invariants (`dyadic_dual_real`, proptest over valid N ≥ 8):**
- For each A interior octave boundary `b ∈ {2,4,…,N/4}` there is exactly one B band, equal
  to `[b/2, 3b/2)` with width `b`, **symmetric about `b`** (so the window peak/centre is
  exactly `b`).
- B band widths are the boundaries themselves (`2, 4, 8, …, N/4`) — all **powers of two**.
- Total output length `== N − 1` (A = `N/2 + 1`, B = `N/2 − 2`).
- Tiling A is a valid exact tiling of `0..=N/2` (contiguous, no gaps/overlaps). A and B
  **overlap** in source bins — that is expected and required (§5); the test asserts A is a
  well-formed exact tiling and each B band is the correct symmetric interval, not
  disjointness.
- No B band reaches Nyquist (top B band is `[N/8, 3N/8)`, and `3N/8 < N/2`).
- All band widths `≥ 1`; no zero-width band; IFFT lengths are powers of two.
- Reject `N < 8` and non-power-of-two `N` with the right errors.

**Parity vs existing C code:**
- Legacy partition vectors still reproduce: `real_partitions(8) == [1,2,4,6,8]`,
  `real_partitions(16) == [1,2,4,8,11,14,16]` (keep the legacy partition function available
  and correct for the `scheme="legacy"` opt-out path, even though the new default differs).
- `dyadic_complex` forward output matches C `gft_1dComplex64` to **~1e-12** on random
  inputs (build the C reference, or capture golden vectors from the current Cython build).

**Round-trip:**
- `dyadic_real.inverse(forward(x)) ≈ x` to ~1e-12 (lossless scheme is invertible).
- `dyadic_dual_real` is an overcomplete frame (overlapping tilings), invertible only via
  coverage normalisation where `coverage[k] = Σ_b |w_b[k]|² > 0`; the plain `inverse` should
  return `NotInvertible` unless that path is implemented (not required — inverse is not a
  priority for this scheme).

**Functional win (the point of the new scheme):**
- **Band-edge tone test:** synthesise pure tones at A's octave boundaries (the dead zones,
  `8, 16, 32, …`). Assert the `dyadic_dual_real` representation captures them with near-full
  energy via the corresponding B band — the tone sits exactly on that B band's symmetric
  **peak** — whereas the legacy single-tiling scheme attenuates them in its band-edge dead
  zone. Assertion: "B response at the boundary ≫ A's dead-zone response". This is the
  regression that justifies the new default.

**Binding-level:**
- shapes/dtypes/finiteness for `gft1d`, `gft1d_real`, `gft2d`, `interpolate_nn`,
  `interpolate_real`. **Updated contract:** `gft1d_real(x)` now returns length `N − 1`
  (default `dyadic_dual_real`) — replace the old "same shape as input" assertion in
  `tests/test_pygft.py`. Add a test that `gft1d_real(x, scheme="dyadic_real")` returns
  `N/2+1`.
- error cases: non-power-of-two rejected with "power of two"; `N<8` for the dual scheme
  rejected; unknown window with "window_type"; upsampling beyond core capacity with "less
  than or equal".

---

## 9. 2D — design for it now, implement later

Do **not** implement 2D yet, but the core must not preclude it:

- The 2D GFT is **separable**: transform all rows, then all columns. The C `gft_2dComplex64`
  does exactly this (rows then cols).
- The enabling contract is the **contiguous-slice** design (§4.1): a 1D transform consumes a
  contiguous spectral slice and writes a contiguous output range. For 2D, transform rows
  in place, **transpose** (so columns become contiguous rows), transform again, transpose
  back. Transpose-based separability keeps every 1D call cache-friendly and lets us reuse
  the exact same `Gaussogram1d` engine.
- Keep `Gaussogram1d::forward` free of any 1D-only assumptions about global buffer layout: it
  should operate on a passed-in slice + output slice, so a future `Gaussogram2d` can call it
  per row/column.
- The interpolation/packed-output helpers should accept a **partition vector** rather than
  hard-coding the dyadic walk (the C `gft_1d_interpolateNN` hard-codes `fstart = 2*fstart`
  and cannot accept a pars vector — do **not** repeat this; §10).

---

## 10. Bugs in the current code to NOT carry over

These are real defects in `gft.c` / `pygft/_core.pyx` / `+gft/*` — fix or avoid in the port,
and pin each decision with a test:

1. **fcentre off-by-one between transform and interpolation.** The C transform uses
   `fcentre = abs(-N + fstart) - fwidth/2`; the Cython `interpolate_real` and the MATLAB
   `gft1dRealInterpolate.m` use `abs(-N + fstart - 1) - fwidth/2` (the `.m` file even
   carries the comment `% !!! different to windowsFromPars`). Pick one correct convention
   and use it everywhere.
2. **Shift convention mismatch.** C complex path: `shift(cdata, N, N/2)`. MATLAB real path
   (`+gft/gft1dReal.c`): `shift(...)` **commented out**. Python real path (`_core.pyx`):
   `np.roll(result, n//2)`. Three behaviours for "the same" transform. Unify (§5).
3. **Integer truncation in window centring.** `floor(fwidth/2)` etc. are applied
   inconsistently; make the centre/width arithmetic explicit and tested.
4. **Interpolator can't take a partition vector.** `gft_1d_interpolateNN` hard-codes the
   dyadic walk (`fstart = 2*fstart`), so it only works for the one scheme. The new
   interpolation helpers must be parameterised by the scheme's actual `bands`.
5. **Prime-length IFFTs in the legacy real scheme.** `gft_1dRealPartitions` produces widths
   from a `2^k + 1` pattern → prime lengths like 17, 59 → slow/awkward FFT plans. The new
   `dyadic_dual_real` widths are all powers of two by construction — keep it that way.
6. **Tests are shallow.** Existing `tests/test_pygft.py` only checks shapes/finiteness and a
   couple of partition values — no round-trip, no band-edge, no numerical-parity tests. Add
   them (§8).

---

## 11. Build & packaging

- **Rust:** standard cargo workspace; `cargo test`, `cargo bench` (criterion).
- **Python:** `maturin` build of `gaussogram-py` (the `_native` module), wrapped by
  `python/gaussogram`. The distribution and import name is **`gaussogram`** (`pyproject.toml`
  `name = "gaussogram"`); the legacy `pygft` package is the code being ported from.
- **MATLAB:** mirror `~/code/stglm/matlab/build_mex.m`:
  - `cargo rustc -p gaussogram-ffi --release -- --crate-type staticlib` to produce a static lib;
  - `mex -R2018a` linking that static lib plus a small C/C++ shim that adapts MEX
    `mxArray`s to the `#[repr(C)]` FFI functions;
  - handle `MACOSX_DEPLOYMENT_TARGET`;
  - **prefer the `fft-rustfft` backend for MEX** to avoid the `libmwfftw3` symbol clash; if
    `fft-fftw` is ever needed in MEX, route through MATLAB's own FFTW the way stglm routes
    BLAS through MATLAB via `STGLM_MATLAB_BLAS`.
  - Ship the new MATLAB package as **`+gaussogram`** (`gaussogram.gft1d`,
    `gaussogram.gft1dReal`, `gaussogram.gft1dRealPartitions`, etc.), keeping the legacy
    function names so the existing `+gft` scripts (`test_par.m`, `test_st.m`) port by a
    namespace rename against the new backend.

---

## 12. Reference files in this repo

- `gft.c`, `gft.h` — numerical reference for window construction, partitioning, the
  forward/IFFT loop, and the 2D separable structure. **Match these numerics** for the
  parity tests.
- `pygft/_core.pyx`, `pyGFT.pyx`, `gftHeaders.pyx` — current Cython bindings (and the bugs
  in §10).
- `+gft/gft1d.c`, `+gft/gft1dReal.c`, `+gft/gft1dRealPartitions.c`,
  `+gft/gft1dRealInterpolate.m` — current MEX entry points and the MATLAB interpolation
  (with its flagged fcentre discrepancy).
- `tests/test_pygft.py` — current (shallow) test contract to preserve/extend.
- `~/code/stglm` — the reference Rust workspace architecture to mirror (core/ffi/py split,
  FFI patterns, `build_mex.m`).
