#!/usr/bin/env python3
"""Reproducible transform benchmarks; each case runs in a fresh subprocess."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS')
METHODS = ('gaussogram', 'dwt', 'cwt', 'stockwell', 'stft', 'grid',
           'dtcwt', 'nsgt', 'ssq_cwt')


def cases(sizes, methods, full=False):
    for n in sizes:
        for method in methods:
            settings = [{}]
            if method in ('gaussogram', 'grid'):
                settings = [dict(scheme=s, bands_per_octave=b, window_type=w,
                                 nyquist_flat_top=f)
                            for s, b, w, f in itertools.product(
                                ('dyadic_real', 'dyadic_dual_real'),
                                (1, 2, 4, 8, 16) if full else (1, 4),
                                ('gaussian', 'box') if full else ('gaussian',),
                                (False, True) if full else (False,))
                            if not f or s == 'dyadic_dual_real']
            elif method == 'dwt':
                settings = [dict(wavelet=w) for w in ('haar', 'sym8')]
            elif method in ('cwt', 'nsgt', 'ssq_cwt'):
                settings = [dict(voices=v) for v in ((4, 12, 32) if full else (12,))]
            elif method == 'stft':
                settings = [dict(window=w, overlap=o) for w, o in
                            itertools.product((64, 256) if full else (64,),
                                              (0.5, 0.75) if full else (0.5,)) if w <= n]
            for opts in settings:
                yield dict(method=method, n=n, settings=opts)


def estimated_bytes(case):
    """Conservative working-array estimate, not a hard RSS bound."""
    n, m, s = case['n'], case['method'], case['settings']
    if m in ('stockwell', 'grid'):
        return 6 * 16 * n * (n // 2 + 1)
    if m in ('cwt', 'ssq_cwt'):
        return 12 * 16 * n * (math.ceil(math.log2(n) * s['voices']) + 1)
    return 256 * n


def arrays(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from arrays(item)
    else:
        raise TypeError(f'Unexpected output type {type(value)}')


def prepare(case, x):
    """Return setup callable, prepared forward callable, optional one-shot call."""
    import numpy as np
    m, n, s = case['method'], case['n'], case['settings']
    # Legacy libraries need the same NumPy 2 compatibility as compare_existing.py.
    # These changes are confined to this case's disposable worker process.
    if m == 'dtcwt' and not hasattr(np, 'asfarray'):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
        s['compatibility'] = 'numpy2-asfarray'
    if m == 'nsgt' and int(np.__version__.split('.')[0]) >= 2:
        original_clip = np.clip
        def clip_compat(a, low, high, out=None, **kwargs):
            if out is not None and out.dtype.kind in 'iu':
                out[...] = original_clip(a, low, high, **kwargs).astype(out.dtype)
                return out
            return original_clip(a, low, high, out=out, **kwargs)
        np.clip = clip_compat
        s['compatibility'] = 'numpy2-integer-clip'
    setup = None
    one_shot = None
    if m in ('gaussogram', 'grid'):
        import gaussogram as g
        setup = lambda: g.Gaussogram1d(n, **s)
        handle = setup()
        forward = lambda: handle.forward(x)
        one_shot = lambda: g.gft1d_real(x, **s)
        if m == 'grid':
            coeffs = forward()
            forward = lambda: g.to_grid(coeffs, n, **s)
            setup = one_shot = None
    elif m == 'dwt':
        import pywt
        s['mode'] = 'periodization'
        s['level'] = pywt.dwt_max_level(n, s['wavelet'])
        forward = lambda: pywt.wavedec(x, **s)
    elif m == 'cwt':
        import pywt
        # Morlet frequency range 1/N .. 1/4 cycles/sample avoids tiny-scale aliasing.
        import inspect
        precision_kw = {'precision': 12} if 'precision' in inspect.signature(pywt.cwt).parameters else {}
        s.update(wavelet='cmor1.5-1.0', method='fft',
                 precision=precision_kw.get('precision', 'library default (not configurable)'))
        scales = 2.0 ** np.linspace(2, math.log2(n),
                                   (int(math.log2(n)) - 2) * s['voices'] + 1)
        s['scale_count'] = len(scales)
        forward = lambda: pywt.cwt(x, scales, s['wavelet'], method='fft', **precision_kw)[0]
    elif m == 'stockwell':
        from stockwell import st
        s.update(lo=0, hi=n // 2, gamma=1.0, win_type='gauss')
        forward = lambda: st.st(x, **s)
    elif m == 'stft':
        from scipy.signal import stft
        forward = lambda: stft(x, window='hann', nperseg=s['window'],
                              noverlap=int(s['window'] * s['overlap']),
                              boundary=None, padded=False)[2]
    elif m == 'dtcwt':
        import dtcwt
        setup = dtcwt.Transform1d
        h = setup()
        s['levels'] = max(1, int(math.log2(n)) - 3)
        def forward():
            p = h.forward(x, nlevels=s['levels'])
            return (p.lowpass, p.highpasses)
    elif m == 'nsgt':
        from nsgt import NSGT, OctScale
        setup = lambda: NSGT(OctScale(1, n / 2, s['voices']), fs=n, Ls=n, real=True)
        h = setup()
        forward = lambda: list(h.forward(x))
    elif m == 'ssq_cwt':
        from ssqueezepy import ssq_cwt
        s['wavelet'] = 'gmw'
        forward = lambda: ssq_cwt(x, wavelet='gmw', nv=s['voices'])[0]
    else:
        raise ValueError(m)
    return setup, forward, one_shot


def measure(fn, repeats, target):
    import numpy as np
    # Explicit warmups also exclude JIT initialization and lazy FFT planning.
    for _ in range(2):
        fn()
    loops = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if elapsed >= target or loops >= 65536:
            break
        loops = min(65536, max(loops + 1, int(loops * target / max(elapsed, 1e-9))))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - start) / 1e9 / loops)
    return dict(samples_s=samples, loops=loops, median_s=float(np.median(samples)),
                p25_s=float(np.percentile(samples, 25)), p75_s=float(np.percentile(samples, 75)))


def worker(case, repeats, target):
    import numpy as np
    x = np.random.default_rng(2026).standard_normal(case['n'])
    setup, forward, one_shot = prepare(case, x)
    result = forward()
    parts = list(arrays(result))
    if not parts or not all(a.size and np.isfinite(a).all() for a in parts):
        raise ValueError('Empty or nonfinite output')
    details = dict(coefficients=sum(a.size for a in parts),
                   output_bytes=sum(a.nbytes for a in parts),
                   output_shapes=[list(a.shape) for a in parts],
                   output_dtypes=sorted({str(a.dtype) for a in parts}))
    if case['method'] == 'gaussogram':
        import gaussogram as g
        assert details['coefficients'] == g.output_len(case['n'], **case['settings'])
        np.testing.assert_allclose(result, one_shot(), rtol=1e-12, atol=1e-12)
    n, method, settings = case['n'], case['method'], case['settings']
    if method in ('stockwell', 'grid'):
        assert result.shape == (n // 2 + 1, n)
    elif method == 'dwt':
        assert details['coefficients'] == n
    elif method == 'cwt':
        assert result.shape == (settings['scale_count'], n)
    elif method == 'stft':
        width = settings['window']
        hop = width - int(width * settings['overlap'])
        assert result.shape == (width // 2 + 1, 1 + (n - width) // hop)
    del result, parts
    timings = {'forward': measure(forward, repeats, target)}
    if setup is not None:
        timings['setup'] = measure(setup, repeats, target)
    if one_shot is not None:
        timings['one_shot'] = measure(one_shot, repeats, target)
    return dict(**case, status='ok', **details, timings=timings)


def metadata(args):
    versions = {}
    for pkg in ('numpy', 'scipy', 'gaussogram', 'PyWavelets', 'stockwell',
                'dtcwt', 'nsgt', 'ssqueezepy', 'matplotlib', 'maturin'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    def command(argv):
        return subprocess.run(argv, capture_output=True, text=True).stdout.strip()
    return dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                platform=platform.platform(), processor=platform.processor(),
                cpu_model=command(['sysctl', '-n', 'machdep.cpu.brand_string']) if sys.platform == 'darwin' else platform.processor(),
                machine=platform.machine(), cpu_count=os.cpu_count(),
                python=sys.version, executable=sys.executable, versions=versions,
                git_commit=command(['git', 'rev-parse', 'HEAD']),
                git_status=command(['git', 'status', '--short']),
                thread_environment={k: os.environ.get(k) for k in THREAD_VARS},
                ssq_parallel=os.environ.get('SSQ_PARALLEL'), arguments=vars(args))


def write_results(out, meta, results):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(dict(metadata=meta, results=results), indent=2) + '\n')
    with (out / 'results.csv').open('w', newline='') as f:
        fields = ['method', 'n', 'settings', 'status', 'reason', 'phase', 'median_s',
                  'p25_s', 'p75_s', 'loops', 'coefficients', 'output_bytes', 'samples_s']
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in results:
            base = {k: r.get(k, '') for k in fields[:5] + ['coefficients', 'output_bytes']}
            base['settings'] = json.dumps(base['settings'], sort_keys=True)
            for phase, timing in r.get('timings', {'': {}}).items():
                writer.writerow(dict(base, phase=phase, **timing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=['quick', 'full'], default='quick')
    p.add_argument('--methods', nargs='+', choices=METHODS,
                   default=['gaussogram', 'dwt', 'cwt', 'stockwell', 'stft'])
    p.add_argument('--sizes', nargs='+', type=int)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--target-seconds', type=float, default=0.02)
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--memory-mib', type=float, default=512)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', default='reports/benchmarks/latest')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        try:
            r = worker(json.loads(args.worker), args.repeats, args.target_seconds)
        except ImportError as e:
            r = dict(status='unavailable', reason=str(e))
        except Exception as e:
            r = dict(status='error', reason=f'{type(e).__name__}: {e}')
        print(json.dumps(r))
        return
    sizes = args.sizes or ([256, 1024, 4096] if args.preset == 'quick' else [2**i for i in range(8, 15)])
    if any(n < 32 or n & (n - 1) for n in sizes):
        p.error('sizes must be powers of two >= 32')
    if min(args.repeats, args.target_seconds, args.timeout, args.memory_mib, args.threads) <= 0:
        p.error('repeats, target, timeout, memory and threads must be positive')
    for k in THREAD_VARS:
        os.environ[k] = str(args.threads)
    os.environ['SSQ_PARALLEL'] = '0' if args.threads == 1 else '1'
    meta, results = metadata(args), []
    for c in cases(sizes, args.methods, args.preset == 'full'):
        r = dict(c)
        if estimated_bytes(c) > args.memory_mib * 2**20:
            r.update(status='skipped', reason='estimated working arrays exceed memory budget')
        else:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       '--worker', json.dumps(c), '--repeats', str(args.repeats),
                                       '--target-seconds', str(args.target_seconds)],
                                      capture_output=True, text=True, timeout=args.timeout)
                if proc.returncode:
                    r.update(status='error', reason=f'worker exit {proc.returncode}: {proc.stderr[-2000:]}')
                else:
                    r.update(json.loads(proc.stdout.splitlines()[-1]))
                    if proc.stderr:
                        r['warnings'] = proc.stderr[-4000:]
            except (json.JSONDecodeError, IndexError) as e:
                r.update(status='error', reason=f'Invalid worker response: {e}')
            except subprocess.TimeoutExpired:
                r.update(status='timeout', reason=f'exceeded {args.timeout}s including imports/setup')
        results.append(r)
        write_results(Path(args.output), meta, results)
        timing = r.get('timings', {}).get('forward', {}).get('median_s')
        print(f"{c['method']:12} N={c['n']:6} {r['status']:11} "
              f"{f'{timing * 1e3:.4f} ms' if timing else r.get('reason', '')} {c['settings']}", flush=True)
    if any(r['status'] == 'error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
