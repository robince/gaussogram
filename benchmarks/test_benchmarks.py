"""Correctness and runner checks; no machine-dependent speed assertions."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('benchmark_run', Path(__file__).with_name('run.py'))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def stockwell_reference(x):
    """Direct small-N DFT definition, with Stockwell's analytic-signal convention."""
    n = len(x)
    k = np.arange(n)
    spectrum = np.exp(-2j * np.pi * np.outer(k, k) / n) @ x
    spectrum[1:n // 2] *= 2
    spectrum[n // 2 + 1:] = 0
    offsets = np.fft.fftfreq(n) * n
    inverse = np.exp(2j * np.pi * np.outer(k, k) / n) / n
    result = np.empty((n // 2 + 1, n), complex)
    result[0] = x.mean()
    for f in range(1, n // 2 + 1):
        window = np.exp(-2 * np.pi**2 * offsets**2 / f**2)
        result[f] = inverse @ (spectrum[(k + f) % n] * window)
    return result


@pytest.mark.parametrize('signal', ['noise', 'constant', 'impulse', 'tone'])
def test_stockwell_reference(signal):
    st = pytest.importorskip('stockwell.st')
    n = 32
    x = {'noise': np.random.default_rng(4).normal(size=n),
         'constant': np.ones(n), 'impulse': np.eye(1, n, 8).ravel(),
         'tone': np.cos(2 * np.pi * 5 * np.arange(n) / n)}[signal]
    np.testing.assert_allclose(st.st(x, 0, n // 2, gamma=1, win_type='gauss'),
                               stockwell_reference(x), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('case', list(b.cases([64], ['gaussogram', 'dwt', 'cwt', 'stockwell', 'stft', 'grid'])))
def test_output_and_timing(case):
    pytest.importorskip({'dwt': 'pywt', 'cwt': 'pywt', 'stockwell': 'stockwell',
                         'stft': 'scipy'}.get(case['method'], 'gaussogram'))
    r = b.worker(case, repeats=2, target=.0001)
    assert r['status'] == 'ok'
    assert r['output_bytes'] > 0
    assert len(r['timings']['forward']['samples_s']) == 2
    assert r['timings']['forward']['median_s'] > 0


def test_cli_and_memory_skip(tmp_path):
    subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')),
                    '--methods', 'gaussogram', 'stockwell', '--sizes', '64',
                    '--memory-mib', '.02', '--repeats', '2', '--target-seconds', '.0001',
                    '--output', str(tmp_path)], check=True)
    data = json.loads((tmp_path / 'results.json').read_text())
    assert any(r['status'] == 'ok' for r in data['results'])
    assert data['results'][-1]['status'] == 'skipped'
    assert (tmp_path / 'results.csv').read_text().startswith('method,n,settings')


def test_full_gaussogram_settings():
    import gaussogram as g
    x = np.random.default_rng(7).normal(size=64)
    configurations = list(b.cases([64], ['gaussogram'], full=True))
    assert len(configurations) == 30
    for case in configurations:
        _, forward, one_shot = b.prepare(case, x)
        result = forward()
        assert result.size == g.output_len(64, **case['settings'])
        np.testing.assert_allclose(result, one_shot(), atol=1e-12, rtol=1e-12)


def test_timeout_is_recorded(tmp_path):
    subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')),
                    '--methods', 'stockwell', '--sizes', '64', '--timeout', '.001',
                    '--output', str(tmp_path)], check=True)
    data = json.loads((tmp_path / 'results.json').read_text())
    assert data['results'][0]['status'] == 'timeout'
