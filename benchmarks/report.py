#!/usr/bin/env python3
"""Generate standalone plots and a Markdown table from benchmark JSON."""
import argparse
import json
from pathlib import Path


def label(r):
    s = r['settings']
    if r['method'] in ('gaussogram', 'grid'):
        return f"{r['method']} {s['scheme']} bpo={s['bands_per_octave']} {s['window_type']} flat={s['nyquist_flat_top']}"
    return r['method'] + ' ' + ', '.join(f'{k}={s[k]}' for k in
                                        ('wavelet', 'voices', 'window', 'overlap') if k in s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    args = p.parse_args()
    data = json.loads(args.results.read_text())
    out = args.results.parent
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good = [r for r in data['results'] if r['status'] == 'ok']
    # Split families so the full sweep remains readable.
    groups = {'comparison': [r for r in good if r['method'] not in ('gaussogram', 'grid') or
                             (r['method'] == 'gaussogram' and r['settings']['window_type'] == 'gaussian'
                              and not r['settings']['nyquist_flat_top'] and r['settings']['bands_per_octave'] == 1)],
              'gaussogram': [r for r in good if r['method'] == 'gaussogram'],
              'grid': [r for r in good if r['method'] == 'grid']}
    figures = []
    for group, rows in groups.items():
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = sorted({label(r) for r in rows})
        for name in labels:
            rs = sorted([r for r in rows if label(r) == name], key=lambda r: r['n'])
            ns = [r['n'] for r in rs]
            ts = [r['timings']['forward'] for r in rs]
            line, = axes[0].loglog(ns, [t['median_s'] * 1000 for t in ts], '.-', label=name)
            axes[0].fill_between(ns, [t['p25_s'] * 1000 for t in ts],
                                 [t['p75_s'] * 1000 for t in ts], alpha=.15, color=line.get_color())
            axes[1].loglog(ns, [r['coefficients'] for r in rs], '.-', label=name)
        axes[0].set(ylabel='Median forward time (ms)', xlabel='Signal length N')
        axes[1].set(ylabel='Stored coefficients (real or complex)', xlabel='Signal length N')
        for ax in axes:
            ax.grid(True, which='both', alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=7)
        fig.tight_layout()
        filename = f'{group}.png'
        fig.savefig(out / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(filename)
    meta = data['metadata']
    lines = ['# Transform runtime benchmark', '',
             f"Run: {meta['created_utc']} on {meta['platform']}; {meta['machine']}.", '',
             'Warm forward calls include Python overhead and output allocation. Setup and one-shot',
             'timings are separate. Shading shows the interquartile range of batch-average timings.',
             'Methods differ in resolution, boundary handling, precision and output size;',
             'these are implementation comparisons, not equal-accuracy comparisons.', '',
             'Full configuration, versions, warnings, thread settings, and raw samples: [results.json](results.json).',
             'Machine-readable phase timings: [results.csv](results.csv).', '']
    for name in figures:
        lines.extend([f'![{name}]({name})', ''])
    lines.extend(['| Method and settings | N | Forward ms | Setup ms | One-shot ms | Coefficients | Output MiB |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for r in good:
        def ms(phase):
            t = r['timings'].get(phase)
            return f"{t['median_s'] * 1000:.4f}" if t else '—'
        lines.append(f"| {label(r)} | {r['n']} | {ms('forward')} | {ms('setup')} | {ms('one_shot')} | {r['coefficients']} | {r['output_bytes'] / 2**20:.3f} |")
    skipped = [r for r in data['results'] if r['status'] != 'ok']
    if skipped:
        lines.extend(['', '## Cases not measured', ''])
        for r in skipped:
            lines.append(f"- {label(r)}, N={r['n']}: {r['status']} — {r.get('reason', '')}")
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(out / 'README.md')


if __name__ == '__main__':
    main()
