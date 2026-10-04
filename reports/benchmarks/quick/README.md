# Transform runtime benchmark

Run: 2026-10-04T07:16:29Z on macOS-26.6.2-arm64-arm-64bit-Mach-O; arm64.

Warm forward calls include Python overhead and output allocation. Setup and one-shot
timings are separate. Shading shows the interquartile range of batch-average timings.
Methods differ in resolution, boundary handling, precision and output size;
these are implementation comparisons, not equal-accuracy comparisons.

Full configuration, versions, warnings, thread settings, and raw samples: [results.json](results.json).
Machine-readable phase timings: [results.csv](results.csv).

![comparison.png](comparison.png)

![gaussogram.png](gaussogram.png)

| Method and settings | N | Forward ms | Setup ms | One-shot ms | Coefficients | Output MiB |
|---|---:|---:|---:|---:|---:|---:|
| gaussogram dyadic_real bpo=1 gaussian flat=False | 256 | 0.0014 | 0.0405 | 0.0411 | 129 | 0.002 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 256 | 0.0015 | 0.0865 | 0.0891 | 129 | 0.002 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 256 | 0.0019 | 0.0658 | 0.0695 | 255 | 0.004 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 256 | 0.0020 | 0.1764 | 0.1790 | 255 | 0.004 |
| dwt wavelet=haar | 256 | 0.0237 | — | — | 256 | 0.002 |
| dwt wavelet=sym8 | 256 | 0.0185 | — | — | 256 | 0.002 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 256 | 4.6472 | — | — | 18688 | 0.285 |
| stockwell  | 256 | 0.2293 | — | — | 33024 | 0.504 |
| stft window=64, overlap=0.5 | 256 | 0.0487 | — | — | 231 | 0.004 |
| gaussogram dyadic_real bpo=1 gaussian flat=False | 1024 | 0.0080 | 0.1780 | 0.1862 | 513 | 0.008 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 1024 | 0.0047 | 0.4543 | 0.4567 | 513 | 0.008 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 1024 | 0.0105 | 0.3054 | 0.3246 | 1023 | 0.016 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 1024 | 0.0069 | 0.9275 | 0.9456 | 1023 | 0.016 |
| dwt wavelet=haar | 1024 | 0.0294 | — | — | 1024 | 0.008 |
| dwt wavelet=sym8 | 1024 | 0.0360 | — | — | 1024 | 0.008 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 1024 | 16.2891 | — | — | 99328 | 1.516 |
| stockwell  | 1024 | 4.2495 | — | — | 525312 | 8.016 |
| stft window=64, overlap=0.5 | 1024 | 0.0529 | — | — | 1023 | 0.016 |
| gaussogram dyadic_real bpo=1 gaussian flat=False | 4096 | 0.0383 | 0.8125 | 0.8354 | 2049 | 0.031 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 4096 | 0.0229 | 2.3564 | 2.3689 | 2049 | 0.031 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 4096 | 0.0472 | 1.5005 | 1.5588 | 4095 | 0.062 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 4096 | 0.0314 | 4.7926 | 4.8086 | 4095 | 0.062 |
| dwt wavelet=haar | 4096 | 0.0444 | — | — | 4096 | 0.031 |
| dwt wavelet=sym8 | 4096 | 0.0883 | — | — | 4096 | 0.031 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 4096 | 72.0979 | — | — | 495616 | 7.562 |
| stockwell  | 4096 | 73.9793 | — | — | 8392704 | 128.062 |
| stft window=64, overlap=0.5 | 4096 | 0.0704 | — | — | 4191 | 0.064 |
