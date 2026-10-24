# Transform runtime benchmark

Run: 2026-10-24T19:28:52Z on macOS-26.6.2-arm64-arm-64bit-Mach-O; arm64.

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
| gaussogram dyadic_real bpo=1 gaussian flat=False | 256 | 0.0009 | 0.0142 | 0.0016 | 129 | 0.002 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 256 | 0.0009 | 0.0183 | 0.0015 | 129 | 0.002 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 256 | 0.0013 | 0.0183 | 0.0019 | 255 | 0.004 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 256 | 0.0011 | 0.0306 | 0.0017 | 255 | 0.004 |
| dwt wavelet=haar | 256 | 0.0183 | — | — | 256 | 0.002 |
| dwt wavelet=sym8 | 256 | 0.0139 | — | — | 256 | 0.002 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 256 | 3.6176 | — | — | 18688 | 0.285 |
| stockwell  | 256 | 0.1813 | — | — | 33024 | 0.504 |
| stft window=64, overlap=0.5 | 256 | 0.0347 | — | — | 231 | 0.004 |
| gaussogram dyadic_real bpo=1 gaussian flat=False | 1024 | 0.0035 | 0.0470 | 0.0043 | 513 | 0.008 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 1024 | 0.0033 | 0.0599 | 0.0041 | 513 | 0.008 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 1024 | 0.0050 | 0.0647 | 0.0058 | 1023 | 0.016 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 1024 | 0.0044 | 0.1108 | 0.0052 | 1023 | 0.016 |
| dwt wavelet=haar | 1024 | 0.0236 | — | — | 1024 | 0.008 |
| dwt wavelet=sym8 | 1024 | 0.0269 | — | — | 1024 | 0.008 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 1024 | 12.7291 | — | — | 99328 | 1.516 |
| stockwell  | 1024 | 3.2272 | — | — | 525312 | 8.016 |
| stft window=64, overlap=0.5 | 1024 | 0.0382 | — | — | 1023 | 0.016 |
| gaussogram dyadic_real bpo=1 gaussian flat=False | 4096 | 0.0165 | 0.1734 | 0.0175 | 2049 | 0.031 |
| gaussogram dyadic_real bpo=4 gaussian flat=False | 4096 | 0.0149 | 0.2284 | 0.0162 | 2049 | 0.031 |
| gaussogram dyadic_dual_real bpo=1 gaussian flat=False | 4096 | 0.0227 | 0.2593 | 0.0243 | 4095 | 0.062 |
| gaussogram dyadic_dual_real bpo=4 gaussian flat=False | 4096 | 0.0207 | 0.4491 | 0.0221 | 4095 | 0.062 |
| dwt wavelet=haar | 4096 | 0.0345 | — | — | 4096 | 0.031 |
| dwt wavelet=sym8 | 4096 | 0.0686 | — | — | 4096 | 0.031 |
| cwt wavelet=cmor1.5-1.0, voices=12 | 4096 | 56.4241 | — | — | 495616 | 7.562 |
| stockwell  | 4096 | 58.4507 | — | — | 8392704 | 128.062 |
| stft window=64, overlap=0.5 | 4096 | 0.0545 | — | — | 4191 | 0.064 |
