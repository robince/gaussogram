%% gaussogram MATLAB demo
% Cell-mode (%%) script mirroring scripts/demo_gaussogram.py. Run BUILD_MEX
% first. Each cell builds a test signal, forward-transforms it, and renders the
% time-frequency magnitude grid. Use "Run Section" (Ctrl/Cmd+Enter) to step
% through, or run the whole file.

addpath(fileparts(mfilename('fullpath')));
N = 512;                       % power of two
t = (0:N-1)' / N;              % normalised time in [0, 1)

%% Gated tone (middle 50% of the record)
sig = tone(N, 64, 0.5);
plot_gaussogram(sig, N, 'Gated tone (f=64, middle 50%)');

%% Two gated tones, different bands
sig = tone(N, 24, 0.5) + 0.8 * tone(N, 160, 0.5);
plot_gaussogram(sig, N, 'Two gated tones (f=24 and f=160)');

%% Impulse (broadband in frequency, localised in time)
sig = impulse(N, 0.5, 1.0);
plot_gaussogram(sig, N, 'Impulse at t=0.5');

%% Boxcar (sharp edges -> broadband content at the transitions)
sig = boxcar(N, 0.4, 0.6, 1.0);
plot_gaussogram(sig, N, 'Boxcar [0.4, 0.6]');

%% Linear chirp (frequency sweep)
sig = sin(2 * pi * (8 + 120 * t) .* t);
plot_gaussogram(sig, N, 'Linear chirp');

%% Multi-component: low tone + chirp + impulse
sig = 0.7 * tone(N, 20, 1.0) ...
    + sin(2 * pi * (40 + 80 * t) .* t) ...
    + 0.8 * impulse(N, 0.75, 1.0);
plot_gaussogram(sig, N, 'Tone + chirp + impulse');

%% Interpolation comparison on a single gated tone
sig = tone(N, 96, 0.5);
c = gaussogram(sig);
figure('Name', 'Interpolation comparison');
tiledlayout(1, 4, 'TileSpacing', 'compact');
nexttile; show_grid(gaussogram_to_grid(c, N, 'interp', 'nearest', 'interp_freq', 'block'), N);
title('nearest time / block freq');
nexttile; show_grid(gaussogram_to_grid(c, N, 'interp', 'linear', 'interp_freq', 'block'), N);
title('linear time / block freq');
nexttile; show_grid(gaussogram_to_grid(c, N, 'interp', 'linear', 'interp_freq', 'linear'), N);
title('linear time + linear freq');
nexttile; show_grid(gaussogram_to_grid(c, N, 'interp', 'linear', 'interp_freq', 'linear', 'smooth', [2, 4]), N);
title('linear t+f + Gaussian smooth');

%% -- local helpers --------------------------------------------------------

function g = gate(sig, extent)
% Keep the middle EXTENT fraction of samples, zero elsewhere.
n = numel(sig);
keep = round(extent * n);
start = floor((n - keep) / 2);
mask = false(n, 1);
mask(start + 1 : start + keep) = true;
g = sig;
g(~mask) = 0;
end

function s = tone(n, freq_bin, extent)
% Unit-amplitude sinusoid at the given integer frequency bin, gated to the
% middle EXTENT fraction of the record.
t = (0:n-1)' / n;
s = sin(2 * pi * freq_bin * t);
s = gate(s, extent);
end

function s = impulse(n, centre, amp)
% Single nonzero sample at the given fractional position.
s = zeros(n, 1);
idx = min(n, max(1, round(centre * n) + 1));
s(idx) = amp;
end

function s = boxcar(n, start_frac, stop_frac, amp)
% Rectangular pulse over [start_frac, stop_frac) of the record.
s = zeros(n, 1);
lo = max(1, round(start_frac * n) + 1);
hi = min(n, round(stop_frac * n));
s(lo:hi) = amp;
end

function plot_gaussogram(sig, n, ttl)
c = gaussogram(sig);
g = gaussogram_to_grid(c, n, 'interp', 'linear', 'smooth', [1.5, 3]);
figure('Name', ttl);
tiledlayout(2, 1, 'TileSpacing', 'compact');
ax1 = nexttile;
plot((0:n-1) / n, sig); grid on; xlim([0, 1]);
xlabel('time'); ylabel('amplitude'); title(ttl);
% Dummy invisible colorbar: reserves the same right-hand gutter the grid's
% colorbar takes, so the two time axes line up.
cb = colorbar(ax1); cb.Visible = 'off';
nexttile;
show_grid(g, n);
end

function show_grid(g, n)
imagesc((0:n-1) / n, 0:(n/2), g);
axis xy; colormap(gca, 'turbo'); colorbar;
xlabel('time'); ylabel('frequency bin');
end
