function grid = gaussogram_to_grid(coeffs, n, varargin)
%GAUSSOGRAM_TO_GRID  Render packed GFT coefficients as a time-frequency image.
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n)
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n, 'scheme', s, 'window', w, ...
%                             'nyquist_flat_top', tf, 'interp', i, ...
%                             'interp_freq', f, 'smooth', sg, ...
%                             'n_freq', nf, 'n_time', nt)
%
%   Produces an n_freq x n_time magnitude grid (default (N/2+1) x N):
%   frequency on rows (low to high), time on columns. Each band carries
%   'width' complex samples spanning the full record; its magnitude is
%   resampled to n_time time points along time ('interp'), then placed on the
%   frequency axis ('interp_freq').
%
%   Options (in addition to the shared scheme/window/nyquist_flat_top):
%     'interp'      - 'linear' (default) piecewise-linear time resampling, or
%                     'nearest' block/step sampling (raw coefficient cells).
%     'interp_freq' - 'block' (default) flat-fills each band's resampled row
%                     into every frequency row it covers, overlapping bands
%                     combined by max; 'linear' anchors each band at its
%                     centre frequency and linearly interpolates between
%                     adjacent band centres (no horizontal block seams);
%                     'gauss' weights each band's row by its Gaussian window
%                     across frequency and normalises across overlapping bands
%                     (partition of unity) - confines each band to its
%                     effective bandwidth (best for dyadic_dual_real).
%     'normalize'   - 'none' (default) raw |coeff|; 'width' multiplies each
%                     band by its width (L1, equal-amplitude tones equally
%                     bright); 'energy' multiplies by sqrt(width) (L2).
%     'smooth'      - [] (default, off), a scalar Gaussian sigma applied to both
%                     axes, or a [sigma_freq, sigma_time] pair, in grid cells.
%     'n_freq'      - number of rows (default N/2+1), at evenly spaced
%                     frequencies from 0 to N/2 bins inclusive.
%     'n_time'      - number of columns (default N), at times k*N/n_time.
%                     The full grid is (N/2+1)*N doubles (16 GiB at N = 2^16),
%                     so pass a screen-sized shape for long signals. With fewer
%                     rows than bins, a band narrower than the row spacing may
%                     cover no row in 'block' / 'gauss' mode.
%
%   Mirrors the Python gaussogram.to_grid helper (bands_per_octave is not yet
%   exposed through the C ABI, so only the bands_per_octave = 1 layout here).
%
%   See also GAUSSOGRAM, GAUSSOGRAM_SCHEME_BANDS.

p = inputParser;
p.FunctionName = 'gaussogram_to_grid';
addParameter(p, 'scheme', 'dyadic_dual_real', @(s) ischar(s) || isstring(s));
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'nyquist_flat_top', false, @(t) isscalar(t) && (islogical(t) || isnumeric(t)));
addParameter(p, 'interp', 'linear', @(s) ischar(s) || isstring(s));
addParameter(p, 'interp_freq', 'block', @(s) ischar(s) || isstring(s));
addParameter(p, 'normalize', 'none', @(s) ischar(s) || isstring(s));
addParameter(p, 'smooth', [], @(s) isempty(s) || (isnumeric(s) && (isscalar(s) || numel(s) == 2)));
addParameter(p, 'n_freq', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v)));
addParameter(p, 'n_time', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v)));
parse(p, varargin{:});

interp = char(lower(string(p.Results.interp)));
if ~ismember(interp, {'linear', 'nearest'})
    error('gaussogram:interp', "interp must be 'linear' or 'nearest'.");
end

interp_freq = char(lower(string(p.Results.interp_freq)));
if ~ismember(interp_freq, {'linear', 'block', 'gauss'})
    error('gaussogram:interp_freq', "interp_freq must be 'linear', 'block', or 'gauss'.");
end

normalize = char(lower(string(p.Results.normalize)));
if ~ismember(normalize, {'none', 'width', 'energy'})
    error('gaussogram:normalize', "normalize must be 'none', 'width', or 'energy'.");
end

coeffs = double(coeffs(:));
bands = gaussogram_scheme_bands(n, ...
    'scheme', p.Results.scheme, 'window', p.Results.window, ...
    'nyquist_flat_top', p.Results.nyquist_flat_top);

nfreq = n / 2 + 1;
if ~isempty(p.Results.n_freq)
    nfreq = double(p.Results.n_freq);
end
ntime = n;
if ~isempty(p.Results.n_time)
    ntime = double(p.Results.n_time);
end
if nfreq < 2 || ntime < 1 || nfreq ~= round(nfreq) || ntime ~= round(ntime)
    error('gaussogram:shape', 'n_freq must be an integer >= 2 and n_time an integer >= 1.');
end
row_freqs = linspace(0, n / 2, nfreq).';   % row frequencies in bins (column)
target_times = (0:(ntime - 1)) * (n / ntime);
% Rows per working block, bounding temporaries to ~2^20 doubles (8 MB).
block_rows = max(1, floor(2^20 / ntime));

if strcmp(interp_freq, 'block')
    grid = zeros(nfreq, ntime);
    for b = 1:numel(bands.src_lo)
        lo = bands.src_lo(b);          % 0-based
        w = bands.width(b);
        off = bands.out_off(b);        % 0-based
        [r0, r1] = local_band_rows(row_freqs, lo, lo + w);
        if r0 > r1
            continue;
        end
        row = local_band_row(coeffs, off, w, n, interp, normalize, target_times);
        for b0 = r0:block_rows:r1
            b1 = min(b0 + block_rows - 1, r1);
            grid(b0:b1, :) = max(grid(b0:b1, :), row);
        end
    end
elseif strcmp(interp_freq, 'linear')  % anchor each band at its centre, interpolate
    node_freqs = [];
    node_profiles = zeros(0, ntime);
    for b = 1:numel(bands.src_lo)
        w = bands.width(b);
        off = bands.out_off(b);        % 0-based
        fc = bands.fcentre(b);
        if fc > n / 2
            continue;
        end
        row = local_band_row(coeffs, off, w, n, interp, normalize, target_times);
        existing = find(node_freqs == fc, 1);
        if isempty(existing)
            node_freqs(end + 1) = fc;             %#ok<AGROW>
            node_profiles(end + 1, :) = row;      %#ok<AGROW>
        else
            % Two bands sharing a centre: keep the larger magnitude per cell.
            node_profiles(existing, :) = max(node_profiles(existing, :), row);
        end
    end
    [node_freqs, order] = sort(node_freqs);
    node_profiles = node_profiles(order, :);
    grid = local_interp_freq_grid(node_freqs, node_profiles, row_freqs, block_rows);
else  % 'gauss': weight each band by its window profile, normalise overlaps
    grid = zeros(nfreq, ntime);
    weight_sum = zeros(nfreq, 1);
    nyquist = logical(p.Results.nyquist_flat_top);
    for b = 1:numel(bands.src_lo)
        lo = bands.src_lo(b);          % 0-based
        w = bands.width(b);
        off = bands.out_off(b);        % 0-based
        fc = bands.fcentre(b);
        [r0, r1] = local_band_rows(row_freqs, lo, lo + w);
        if r0 > r1
            continue;
        end
        row = local_band_row(coeffs, off, w, n, interp, normalize, target_times);
        f = row_freqs(r0:r1);          % frequencies of the covered rows (column)
        if w == 1 || fc == 0
            % DC / width-1 bands carry no taper: flat over their support.
            wj = ones(numel(f), 1);
        else
            sigma = fc / (2.0 * pi);   % freq-domain sigma of the band Gaussian
            wj = exp(-((f - fc).^2) / (2.0 * sigma^2));
            % Mirror the forward transform's Nyquist flat-top on the top band.
            if nyquist && lo + w >= n / 2 + 1
                wj(f >= fc) = 1.0;
            end
        end
        for b0 = r0:block_rows:r1
            b1 = min(b0 + block_rows - 1, r1);
            grid(b0:b1, :) = grid(b0:b1, :) + wj(b0 - r0 + 1 : b1 - r0 + 1) .* row;
        end
        weight_sum(r0:r1) = weight_sum(r0:r1) + wj;
    end
    weight_sum(weight_sum <= 0) = 1.0;
    grid = grid ./ weight_sum;
end

if ~isempty(p.Results.smooth)
    grid = local_gaussian_blur(grid, p.Results.smooth);
end
end

function [r0, r1] = local_band_rows(row_freqs, lo, hi)
%LOCAL_BAND_ROWS  1-based rows r0..r1 whose frequency lies in [lo, hi).
r0 = sum(row_freqs < lo) + 1;
r1 = sum(row_freqs < hi);
end

function row = local_band_row(coeffs, off, w, n, interp, normalize, target_times)
%LOCAL_BAND_ROW  Time-resample one band's magnitude to the target times.
mag = abs(coeffs(off + 1 : off + w));   % 1-based slice
% |coeff| ~ 1/width for a fixed-amplitude line; x width (L1) equalises tone
% brightness across bands, x sqrt(width) (L2) is the energy-preserving middle.
if strcmp(normalize, 'width')
    mag = mag * w;
elseif strcmp(normalize, 'energy')
    mag = mag * sqrt(w);
end
if w == 1
    row = repmat(mag(1), 1, numel(target_times));
    return;
end
% The unshifted per-band inverse FFT puts sample j at time j*n/w (matching
% the Python to_grid and coefficient_geometry). The transform is circular,
% so resample periodically: the last sample wraps back to sample 0 at t=n.
if strcmp(interp, 'nearest')
    % Circularly nearest sample, computed directly rather than via a
    % (numel(target_times) x w) distance matrix. Ties go to the lower index,
    % so the wrap-around tie between samples w-1 and 0 goes to 0.
    k = target_times * (w / n);
    idx = ceil(k - 0.5);
    idx(k == w - 0.5) = 0;
    row = reshape(mag(mod(idx, w) + 1), 1, []);
else  % linear, periodic (matches numpy.interp with period=n)
    band_times = (0:(w - 1)) * (n / w);
    row = interp1([band_times, n], [mag(:).', mag(1)], target_times, 'linear');
end
end

function grid = local_interp_freq_grid(node_freqs, node_profiles, row_freqs, block_rows)
%LOCAL_INTERP_FREQ_GRID  Linearly interpolate band-centre rows across frequency.
%   node_profiles is (M x T): one time-row per band, anchored at node_freqs
%   (length M, sorted ascending). Each output row (at frequency row_freqs(r),
%   in bins) interpolates column-by-column between the bracketing band
%   centres. Rows outside the centre range clamp to the nearest band. Rows
%   are filled in blocks to bound temporaries.
node_freqs = node_freqs(:);      % force M x 1 column
m = numel(node_freqs);
x = row_freqs(:);                % nrows x 1
nrows = numel(x);
if m == 1
    grid = repmat(node_profiles(1, :), nrows, 1);
    return;
end
% Bracketing lower-node index for each row (clamped so endpoints stay flat):
% the number of nodes at or below the row frequency.
idx = sum(x >= node_freqs.', 2);
idx = min(max(idx, 1), m - 1);
x0 = node_freqs(idx);            % nrows x 1
x1 = node_freqs(idx + 1);        % nrows x 1
denom = x1 - x0;                 % nrows x 1
wgt = zeros(nrows, 1);
nz = denom > 0;
wgt(nz) = (x(nz) - x0(nz)) ./ denom(nz);
wgt = min(max(wgt, 0.0), 1.0);   % clamp rows below/above the centre range
grid = zeros(nrows, size(node_profiles, 2));
for b0 = 1:block_rows:nrows
    b1 = min(b0 + block_rows - 1, nrows);
    r = b0:b1;
    grid(r, :) = (1.0 - wgt(r)) .* node_profiles(idx(r), :) ...
        + wgt(r) .* node_profiles(idx(r) + 1, :);   % implicit expansion
end
end

function grid = local_gaussian_blur(grid, sigma)
%LOCAL_GAUSSIAN_BLUR  Separable edge-padded Gaussian blur (sigma in cells).
if isscalar(sigma)
    sf = double(sigma);
    st = double(sigma);
else
    sf = double(sigma(1));
    st = double(sigma(2));
end
if sf > 0
    grid = blur_axis(grid, sf, 1);
end
if st > 0
    grid = blur_axis(grid, st, 2);
end
end

function a = blur_axis(a, sigma, axis)
%BLUR_AXIS  Edge-padded 1-D Gaussian convolution along one axis, via conv2 on
%   blocks of the other axis (each block is padded, convolved, written back).
radius = max(1, round(3.0 * sigma));
x = (-radius:radius);
kernel = exp(-(x.^2) / (2.0 * sigma^2));
kernel = kernel / sum(kernel);

[nr, nc] = size(a);
if axis == 1
    step = max(1, floor(2^20 / (nr + 2 * radius)));
    for c0 = 1:step:nc
        c = c0:min(c0 + step - 1, nc);
        blk = a(:, c);
        padded = [repmat(blk(1, :), radius, 1); blk; repmat(blk(end, :), radius, 1)];
        a(:, c) = conv2(padded, kernel(:), 'valid');
    end
else
    step = max(1, floor(2^20 / (nc + 2 * radius)));
    for r0 = 1:step:nr
        r = r0:min(r0 + step - 1, nr);
        blk = a(r, :);
        padded = [repmat(blk(:, 1), 1, radius), blk, repmat(blk(:, end), 1, radius)];
        a(r, :) = conv2(padded, kernel, 'valid');
    end
end
end
