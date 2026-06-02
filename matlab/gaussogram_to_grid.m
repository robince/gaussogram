function grid = gaussogram_to_grid(coeffs, n, varargin)
%GAUSSOGRAM_TO_GRID  Render packed GFT coefficients as a time-frequency image.
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n)
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n, 'scheme', s, 'window', w, ...
%                             'nyquist_flat_top', tf, 'interp', i, ...
%                             'interp_freq', f, 'smooth', sg)
%
%   Produces an (N/2+1) x N magnitude grid: frequency on rows (low to high),
%   time on columns. Each band carries 'width' complex samples spanning the
%   full record; its magnitude is resampled to N time points along time
%   ('interp'), then placed on the frequency axis ('interp_freq').
%
%   Options (in addition to the shared scheme/window/nyquist_flat_top):
%     'interp'      - 'linear' (default) piecewise-linear time resampling, or
%                     'nearest' block/step sampling (raw coefficient cells).
%     'interp_freq' - 'block' (default) flat-fills each band's resampled row
%                     into every frequency row it covers, overlapping bands
%                     combined by max; or 'linear' anchors each band at its
%                     centre frequency and linearly interpolates between
%                     adjacent band centres (no horizontal block seams).
%     'smooth'      - [] (default, off), a scalar Gaussian sigma applied to both
%                     axes, or a [sigma_freq, sigma_time] pair, in grid cells.
%
%   Mirrors the Python gaussogram.to_grid helper.
%
%   See also GAUSSOGRAM, GAUSSOGRAM_SCHEME_BANDS.

p = inputParser;
p.FunctionName = 'gaussogram_to_grid';
addParameter(p, 'scheme', 'dyadic_dual_real', @(s) ischar(s) || isstring(s));
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'nyquist_flat_top', false, @(t) isscalar(t) && (islogical(t) || isnumeric(t)));
addParameter(p, 'interp', 'linear', @(s) ischar(s) || isstring(s));
addParameter(p, 'interp_freq', 'block', @(s) ischar(s) || isstring(s));
addParameter(p, 'smooth', [], @(s) isempty(s) || (isnumeric(s) && (isscalar(s) || numel(s) == 2)));
parse(p, varargin{:});

interp = char(lower(string(p.Results.interp)));
if ~ismember(interp, {'linear', 'nearest'})
    error('gaussogram:interp', "interp must be 'linear' or 'nearest'.");
end

interp_freq = char(lower(string(p.Results.interp_freq)));
if ~ismember(interp_freq, {'linear', 'block'})
    error('gaussogram:interp_freq', "interp_freq must be 'linear' or 'block'.");
end

coeffs = double(coeffs(:));
bands = gaussogram_scheme_bands(n, ...
    'scheme', p.Results.scheme, 'window', p.Results.window, ...
    'nyquist_flat_top', p.Results.nyquist_flat_top);

nfreq = n / 2 + 1;
target_times = 0:(n - 1);

if strcmp(interp_freq, 'block')
    grid = zeros(nfreq, n);
    for b = 1:numel(bands.src_lo)
        lo = bands.src_lo(b);          % 0-based
        w = bands.width(b);
        off = bands.out_off(b);        % 0-based
        if lo >= nfreq
            continue;
        end
        row = local_band_row(coeffs, off, w, n, interp, target_times);
        hi = min(lo + w, nfreq);       % exclusive upper, 0-based -> rows lo+1..hi
        rows = (lo + 1):hi;
        grid(rows, :) = max(grid(rows, :), row);
    end
else  % linear in frequency: anchor each band at its centre, interpolate
    node_freqs = [];
    node_profiles = zeros(0, n);
    for b = 1:numel(bands.src_lo)
        w = bands.width(b);
        off = bands.out_off(b);        % 0-based
        fc = bands.fcentre(b);
        if fc >= nfreq
            continue;
        end
        row = local_band_row(coeffs, off, w, n, interp, target_times);
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
    grid = local_interp_freq_grid(node_freqs, node_profiles, nfreq);
end

if ~isempty(p.Results.smooth)
    grid = local_gaussian_blur(grid, p.Results.smooth);
end
end

function row = local_band_row(coeffs, off, w, n, interp, target_times)
%LOCAL_BAND_ROW  Time-resample one band's magnitude to all N columns.
mag = abs(coeffs(off + 1 : off + w));   % 1-based slice
if w == 1
    row = repmat(mag(1), 1, n);
    return;
end
band_times = linspace(0, n - 1, w);
if strcmp(interp, 'nearest')
    [~, idx] = min(abs(target_times.' - band_times), [], 2);
    row = mag(idx).';
else  % linear
    row = interp1(band_times, mag, target_times, 'linear');
end
end

function grid = local_interp_freq_grid(node_freqs, node_profiles, nfreq)
%LOCAL_INTERP_FREQ_GRID  Linearly interpolate band-centre rows across frequency.
%   node_profiles is (M x N): one time-row per band, anchored at node_freqs
%   (length M, sorted ascending). For each output row r in [0, nfreq) we
%   interpolate column-by-column between the bracketing band centres. Rows
%   outside the centre range clamp to the nearest band (interp1 'extrap' off).
node_freqs = node_freqs(:);      % force M x 1 column
m = numel(node_freqs);
x = (0:(nfreq - 1)).';           % nfreq x 1
if m == 1
    grid = repmat(node_profiles(1, :), nfreq, 1);
    return;
end
% Bracketing lower-node index for each row (clamped so endpoints stay flat).
idx = zeros(nfreq, 1);
for r = 1:nfreq
    k = find(node_freqs <= x(r), 1, 'last');
    if isempty(k)
        k = 1;
    elseif k >= m
        k = m - 1;
    end
    idx(r) = k;
end
x0 = node_freqs(idx);            % nfreq x 1
x1 = node_freqs(idx + 1);        % nfreq x 1
denom = x1 - x0;                 % nfreq x 1
wgt = zeros(nfreq, 1);
nz = denom > 0;
wgt(nz) = (x(nz) - x0(nz)) ./ denom(nz);
wgt = min(max(wgt, 0.0), 1.0);   % clamp rows below/above the centre range
lo_prof = node_profiles(idx, :);     % nfreq x N
hi_prof = node_profiles(idx + 1, :); % nfreq x N
grid = (1.0 - wgt) .* lo_prof + wgt .* hi_prof;   % implicit expansion over cols
end

function out = local_gaussian_blur(grid, sigma)
%LOCAL_GAUSSIAN_BLUR  Separable edge-padded Gaussian blur (sigma in cells).
if isscalar(sigma)
    sf = double(sigma);
    st = double(sigma);
else
    sf = double(sigma(1));
    st = double(sigma(2));
end
out = grid;
if sf > 0
    out = blur_axis(out, sf, 1);
end
if st > 0
    out = blur_axis(out, st, 2);
end
end

function out = blur_axis(a, sigma, axis)
%BLUR_AXIS  Edge-padded 1-D Gaussian convolution along one axis.
radius = max(1, round(3.0 * sigma));
x = (-radius:radius);
kernel = exp(-(x.^2) / (2.0 * sigma^2));
kernel = kernel / sum(kernel);

if axis == 2
    a = a.';   % operate along columns, transpose back at the end
end

[nr, nc] = size(a);
out = zeros(nr, nc);
for c = 1:nc
    col = a(:, c);
    padded = [repmat(col(1), radius, 1); col; repmat(col(end), radius, 1)];
    conv_full = conv(padded, kernel(:), 'valid');
    out(:, c) = conv_full;
end

if axis == 2
    out = out.';
end
end
