function grid = gaussogram_to_grid(coeffs, n, varargin)
%GAUSSOGRAM_TO_GRID  Render packed GFT coefficients as a time-frequency image.
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n)
%   grid = GAUSSOGRAM_TO_GRID(coeffs, n, 'scheme', s, 'window', w, ...
%                             'nyquist_flat_top', tf, 'interp', i, 'smooth', sg)
%
%   Produces an (N/2+1) x N magnitude grid: frequency on rows (low to high),
%   time on columns. Each band carries 'width' complex samples spanning the
%   full record; its magnitude is resampled to N time points and written to
%   every frequency row the band covers (block fill in frequency). Overlapping
%   bands combine by taking the larger magnitude per cell.
%
%   Options (in addition to the shared scheme/window/nyquist_flat_top):
%     'interp' - 'linear' (default) piecewise-linear time resampling, or
%                'nearest' block/step sampling (raw coefficient cells).
%     'smooth' - [] (default, off), a scalar Gaussian sigma applied to both
%                axes, or a [sigma_freq, sigma_time] pair, in grid cells.
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
addParameter(p, 'smooth', [], @(s) isempty(s) || (isnumeric(s) && (isscalar(s) || numel(s) == 2)));
parse(p, varargin{:});

interp = char(lower(string(p.Results.interp)));
if ~ismember(interp, {'linear', 'nearest'})
    error('gaussogram:interp', "interp must be 'linear' or 'nearest'.");
end

coeffs = double(coeffs(:));
bands = gaussogram_scheme_bands(n, ...
    'scheme', p.Results.scheme, 'window', p.Results.window, ...
    'nyquist_flat_top', p.Results.nyquist_flat_top);

nfreq = n / 2 + 1;
grid = zeros(nfreq, n);
target_times = 0:(n - 1);

for b = 1:numel(bands.src_lo)
    lo = bands.src_lo(b);          % 0-based
    w = bands.width(b);
    off = bands.out_off(b);        % 0-based
    mag = abs(coeffs(off + 1 : off + w));   % 1-based slice

    if w == 1
        row = repmat(mag(1), 1, n);
    else
        band_times = linspace(0, n - 1, w);
        if strcmp(interp, 'nearest')
            [~, idx] = min(abs(target_times.' - band_times), [], 2);
            row = mag(idx).';
        else  % linear
            row = interp1(band_times, mag, target_times, 'linear');
        end
    end

    if lo >= nfreq
        continue;
    end
    hi = min(lo + w, nfreq);       % exclusive upper, 0-based -> rows lo+1..hi
    rows = (lo + 1):hi;
    grid(rows, :) = max(grid(rows, :), row);
end

if ~isempty(p.Results.smooth)
    grid = local_gaussian_blur(grid, p.Results.smooth);
end
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
