function geom = gaussogram_coefficient_geometry(n, varargin)
%GAUSSOGRAM_COEFFICIENT_GEOMETRY  Time-frequency tile of every packed coefficient.
%   geom = GAUSSOGRAM_COEFFICIENT_GEOMETRY(n)
%   geom = GAUSSOGRAM_COEFFICIENT_GEOMETRY(n, 'scheme', s, 'window', w, ...
%              'nyquist_flat_top', tf, 'bands_per_octave', b)
%
%   Describes where each coefficient lives in the time-frequency plane, so a
%   model can work on the packed ~N vector directly (positional features for a
%   transformer, coordinates for a continuous-kernel convolution) without
%   densifying it to a grid. Signal-independent: compute once and reuse. Port
%   of the Python gaussogram.coefficient_geometry.
%
%   Returns a struct of parallel (output_len x 1) double columns; row k
%   describes coeffs(k). Each coefficient owns a tile of size dt (time) x df
%   (frequency), and dt .* df == n for every tile.
%     geom.time   - tile centre time in samples, j*n/w for sample j (0-based)
%                   of a width-w band; in [0, n).
%     geom.freq   - tile centre frequency in bins (negative for the
%                   negative-frequency bands of 'dyadic_complex').
%     geom.dt     - time extent in samples, n / width.
%     geom.df     - frequency extent in bins, width.
%     geom.logf   - log2(max(abs(freq), 0.5)); DC is -1.
%     geom.level  - octave index, floor(log2(max(abs(freq), 1))).
%     geom.sub    - within-octave sub-band rank, 0-based, low to high
%                   frequency (bands sharing tiling, sign and octave).
%     geom.tiling - 0 = tiling A, 1 = tiling B ('dyadic_dual_real').
%     geom.band   - 1-based index of the coefficient's band into the
%                   GAUSSOGRAM_SCHEME_BANDS arrays (Python's band + 1).
%
%   With 'bands_per_octave' > 1, level and sub together locate a coefficient
%   on the (log-frequency, sub-band) lattice.
%
%   See also GAUSSOGRAM_COEFFICIENT_ADJACENCY, GAUSSOGRAM_SCHEME_BANDS,
%   GAUSSOGRAM_TO_GRID.

opts = gaussogram_parse_opts(varargin{:});
bands = gaussogram_scheme_bands(n, 'scheme', opts.scheme, 'window', opts.window, ...
    'nyquist_flat_top', opts.nyquist_flat_top, 'bands_per_octave', opts.bands_per_octave);
n = double(n);
[b_freq, b_tiling] = gaussogram_band_freq_tiling(bands, n, opts.scheme);
nb = numel(bands.width);

% Per-band octave index, then the within-octave rank: bands that share a
% (tiling, sign, octave) are ranked low -> high by source frequency.
b_octave = local_floor_log2(max(abs(b_freq), 1));
b_sub = zeros(nb, 1);
[~, ~, group] = unique([b_tiling, b_freq < 0, b_octave], 'rows');
for g = 1:max(group)
    members = find(group == g);
    [~, order] = sort(bands.src_lo(members));   % stable, like Python's sorted
    b_sub(members(order)) = 0:(numel(members) - 1);
end

total = max(bands.out_off + bands.width);
geom.time = zeros(total, 1);
geom.freq = zeros(total, 1);
geom.dt = zeros(total, 1);
geom.df = zeros(total, 1);
geom.level = zeros(total, 1);
geom.sub = zeros(total, 1);
geom.tiling = zeros(total, 1);
geom.band = zeros(total, 1);
for b = 1:nb
    w = bands.width(b);
    j = (0:(w - 1)).';
    idx = bands.out_off(b) + j + 1;    % 1-based packed positions
    geom.time(idx) = j * (n / w);
    geom.freq(idx) = b_freq(b);
    geom.dt(idx) = n / w;
    geom.df(idx) = w;
    geom.level(idx) = b_octave(b);
    geom.sub(idx) = b_sub(b);
    geom.tiling(idx) = b_tiling(b);
    geom.band(idx) = b;
end
geom.logf = log2(max(abs(geom.freq), 0.5));
% Same field order as the Python CoeffGeometry.
geom = orderfields(geom, {'time', 'freq', 'dt', 'df', 'logf', 'level', 'sub', ...
    'tiling', 'band'});
end

function e = local_floor_log2(x)
%LOCAL_FLOOR_LOG2  floor(log2(x)) for positive integers, exactly: log2's
%   two-output form returns x = f * 2^e with f in [0.5, 1).
[~, e] = log2(x);
e = e - 1;
end
