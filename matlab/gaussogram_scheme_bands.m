function bands = gaussogram_scheme_bands(n, varargin)
%GAUSSOGRAM_SCHEME_BANDS  Per-band layout of a scheme at length N.
%   bands = GAUSSOGRAM_SCHEME_BANDS(n)
%   bands = GAUSSOGRAM_SCHEME_BANDS(n, 'scheme', s, 'window', w, 'nyquist_flat_top', tf)
%
%   Returns a struct with parallel column vectors describing each band:
%     bands.src_lo   - low frequency-bin index the band starts at (0-based).
%     bands.width    - number of complex samples the band carries.
%     bands.fcentre  - centre frequency bin of the band.
%     bands.out_off  - offset of the band's first coefficient in the packed
%                      output vector (0-based).
%     bands.src_hi   - src_lo + width (convenience).
%
%   These mirror the Python BandLayout fields and feed GAUSSOGRAM_TO_GRID.
%
%   See also GAUSSOGRAM, GAUSSOGRAM_TO_GRID, GAUSSOGRAM_OUTPUT_LEN.

opts = gaussogram_parse_opts(varargin{:});
[scheme_id, window_id] = gaussogram_ids(opts.scheme, opts.window);

gaussogram_check_built();
m = gaussogram_mex('scheme_bands', double(n), scheme_id, window_id, ...
    double(opts.nyquist_flat_top));   % nb x 4 [lo, width, fcentre, out_off]

bands.src_lo  = m(:, 1);
bands.width   = m(:, 2);
bands.fcentre = m(:, 3);
bands.out_off = m(:, 4);
bands.src_hi  = bands.src_lo + bands.width;
end
