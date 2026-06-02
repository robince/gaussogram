function coeffs = gaussogram(signal, varargin)
%GAUSSOGRAM  Forward fast Gaussian S-transform (GFT) of a real signal.
%   coeffs = GAUSSOGRAM(signal)
%   coeffs = GAUSSOGRAM(signal, 'scheme', s, 'window', w, 'nyquist_flat_top', tf)
%
%   Inputs:
%     signal - real double vector of length N (a power of two), OR an N x count
%              matrix whose columns are independent segments (transformed in
%              parallel via the Rust core's batch path).
%   Name-value options:
%     'scheme'           - 'dyadic_dual_real' (default, N-1 coeffs),
%                          'dyadic_real' (invertible, N/2+1 coeffs).
%     'window'           - 'gaussian' (default) or 'box'.
%     'nyquist_flat_top' - logical, flatten the top band window (default false).
%
%   Output:
%     coeffs - complex array, (output_len x count). For a single segment this is
%              an (output_len x 1) column vector. Use GAUSSOGRAM_TO_GRID to
%              render it as a time-frequency image, and GAUSSOGRAM_SCHEME_BANDS
%              to inspect the band layout.
%
%   See also GAUSSOGRAM_TO_GRID, GAUSSOGRAM_SCHEME_BANDS, GAUSSOGRAM_OUTPUT_LEN,
%   GAUSSOGRAM_INVERSE_REAL, BUILD_MEX.

opts = gaussogram_parse_opts(varargin{:});
[scheme_id, window_id] = gaussogram_ids(opts.scheme, opts.window);

if ~isa(signal, 'double') || ~isreal(signal) || issparse(signal)
    error('gaussogram:signal', 'signal must be a real full double array.');
end

gaussogram_check_built();
coeffs = gaussogram_mex('forward', signal, scheme_id, window_id, double(opts.nyquist_flat_top));
end
