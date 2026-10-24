function coeffs = gaussogram(signal, varargin)
%GAUSSOGRAM  Forward fast Gaussian S-transform (GFT) of a signal.
%   coeffs = GAUSSOGRAM(signal)
%   coeffs = GAUSSOGRAM(signal, 'scheme', s, 'window', w, 'nyquist_flat_top', tf, ...
%                       'bands_per_octave', b)
%
%   Inputs:
%     signal - double vector of length N (a power of two), OR an N x count
%              matrix whose columns are independent segments (real segments
%              are transformed in parallel via the Rust core's batch path).
%              Real for the real schemes; real or complex for 'dyadic_complex'.
%   Name-value options:
%     'scheme'           - 'dyadic_dual_real' (default, N-1 coeffs,
%                          least-squares frame inverse),
%                          'dyadic_real' (exactly invertible, N/2+1 coeffs), or
%                          'dyadic_complex' (complex input, N coeffs; the port
%                          of the legacy complex transform, Python gft1d).
%     'window'           - 'gaussian' (default) or 'box'.
%     'nyquist_flat_top' - logical, flatten the top band window (default
%                          false; dyadic_dual_real only).
%     'bands_per_octave' - power of two >= 1 (default 1): split each dyadic
%                          octave into that many sub-bands, trading time
%                          resolution for frequency resolution. The output
%                          length does not change. Real schemes only.
%
%   Output:
%     coeffs - complex array, (output_len x count). For a single segment this is
%              an (output_len x 1) column vector. Use GAUSSOGRAM_TO_GRID to
%              render it as a time-frequency image, and GAUSSOGRAM_SCHEME_BANDS
%              to inspect the band layout.
%
%   See also GAUSSOGRAM_TO_GRID, GAUSSOGRAM_SCHEME_BANDS, GAUSSOGRAM_OUTPUT_LEN,
%   GAUSSOGRAM_INVERSE_REAL, GAUSSOGRAM_COEFFICIENT_GEOMETRY, GAUSSOGRAM1D,
%   BUILD_MEX.

opts = gaussogram_parse_opts(varargin{:});
[scheme_id, window_id] = gaussogram_ids(opts.scheme, opts.window);

if ~isa(signal, 'double') || issparse(signal)
    error('gaussogram:signal', 'signal must be a full double array.');
end
if scheme_id == 2
    signal = complex(signal);   % the complex scheme takes complex input
elseif ~isreal(signal)
    error('gaussogram:signal', ...
        "signal must be real for the real schemes; use 'scheme', 'dyadic_complex' for complex input.");
end

gaussogram_check_built();
coeffs = gaussogram_mex('forward', signal, scheme_id, window_id, ...
    double(opts.nyquist_flat_top), opts.bands_per_octave);
end
