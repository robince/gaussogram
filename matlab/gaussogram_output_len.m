function olen = gaussogram_output_len(n, varargin)
%GAUSSOGRAM_OUTPUT_LEN  Packed coefficient count for a scheme at length N.
%   olen = GAUSSOGRAM_OUTPUT_LEN(n)
%   olen = GAUSSOGRAM_OUTPUT_LEN(n, 'scheme', s, 'window', w, 'nyquist_flat_top', tf)
%
%   Returns the number of complex coefficients a forward transform of a
%   length-N signal produces under the given scheme. For 'dyadic_dual_real'
%   this is N-1; for 'dyadic_real' it is N/2+1.
%
%   See also GAUSSOGRAM, GAUSSOGRAM_SCHEME_BANDS.

opts = gaussogram_parse_opts(varargin{:});
[scheme_id, window_id] = gaussogram_ids(opts.scheme, opts.window);

gaussogram_check_built();
olen = gaussogram_mex('output_len', double(n), scheme_id, window_id, ...
    double(opts.nyquist_flat_top));
end
