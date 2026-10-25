function signal = gaussogram_inverse_real(coeffs, varargin)
%GAUSSOGRAM_INVERSE_REAL  Inverse of a real-input scheme.
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs)
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs, 'scheme', s, 'window', w, ...
%                                    'bands_per_octave', b, 'top_window', t)
%
%   coeffs is a complex coefficient vector produced by GAUSSOGRAM. Returns
%   the reconstructed real signal as a length-N column vector.
%     'dyadic_real'      (length N/2+1): exact per-band inverse.
%     'dyadic_dual_real' (length N-1): least-squares (canonical dual) frame
%                        inverse. Exact for unmodified coefficients; for
%                        modified ones (thresholded, masked, denoised) it
%                        returns the signal whose coefficients are closest to
%                        them, using both tilings.
%   'scheme' [] (default) infers the scheme from the length: N/2+1 is always
%   2^k+1 and N-1 is always 2^k-1, which never coincide. 'dyadic_complex' is
%   not invertible.
%
%   'window', 'bands_per_octave' and 'top_window' (dyadic_dual_real only;
%   'nyquist_flat_top' is a deprecated alias) must match the forward
%   transform. A mismatch cannot be detected - the packed length is the same
%   for every bands_per_octave and top_window - and silently reconstructs the
%   wrong signal. Mirrors the Python gaussogram.inverse_real.
%
%   See also GAUSSOGRAM, GAUSSOGRAM1D.

p = inputParser;
p.FunctionName = 'gaussogram_inverse_real';
addParameter(p, 'scheme', [], @(s) isempty(s) || ischar(s) || isstring(s));
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'bands_per_octave', 1);
addParameter(p, 'top_window', '', @(s) ischar(s) || isstring(s));
addParameter(p, 'nyquist_flat_top', [], @(t) isempty(t) || (isscalar(t) && (islogical(t) || isnumeric(t))));
parse(p, varargin{:});
bpo = gaussogram_check_bpo(p.Results.bands_per_octave);
[~, top_window_id] = gaussogram_resolve_top_window( ...
    p.Results.top_window, p.Results.nyquist_flat_top);

if ~isa(coeffs, 'double') || issparse(coeffs)
    error('gaussogram:coeffs', 'coeffs must be a full double array.');
end
if ~(ismatrix(coeffs) && isvector(coeffs))
    error('gaussogram:coeffsShape', 'coeffs must be a vector.');
end

len = numel(coeffs);
scheme = p.Results.scheme;
if isempty(scheme)
    if len >= 3 && local_is_pow2(len - 1)
        scheme = 'dyadic_real';
    elseif len >= 7 && local_is_pow2(len + 1)
        scheme = 'dyadic_dual_real';
    else
        error('gaussogram:coeffsLen', ...
            ['coeffs length %d matches neither dyadic_real (N/2+1) nor ' ...
             'dyadic_dual_real (N-1) for a power-of-two N.'], len);
    end
end
[scheme_id, window_id] = gaussogram_ids(scheme, p.Results.window);
if scheme_id == 2
    error('gaussogram:notInvertible', "'dyadic_complex' is not invertible.");
end
if top_window_id ~= 0 && scheme_id ~= 0
    error('gaussogram:top_window', ...
        "top_window is only supported by 'dyadic_dual_real'.");
end

gaussogram_check_built();
signal = gaussogram_mex('inverse_real', coeffs, window_id, bpo, scheme_id, top_window_id);
end

function tf = local_is_pow2(v)
[f, ~] = log2(v);   % mantissa is exactly 0.5 for a power of two
tf = v >= 1 && f == 0.5;
end
