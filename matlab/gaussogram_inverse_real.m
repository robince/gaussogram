function signal = gaussogram_inverse_real(coeffs, varargin)
%GAUSSOGRAM_INVERSE_REAL  Inverse of the 'dyadic_real' scheme.
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs)
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs, 'window', w, 'bands_per_octave', b)
%
%   coeffs is the length-(N/2+1) complex coefficient vector produced by
%   GAUSSOGRAM(..., 'scheme', 'dyadic_real'). Returns the reconstructed real
%   signal as a length-N column vector. The scheme is implicitly 'dyadic_real'
%   and only the 'window' and 'bands_per_octave' options are honoured; both
%   must match the forward transform (a bands_per_octave mismatch cannot be
%   detected, since the packed length is the same for every value, and
%   silently reconstructs the wrong signal). (The core also inverts
%   'dyadic_dual_real'; that path is not wired into MATLAB yet.)
%
%   See also GAUSSOGRAM.

p = inputParser;
p.FunctionName = 'gaussogram_inverse_real';
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'bands_per_octave', 1);
parse(p, varargin{:});
[~, window_id] = gaussogram_ids('dyadic_real', p.Results.window);
bpo = gaussogram_check_bpo(p.Results.bands_per_octave);

if ~isa(coeffs, 'double') || issparse(coeffs)
    error('gaussogram:coeffs', 'coeffs must be a full double array.');
end

gaussogram_check_built();
signal = gaussogram_mex('inverse_real', coeffs, window_id, bpo);
end
