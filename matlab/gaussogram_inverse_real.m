function signal = gaussogram_inverse_real(coeffs, varargin)
%GAUSSOGRAM_INVERSE_REAL  Inverse of the invertible 'dyadic_real' scheme.
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs)
%   signal = GAUSSOGRAM_INVERSE_REAL(coeffs, 'window', w)
%
%   coeffs is the length-(N/2+1) complex coefficient vector produced by
%   GAUSSOGRAM(..., 'scheme', 'dyadic_real'). Returns the reconstructed real
%   signal as a length-N column vector. The scheme is implicitly 'dyadic_real'
%   (the only invertible real scheme); only the 'window' option is honoured.
%
%   See also GAUSSOGRAM.

p = inputParser;
p.FunctionName = 'gaussogram_inverse_real';
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
parse(p, varargin{:});
[~, window_id] = gaussogram_ids('dyadic_real', p.Results.window);

if ~isa(coeffs, 'double') || issparse(coeffs)
    error('gaussogram:coeffs', 'coeffs must be a full double array.');
end

gaussogram_check_built();
signal = gaussogram_mex('inverse_real', coeffs, window_id);
end
