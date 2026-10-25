function bpo = gaussogram_check_bpo(bpo)
%GAUSSOGRAM_CHECK_BPO  Validate a bands_per_octave value (a power of two >= 1).
%   Returns it as a double scalar; errors with gaussogram:bands_per_octave.
if ~(isnumeric(bpo) && isscalar(bpo) && isreal(bpo) && isfinite(bpo) ...
        && bpo >= 1 && bpo == floor(bpo))
    error('gaussogram:bands_per_octave', ...
        'bands_per_octave must be a power of two >= 1.');
end
bpo = double(bpo);
[f, ~] = log2(bpo);   % mantissa is exactly 0.5 for a power of two
if f ~= 0.5
    error('gaussogram:bands_per_octave', ...
        'bands_per_octave must be a power of two >= 1 (got %d).', bpo);
end
end
