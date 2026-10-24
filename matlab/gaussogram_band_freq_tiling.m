function [b_freq, b_tiling] = gaussogram_band_freq_tiling(bands, n, scheme)
%GAUSSOGRAM_BAND_FREQ_TILING  Per-band signed centre frequency and tiling id.
%   Shared by GAUSSOGRAM_COEFFICIENT_GEOMETRY and
%   GAUSSOGRAM_COEFFICIENT_ADJACENCY. Real schemes pack tiling A first
%   (N/2+1 coefficients), then tiling B. 'dyadic_complex' is a single tiling
%   over the full spectrum: its bands from bin N/2 up are negative
%   frequencies, whose fcentre is stored as a magnitude.
if strcmpi(scheme, 'dyadic_complex')
    b_freq = bands.fcentre;
    neg = bands.src_lo >= floor(n / 2);
    b_freq(neg) = -b_freq(neg);
    b_tiling = zeros(size(b_freq));
else
    b_freq = bands.fcentre;
    b_tiling = double(bands.out_off >= floor(n / 2) + 1);
end
end
