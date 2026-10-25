function opts = gaussogram_parse_opts(varargin)
%GAUSSOGRAM_PARSE_OPTS  Shared name-value option parser for the wrappers.
%   Fields: scheme (char), window (char), top_window (char), top_window_id
%   (C ABI id), bands_per_octave (double, a power of two >= 1).
%   'nyquist_flat_top' is accepted as a deprecated alias for 'top_window'.
p = inputParser;
p.FunctionName = 'gaussogram';
addParameter(p, 'scheme', 'dyadic_dual_real', @(s) ischar(s) || isstring(s));
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'top_window', '', @(s) ischar(s) || isstring(s));
addParameter(p, 'nyquist_flat_top', [], @(t) isempty(t) || (isscalar(t) && (islogical(t) || isnumeric(t))));
addParameter(p, 'bands_per_octave', 1);
parse(p, varargin{:});

opts.scheme = char(p.Results.scheme);
opts.window = char(p.Results.window);
[opts.top_window, opts.top_window_id] = gaussogram_resolve_top_window( ...
    p.Results.top_window, p.Results.nyquist_flat_top);
opts.bands_per_octave = gaussogram_check_bpo(p.Results.bands_per_octave);
end
