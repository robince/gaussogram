function opts = gaussogram_parse_opts(varargin)
%GAUSSOGRAM_PARSE_OPTS  Shared name-value option parser for the wrappers.
%   Fields: scheme (char), window (char), nyquist_flat_top (logical).
p = inputParser;
p.FunctionName = 'gaussogram';
addParameter(p, 'scheme', 'dyadic_dual_real', @(s) ischar(s) || isstring(s));
addParameter(p, 'window', 'gaussian', @(s) ischar(s) || isstring(s));
addParameter(p, 'nyquist_flat_top', false, @(t) isscalar(t) && (islogical(t) || isnumeric(t)));
parse(p, varargin{:});

opts.scheme = char(p.Results.scheme);
opts.window = char(p.Results.window);
opts.nyquist_flat_top = logical(p.Results.nyquist_flat_top);
end
