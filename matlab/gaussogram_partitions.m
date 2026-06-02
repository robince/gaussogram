function parts = gaussogram_partitions(n, kind)
%GAUSSOGRAM_PARTITIONS  Legacy partition boundaries for length N.
%   parts = GAUSSOGRAM_PARTITIONS(n)            % real scheme (default)
%   parts = GAUSSOGRAM_PARTITIONS(n, 'real')    % real scheme
%   parts = GAUSSOGRAM_PARTITIONS(n, 'complex') % complex scheme
%
%   Returns the int32 partition boundaries used by the legacy pygft layout.
%   Provided for parity/inspection; the modern API uses GAUSSOGRAM_SCHEME_BANDS.
%
%   See also GAUSSOGRAM_SCHEME_BANDS.

if nargin < 2
    kind = 'real';
end
switch lower(string(kind))
    case "real"
        kind_id = 0;
    case "complex"
        kind_id = 1;
    otherwise
        error('gaussogram:kind', "kind must be 'real' or 'complex'.");
end

gaussogram_check_built();
parts = gaussogram_mex('partitions', double(n), kind_id);
end
