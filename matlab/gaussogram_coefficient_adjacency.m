function adj = gaussogram_coefficient_adjacency(n, varargin)
%GAUSSOGRAM_COEFFICIENT_ADJACENCY  Neighbourhood graph over the packed coefficients.
%   adj = GAUSSOGRAM_COEFFICIENT_ADJACENCY(n)
%   adj = GAUSSOGRAM_COEFFICIENT_ADJACENCY(n, 'scheme', s, 'window', w, ...
%             'top_window', t, 'bands_per_octave', b, 'circular', c)
%
%   Sparse undirected graph whose nodes are the output_len packed
%   coefficients, for GNNs, graph-Laplacian smoothness priors, structured
%   sparsity or cluster statistics. Signal-independent: compute once and
%   reuse. Port of the Python gaussogram.coefficient_adjacency; the edges are
%   identical, but node indices are 1-BASED here (Python's + 1).
%
%   Three edge types:
%     time (0) - consecutive coefficients within a band (intra-band time axis);
%     band (1) - frequency-adjacent bands within a tiling, time-aligned (the
%                cross-octave / cross-band "scale" relation);
%     dual (2) - tiling-A <-> tiling-B coefficients with overlapping frequency
%                support (the redundancy of 'dyadic_dual_real'; none otherwise).
%   Cross-band (band / dual) edges join coefficients whose time tiles overlap:
%   in a 2:1 octave step every interior coarse coefficient links to three fine
%   ones, and a fine coefficient on a coarse tile boundary links to both
%   coarse neighbours.
%
%   'circular' (default false): the transform is circular, but by default the
%   graph is not, so no edge joins the end of the record to its start and
%   clusters cannot wrap from t ~ n to t = 0. Pass true for the periodic
%   graph, which adds the wrap-around time edge within every band of width
%   >= 3 and wraps cross-band edges too.
%
%   Returns a struct (each edge stored once, with i < j, no duplicates):
%     adj.edge_index  - 2 x E double, 1-based endpoint indices into coeffs.
%     adj.edge_type   - 1 x E double, 0 = time, 1 = band, 2 = dual.
%     adj.edge_weight - 1 x E double structural weights (all 1).
%     adj.type_names  - {'time', 'band', 'dual'}; type t is type_names{t+1}.
%
%   See also GAUSSOGRAM_TO_SPARSE_ADJACENCY, GAUSSOGRAM_COEFFICIENT_GEOMETRY.

[opts, circular] = local_parse(varargin{:});
bands = gaussogram_scheme_bands(n, 'scheme', opts.scheme, 'window', opts.window, ...
    'top_window', opts.top_window, 'bands_per_octave', opts.bands_per_octave);
n = double(n);
[b_freq, b_tiling] = gaussogram_band_freq_tiling(bands, n, opts.scheme);
nb = numel(bands.width);
% 0-based global index of each band's first coefficient; band b's
% coefficients are off(b) + (0:w-1), matching the Python gidx.
off = bands.out_off;
w_all = bands.width;
lo_all = bands.src_lo;
hi_all = bands.src_hi;

% Collect (src, dst, type) chunks in generation order (the same order as the
% Python implementation, so de-duplication keeps the same edge order).
src = {};
dst = {};
typ = {};

% time: consecutive coefficients inside each band
for b = 1:nb
    w = w_all(b);
    g = off(b) + (0:(w - 1));
    if w >= 2
        src{end + 1} = g(1:end - 1);          %#ok<AGROW>
        dst{end + 1} = g(2:end);              %#ok<AGROW>
        typ{end + 1} = zeros(1, w - 1);       %#ok<AGROW>
    end
    if circular && w >= 3
        src{end + 1} = g(end);                %#ok<AGROW>
        dst{end + 1} = g(1);                  %#ok<AGROW>
        typ{end + 1} = 0;                     %#ok<AGROW>
    end
end
% band: frequency-adjacent bands within each tiling, time-aligned
for tl = 0:1
    members = find(b_tiling == tl);
    [~, order] = sort(b_freq(members));       % stable, like Python's sorted
    members = members(order);
    for k = 1:(numel(members) - 1)
        a = members(k);
        c = members(k + 1);
        [s, d] = local_connect_by_time(off(a), w_all(a), off(c), w_all(c), circular);
        src{end + 1} = s;                     %#ok<AGROW>
        dst{end + 1} = d;                     %#ok<AGROW>
        typ{end + 1} = ones(1, numel(s));     %#ok<AGROW>
    end
end
% dual: tiling A <-> tiling B with overlapping frequency support
a_bands = find(b_tiling == 0);
b_bands = find(b_tiling == 1);
for a = a_bands.'
    for bb = b_bands.'
        if lo_all(a) < hi_all(bb) && lo_all(bb) < hi_all(a)
            [s, d] = local_connect_by_time(off(a), w_all(a), off(bb), w_all(bb), circular);
            src{end + 1} = s;                 %#ok<AGROW>
            dst{end + 1} = d;                 %#ok<AGROW>
            typ{end + 1} = 2 * ones(1, numel(s)); %#ok<AGROW>
        end
    end
end

adj.type_names = {'time', 'band', 'dual'};
if isempty(src)
    adj.edge_index = zeros(2, 0);
    adj.edge_type = zeros(1, 0);
    adj.edge_weight = zeros(1, 0);
    adj = orderfields(adj, {'edge_index', 'edge_type', 'edge_weight', 'type_names'});
    return;
end
s = [src{:}];
d = [dst{:}];
t = [typ{:}];
lo_i = min(s, d);
hi_i = max(s, d);
keep = lo_i ~= hi_i;   % drop any self-loops
lo_i = lo_i(keep);
hi_i = hi_i(keep);
t = t(keep);
% De-duplicate identical (i, j, type) triples, keeping first occurrences in
% generation order (as Python's np.unique + sort of the first indices).
[~, uniq] = unique([lo_i(:), hi_i(:), t(:)], 'rows', 'stable');
adj.edge_index = [lo_i(uniq); hi_i(uniq)] + 1;   % 1-based node indices
adj.edge_type = t(uniq);
adj.edge_weight = ones(1, numel(uniq));
adj = orderfields(adj, {'edge_index', 'edge_type', 'edge_weight', 'type_names'});
end

function [opts, circular] = local_parse(varargin)
%LOCAL_PARSE  Split off 'circular'; the rest go to GAUSSOGRAM_PARSE_OPTS.
circular = false;
rest = {};
k = 1;
while k <= numel(varargin)
    name = varargin{k};
    if (ischar(name) || isstring(name)) && strcmpi(name, 'circular')
        if k == numel(varargin)
            error('gaussogram:circular', "'circular' needs a value.");
        end
        val = varargin{k + 1};
        if ~(isscalar(val) && (islogical(val) || isnumeric(val)))
            error('gaussogram:circular', "'circular' must be a logical scalar.");
        end
        circular = logical(val);
        k = k + 2;
    else
        rest{end + 1} = name; %#ok<AGROW>
        k = k + 1;
    end
end
opts = gaussogram_parse_opts(rest{:});
end

function [coarse_idx, fine_idx] = local_connect_by_time(off1, w1, off2, w2, circular)
%LOCAL_CONNECT_BY_TIME  Time-aligned cross-band edges (0-based global indices).
%   Link each coefficient of the finer-time band to every coefficient of the
%   coarser-time band whose time tile it overlaps (one, or two when it
%   straddles a coarse tile boundary).
%
%   Sample j of a width-w band is the centre of the tile
%   [(j - 1/2)*n/w, (j + 1/2)*n/w). In units of n / (2*wf*wc) the fine tile j
%   spans ((2j-1)*wc, (2j+1)*wc) and the coarse tile k spans
%   ((2k-1)*wf, (2k+1)*wf), so the overlapping k follow in integer
%   arithmetic (exact in doubles: every operand is an integer far below
%   2^53, and floor/ceil of an exact integer quotient is then exact). A fine
%   tile is never wider than a coarse one, so at most two match. Past the
%   last coarse tile, circular wraps to coarse tile 0 (the transform is
%   periodic); otherwise the end stays on the last coarse tile.
if w1 >= w2
    off_f = off1; wf = w1; off_c = off2; wc = w2;
else
    off_f = off2; wf = w2; off_c = off1; wc = w1;
end
j = 0:(wf - 1);
% Smallest k with (2k+1)*wf > (2j-1)*wc; largest k with (2k-1)*wf < (2j+1)*wc.
k_lo = floor(((2 * j - 1) * wc - wf) / (2 * wf)) + 1;
k_hi = ceil(((2 * j + 1) * wc + wf) / (2 * wf)) - 1;
two = k_hi > k_lo;
jj = [j, j(two)];
kk = [k_lo, k_hi(two)];
if circular
    kk = mod(kk, wc);
else
    kk = min(max(kk, 0), wc - 1);
end
coarse_idx = off_c + kk;
fine_idx = off_f + jj;
end
