function A = gaussogram_to_sparse_adjacency(adj, n_nodes, varargin)
%GAUSSOGRAM_TO_SPARSE_ADJACENCY  Sparse adjacency matrix from a coefficient graph.
%   A = GAUSSOGRAM_TO_SPARSE_ADJACENCY(adj, n_nodes)
%   A = GAUSSOGRAM_TO_SPARSE_ADJACENCY(adj, n_nodes, 'types', t, ...
%           'symmetric', tf, 'weighted', tf)
%
%   Builds an n_nodes x n_nodes MATLAB sparse matrix from the struct returned
%   by GAUSSOGRAM_COEFFICIENT_ADJACENCY, for cluster-based statistics
%   (FieldTrip / MNE permutation clustering, TFCE) or graph connected
%   components (conncomp(graph(A))). Port of the Python
%   gaussogram.to_sparse_adjacency; rows and columns are the 1-based
%   coefficient indices.
%
%   Inputs:
%     n_nodes     - number of coefficients (= GAUSSOGRAM_OUTPUT_LEN). Pass it
%                   explicitly; the edge list alone can under-count nodes if
%                   the last coefficients are isolated.
%   Name-value options:
%     'types'     - edge relations to include, as names ({'time', 'band',
%                   'dual'}) or type ids (0, 1, 2). Default {'time', 'band'},
%                   the genuine time-frequency neighbourhood. 'dual' (the
%                   tiling-A <-> B redundancy edges) is excluded by default:
%                   those link redundant copies of the same location, which
%                   would merge them in a cluster.
%     'symmetric' - true (default) adds both directions so A is symmetric.
%     'weighted'  - true uses adj.edge_weight; false (default) binary (1).
%   An (i, j) pair present under several included types sums its entries,
%   as in Python (scipy's coo -> csr conversion).
%
%   Caveat for multiresolution clustering: a low-frequency coefficient spans
%   the whole time axis (dt = n/width), so band edges can bridge temporally
%   distant activity through coarse tiles and inflate clusters. To avoid that,
%   use 'types', {'time'}, or weight/prune edges using
%   GAUSSOGRAM_COEFFICIENT_GEOMETRY before building the matrix.
%
%   See also GAUSSOGRAM_COEFFICIENT_ADJACENCY, GAUSSOGRAM_COEFFICIENT_GEOMETRY.

p = inputParser;
p.FunctionName = 'gaussogram_to_sparse_adjacency';
addParameter(p, 'types', {'time', 'band'});
addParameter(p, 'symmetric', true, @(t) isscalar(t) && (islogical(t) || isnumeric(t)));
addParameter(p, 'weighted', false, @(t) isscalar(t) && (islogical(t) || isnumeric(t)));
parse(p, varargin{:});

if ~isstruct(adj) || ~all(isfield(adj, {'edge_index', 'edge_type', 'edge_weight', 'type_names'}))
    error('gaussogram:adj', ...
        'adj must be a struct from gaussogram_coefficient_adjacency.');
end
if ~(isnumeric(n_nodes) && isscalar(n_nodes) && n_nodes >= 0 && n_nodes == floor(n_nodes))
    error('gaussogram:n_nodes', 'n_nodes must be a nonnegative integer.');
end
if ~isempty(adj.edge_index) && max(adj.edge_index(:)) > n_nodes
    error('gaussogram:n_nodes', ...
        'n_nodes (%d) is smaller than the largest node index in adj (%d).', ...
        n_nodes, max(adj.edge_index(:)));
end

sel = local_type_ids(p.Results.types, adj.type_names);
mask = ismember(adj.edge_type, sel);
rows = adj.edge_index(1, mask);
cols = adj.edge_index(2, mask);
if p.Results.weighted
    w = double(adj.edge_weight(mask));
else
    w = ones(1, nnz(mask));
end
if p.Results.symmetric
    [rows, cols] = deal([rows, cols], [cols, rows]);
    w = [w, w];
end
A = sparse(rows, cols, w, double(n_nodes), double(n_nodes));
end

function ids = local_type_ids(types, type_names)
%LOCAL_TYPE_IDS  Map edge type names / ids to numeric ids (0-based).
if ischar(types) || isstring(types)
    types = cellstr(types);
end
if isnumeric(types)
    types = num2cell(types);
end
if ~iscell(types)
    error('gaussogram:types', "types must be a cell array of names or numeric ids.");
end
ids = zeros(1, numel(types));
for k = 1:numel(types)
    t = types{k};
    if ischar(t) || isstring(t)
        hit = find(strcmpi(type_names, char(t)), 1);
        if isempty(hit)
            error('gaussogram:types', "unknown edge type '%s' (expected %s).", ...
                char(t), strjoin(type_names, ', '));
        end
        ids(k) = hit - 1;
    elseif isnumeric(t) && isscalar(t)
        ids(k) = double(t);
    else
        error('gaussogram:types', "types must be a cell array of names or numeric ids.");
    end
end
end
