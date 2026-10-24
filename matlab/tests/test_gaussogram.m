function test_gaussogram()
%TEST_GAUSSOGRAM  Smoke + correctness tests for the gaussogram MATLAB wrapper.
%   Run from the matlab/ directory (or with matlab/ on the path) after BUILD_MEX:
%       test_gaussogram
%   Errors on the first failing assertion; prints a summary on success.

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));   % ensure the wrapper .m files are reachable

passed = 0;
passed = passed + run_case(@t_output_len);
passed = passed + run_case(@t_forward_shape_and_finite);
passed = passed + run_case(@t_batch_matches_single);
passed = passed + run_case(@t_roundtrip_dyadic_real);
passed = passed + run_case(@t_scheme_bands_layout);
passed = passed + run_case(@t_to_grid_modes);
passed = passed + run_case(@t_to_grid_output_shape);
passed = passed + run_case(@t_to_grid_rejects_bad_coeffs);
passed = passed + run_case(@t_rejects_complex_input);
passed = passed + run_case(@t_rejects_non_power_of_two);
passed = passed + run_case(@t_handle_matches_function);
passed = passed + run_case(@t_handle_reused_and_batch);
passed = passed + run_case(@t_handle_rejects_wrong_length);
passed = passed + run_case(@t_handle_survives_save_load);
passed = passed + run_case(@t_stale_handle_rejected);
passed = passed + run_case(@t_rejects_dyadic_complex_forward);
passed = passed + run_case(@t_rejects_non_integer_n);
passed = passed + run_case(@t_partitions_and_inverse_validate);

fprintf('\nAll %d gaussogram test cases passed.\n', passed);
end

% -- helpers ---------------------------------------------------------------

function ok = run_case(fn)
name = func2str(fn);
try
    fn();
    fprintf('  ok   %s\n', name);
    ok = 1;
catch err
    fprintf(2, '  FAIL %s: %s\n', name, err.message);
    rethrow(err);
end
end

function assert_true(cond, msg)
if ~cond
    error('gaussogram:test', '%s', msg);
end
end

% -- cases -----------------------------------------------------------------

function t_output_len()
n = 256;
assert_true(gaussogram_output_len(n, 'scheme', 'dyadic_dual_real') == n - 1, ...
    'dyadic_dual_real output_len should be N-1');
assert_true(gaussogram_output_len(n, 'scheme', 'dyadic_real') == n / 2 + 1, ...
    'dyadic_real output_len should be N/2+1');
end

function t_forward_shape_and_finite()
n = 128;
t = (0:n-1)' / n;
sig = sin(2 * pi * 8 * t);
c = gaussogram(sig);
assert_true(iscolumn(c), 'single-segment output should be a column vector');
assert_true(numel(c) == gaussogram_output_len(n), 'output length mismatch');
assert_true(~isreal(c), 'coefficients should be complex');
assert_true(all(isfinite(c)), 'coefficients should be finite');
end

function t_batch_matches_single()
n = 64;
count = 5;
rng(0);
X = randn(n, count);
batch = gaussogram(X);
assert_true(isequal(size(batch), [n - 1, count]), 'batch shape should be (N-1) x count');
for c = 1:count
    single_col = gaussogram(X(:, c));
    assert_true(max(abs(batch(:, c) - single_col)) < 1e-12, ...
        sprintf('batch column %d should match single transform', c));
end
end

function t_roundtrip_dyadic_real()
n = 256;
rng(1);
sig = randn(n, 1);
c = gaussogram(sig, 'scheme', 'dyadic_real');
assert_true(numel(c) == n / 2 + 1, 'dyadic_real coeff count should be N/2+1');
rec = gaussogram_inverse_real(c);
assert_true(numel(rec) == n, 'reconstruction length should be N');
err = max(abs(rec - sig));
assert_true(err < 1e-9, sprintf('round-trip error too large: %g', err));
end

function t_scheme_bands_layout()
n = 128;
bands = gaussogram_scheme_bands(n);
nb = numel(bands.src_lo);
assert_true(nb > 0, 'expected at least one band');
assert_true(isequal(numel(bands.width), nb) && isequal(numel(bands.out_off), nb), ...
    'band field lengths should match');
assert_true(isequal(bands.src_hi, bands.src_lo + bands.width), 'src_hi == src_lo + width');
% out_off must cover the packed output exactly once, in order.
assert_true(bands.out_off(1) == 0, 'first band out_off should be 0');
total = bands.out_off(end) + bands.width(end);
assert_true(total == gaussogram_output_len(n), 'bands should tile the packed output');
end

function t_to_grid_modes()
n = 128;
t = (0:n-1)' / n;
sig = sin(2 * pi * 16 * t);
c = gaussogram(sig);
g_lin = gaussogram_to_grid(c, n, 'interp', 'linear');
g_near = gaussogram_to_grid(c, n, 'interp', 'nearest');
g_smooth = gaussogram_to_grid(c, n, 'interp', 'linear', 'smooth', [2, 4]);
g_blockf = gaussogram_to_grid(c, n, 'interp', 'linear', 'interp_freq', 'block');
g_linf = gaussogram_to_grid(c, n, 'interp', 'linear', 'interp_freq', 'linear');
g_linf_sm = gaussogram_to_grid(c, n, 'interp', 'linear', 'interp_freq', 'linear', 'smooth', [2, 4]);
assert_true(isequal(size(g_lin), [n / 2 + 1, n]), 'grid shape should be (N/2+1) x N');
assert_true(isequal(size(g_linf), [n / 2 + 1, n]), 'linear-freq grid shape should be (N/2+1) x N');
assert_true(all(isfinite(g_lin(:))) && all(isfinite(g_near(:))) && all(isfinite(g_smooth(:))), ...
    'grids should be finite');
assert_true(all(isfinite(g_linf(:))) && all(isfinite(g_linf_sm(:))), 'freq-interp grids should be finite');
assert_true(all(g_lin(:) >= 0), 'magnitude grid should be nonnegative');
% 'block' is the default, so it must match an explicit interp_freq='block'.
assert_true(isequal(g_lin, g_blockf), "interp_freq='block' should be the default");
% Frequency interpolation must change the result vs block fill.
assert_true(~isequal(g_blockf, g_linf), "interp_freq='linear' should differ from 'block'");
assert_true(~isequal(g_linf, g_linf_sm), 'smoothing should change the linear-freq grid');
end

function t_rejects_complex_input()
threw = false;
try
    gaussogram(complex(ones(64, 1), 1));
catch
    threw = true;
end
assert_true(threw, 'complex input should be rejected');
end

function t_rejects_non_power_of_two()
threw = false;
try
    gaussogram(ones(100, 1));
catch
    threw = true;
end
assert_true(threw, 'non-power-of-two length should be rejected');
end

function t_handle_matches_function()
n = 256;
t = (0:n-1)' / n;
sig = sin(2 * pi * 11 * t);
eng = Gaussogram1d(n);
assert_true(eng.n == n, 'engine n should match');
assert_true(eng.output_len == gaussogram_output_len(n), 'engine output_len should match');
c_handle = eng.forward(sig);
c_func = gaussogram(sig);
assert_true(max(abs(c_handle - c_func)) < 1e-12, ...
    'reusable engine should match the one-shot function bit-for-bit');
delete(eng);
end

function t_handle_reused_and_batch()
n = 128;
eng = Gaussogram1d(n);
rng(7);
% Repeated single-segment calls should be stable and match the function.
for k = 1:4
    x = randn(n, 1);
    assert_true(max(abs(eng.forward(x) - gaussogram(x))) < 1e-12, ...
        sprintf('reused engine call %d should match function', k));
end
% Batch through the same engine.
X = randn(n, 6);
batch = eng.forward(X);
assert_true(isequal(size(batch), [n - 1, 6]), 'engine batch shape should be (N-1) x count');
for c = 1:6
    assert_true(max(abs(batch(:, c) - eng.forward(X(:, c)))) < 1e-12, ...
        'engine batch column should match single transform');
end
delete(eng);
end

function t_handle_rejects_wrong_length()
eng = Gaussogram1d(64);
threw = false;
try
    eng.forward(ones(128, 1));   % length mismatch
catch
    threw = true;
end
assert_true(threw, 'engine should reject a wrong-length signal');
delete(eng);
end

function t_to_grid_output_shape()
% Rows at 0, 2, 4, ... bins and columns at 0, 2, 4, ... samples are exactly
% the even rows/columns of the full-resolution render.
n = 256;
rng(11);
c = gaussogram(randn(n, 1));
for interp = {'linear', 'nearest'}
    for f = {'block', 'linear', 'gauss'}
        args = {'interp', interp{1}, 'interp_freq', f{1}};
        full = gaussogram_to_grid(c, n, args{:});
        assert_true(isequal(size(full), [n / 2 + 1, n]), 'default grid should be (N/2+1) x N');
        small = gaussogram_to_grid(c, n, args{:}, 'n_freq', n / 4 + 1, 'n_time', n / 2);
        assert_true(isequal(small, full(1:2:end, 1:2:end)), ...
            sprintf('decimated %s/%s grid should subsample the full grid', interp{1}, f{1}));
    end
end
% Exact nearest-sample ties (same cases as the Python test): with n = 1024,
% n_time = 196 and a width-2 band, column 147 is the wrap-around tie and must
% take sample 0. Check via a width-2 band of a dyadic_real render.
n2 = 1024;
c2 = gaussogram(randn(n2, 1), 'scheme', 'dyadic_real');
bands = gaussogram_scheme_bands(n2, 'scheme', 'dyadic_real');
b = find(bands.width == 2, 1);
g2 = gaussogram_to_grid(c2, n2, 'scheme', 'dyadic_real', 'interp', 'nearest', ...
    'n_time', 196);
row = bands.src_lo(b) + 1;   % 1-based grid row of the band's first bin
mag = abs(c2(bands.out_off(b) + (1:2)));
assert_true(g2(row, 148) == mag(1), 'wrap-around tie should take sample 0');
threw = false;
try
    gaussogram_to_grid(c, n, 'n_freq', 1);
catch err
    threw = strcmp(err.identifier, 'gaussogram:shape');
end
assert_true(threw, 'n_freq < 2 should be rejected');
end

function t_to_grid_rejects_bad_coeffs()
n = 64;
c = gaussogram(randn(n, 1));   % default dyadic_dual_real: N - 1 coeffs
cases = { ...
    {c, {'scheme', 'dyadic_real'}}, ...   % too many for dyadic_real
    {c(1:end - 1), {}}, ...               % too short
    {[c, c], {}}, ...                     % a matrix, not a vector
    };
for i = 1:numel(cases)
    threw = false;
    try
        gaussogram_to_grid(cases{i}{1}, n, cases{i}{2}{:});
    catch err
        threw = strcmp(err.identifier, 'gaussogram:coeffsLen');
    end
    assert_true(threw, sprintf('mismatched coeffs case %d should be rejected', i));
end
threw = false;
try
    gaussogram_to_grid(complex(zeros(n, 1)), n, 'scheme', 'dyadic_complex');
catch err
    threw = strcmp(err.identifier, 'gaussogram:scheme');
end
assert_true(threw, 'dyadic_complex should be rejected by to_grid');
end

function t_partitions_and_inverse_validate()
assert_true(isequal(gaussogram_partitions(8), int32([1; 2; 4; 6; 8])), ...
    'real partitions of 8 should match the legacy layout');
bad = {{12, 'real'}, {2, 'real'}, {12, 'complex'}, {4, 'complex'}};
for i = 1:numel(bad)
    threw = false;
    try
        gaussogram_partitions(bad{i}{:});
    catch err
        threw = strcmp(err.identifier, 'gaussogram:partitions');
    end
    assert_true(threw, sprintf('partitions(%d, %s) should be rejected', bad{i}{:}));
end
c = gaussogram(randn(64, 1), 'scheme', 'dyadic_real');
threw = false;
try
    gaussogram_inverse_real([c, c]);
catch err
    threw = strcmp(err.identifier, 'gaussogram:coeffsShape');
end
assert_true(threw, 'inverse_real should reject a matrix of coefficients');
end

function t_handle_survives_save_load()
% The native pointer is Transient: a loaded copy (as parfor workers get) must
% rebuild its engine rather than dereference the saved pointer value.
n = 128;
eng = Gaussogram1d(n);
x = randn(n, 1);
expected = eng.forward(x);
f = [tempname, '.mat'];
cleanup = onCleanup(@() delete(f));
save(f, 'eng');
delete(eng);
s = load(f);
assert_true(max(abs(s.eng.forward(x) - expected)) < 1e-12, ...
    'a loaded engine should rebuild and match the original');
delete(s.eng);
end

function t_stale_handle_rejected()
% A raw handle value must be rejected once destroyed, and destroying it again
% must be a no-op (not a double free).
h = gaussogram_mex('create', 64, 0, 0, 0);
gaussogram_mex('destroy', h);
gaussogram_mex('destroy', h);
threw = false;
try
    gaussogram_mex('forward_handle', h, ones(64, 1));
catch err
    threw = strcmp(err.identifier, 'gaussogram:handle');
end
assert_true(threw, 'a destroyed handle should be rejected with gaussogram:handle');
threw = false;
try
    gaussogram_mex('forward_handle', uint64(12345), ones(64, 1));
catch err
    threw = strcmp(err.identifier, 'gaussogram:handle');
end
assert_true(threw, 'an arbitrary handle value should be rejected');
% A new engine never takes over a destroyed handle's id, even if the
% allocator reuses the destroyed engine's address.
h2 = gaussogram_mex('create', 64, 0, 0, 0);
assert_true(h2 ~= h, 'handle ids must not be reused');
gaussogram_mex('destroy', h);   % stale: must not destroy h2's engine
c = gaussogram_mex('forward_handle', h2, ones(64, 1));
assert_true(numel(c) == 63, 'the live engine should be unaffected by a stale destroy');
gaussogram_mex('destroy', h2);
threw = false;
try
    gaussogram_mex('create', Inf, 0, 0, 0);
catch err
    threw = strcmp(err.identifier, 'gaussogram:n');
end
assert_true(threw, 'a non-finite n should be rejected');
end

function t_rejects_dyadic_complex_forward()
threw = false;
try
    gaussogram(ones(64, 1), 'scheme', 'dyadic_complex');
catch err
    threw = strcmp(err.identifier, 'gaussogram:scheme');
end
assert_true(threw, 'dyadic_complex forward should be rejected clearly');
end

function t_rejects_non_integer_n()
threw = false;
try
    gaussogram_output_len(64.5);
catch
    threw = true;
end
assert_true(threw, 'a non-integer n should be rejected');
end
