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
passed = passed + run_case(@t_rejects_complex_input);
passed = passed + run_case(@t_rejects_non_power_of_two);
passed = passed + run_case(@t_handle_matches_function);
passed = passed + run_case(@t_handle_reused_and_batch);
passed = passed + run_case(@t_handle_rejects_wrong_length);

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
assert_true(isequal(size(g_lin), [n / 2 + 1, n]), 'grid shape should be (N/2+1) x N');
assert_true(all(isfinite(g_lin(:))) && all(isfinite(g_near(:))) && all(isfinite(g_smooth(:))), ...
    'grids should be finite');
assert_true(all(g_lin(:) >= 0), 'magnitude grid should be nonnegative');
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
