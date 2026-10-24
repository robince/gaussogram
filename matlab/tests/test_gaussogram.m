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
passed = passed + run_case(@t_dyadic_complex_forward);
passed = passed + run_case(@t_rejects_non_integer_n);
passed = passed + run_case(@t_partitions_and_inverse_validate);
passed = passed + run_case(@t_bpo_output_len_and_bands);
passed = passed + run_case(@t_bpo_roundtrip_dyadic_real);
passed = passed + run_case(@t_bpo_handle_matches_function);
passed = passed + run_case(@t_bpo_rejects_invalid);
passed = passed + run_case(@t_bpo_to_grid);
passed = passed + run_case(@t_handle_inverse);
passed = passed + run_case(@t_geometry_fields);
passed = passed + run_case(@t_adjacency_structure);
passed = passed + run_case(@t_adjacency_circular_wrap);
passed = passed + run_case(@t_sparse_adjacency);
passed = passed + run_case(@t_geometry_adjacency_match_python);

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
c = gaussogram(randn(64, 1), 'scheme', 'dyadic_real');   % 33 coeffs
for bad = {[c, c], reshape(c, [1, 3, 11])}
    threw = false;
    try
        gaussogram_inverse_real(bad{1});
    catch err
        threw = strcmp(err.identifier, 'gaussogram:coeffsShape');
    end
    assert_true(threw, sprintf('inverse_real should reject a %s array of coefficients', ...
        mat2str(size(bad{1}))));
end
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

function t_dyadic_complex_forward()
n = 64;
rng(5);
Z = complex(randn(n, 3), randn(n, 3));
c = gaussogram(Z(:, 1), 'scheme', 'dyadic_complex');
assert_true(isequal(size(c), [n, 1]), 'dyadic_complex output should be N x 1');
assert_true(all(isfinite(c)), 'dyadic_complex coefficients should be finite');
% Linear in the input, and a real signal is promoted to complex.
cx = gaussogram(real(Z(:, 1)), 'scheme', 'dyadic_complex');
cy = gaussogram(imag(Z(:, 1)), 'scheme', 'dyadic_complex');
assert_true(max(abs(c - (cx + 1i * cy))) < 1e-10, 'complex forward should be linear');
% Batch (one column at a time in the MEX) matches single transforms, and a
% persistent engine matches the function.
batch = gaussogram(Z, 'scheme', 'dyadic_complex');
eng = Gaussogram1d(n, 'scheme', 'dyadic_complex');
assert_true(eng.output_len == n && ~eng.invertible, 'complex engine properties');
for k = 1:3
    ck = gaussogram(Z(:, k), 'scheme', 'dyadic_complex');
    assert_true(isequal(batch(:, k), ck), 'complex batch column should match single');
    assert_true(isequal(eng.forward(Z(:, k)), ck), 'complex engine should match function');
end
delete(eng);
% Complex input to a real scheme is still rejected.
threw = false;
try
    gaussogram(Z(:, 1));
catch err
    threw = strcmp(err.identifier, 'gaussogram:signal');
end
assert_true(threw, 'complex input to a real scheme should be rejected');
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

% -- bands_per_octave ------------------------------------------------------

function t_bpo_output_len_and_bands()
n = 256;
for scheme = {'dyadic_dual_real', 'dyadic_real'}
    nb_prev = 0;
    for bpo = [1, 2, 4, 8]
        args = {'scheme', scheme{1}, 'bands_per_octave', bpo};
        olen = gaussogram_output_len(n, args{:});
        assert_true(olen == gaussogram_output_len(n, 'scheme', scheme{1}), ...
            sprintf('%s output_len should not depend on bpo (bpo=%d)', scheme{1}, bpo));
        bands = gaussogram_scheme_bands(n, args{:});
        nb = numel(bands.width);
        assert_true(nb > nb_prev, 'more bands per octave should give more bands');
        nb_prev = nb;
        assert_true(bands.out_off(1) == 0 && ...
            isequal(bands.out_off(2:end), bands.out_off(1:end - 1) + bands.width(1:end - 1)), ...
            'bands should pack the output contiguously');
        assert_true(bands.out_off(end) + bands.width(end) == olen, 'bands should tile the output');
        assert_true(all(bitand(bands.width, bands.width - 1) == 0), 'band widths are powers of two');
        if strcmp(scheme{1}, 'dyadic_real')
            % A disjoint cover of 0..N/2.
            assert_true(isequal(bands.src_lo(2:end), bands.src_hi(1:end - 1)) && ...
                bands.src_lo(1) == 0 && bands.src_hi(end) == n / 2 + 1, ...
                'dyadic_real bands should partition 0..N/2');
        end
    end
end
% bpo = 2 at n = 256: DC, the width-1 octave [1,2), six split octaves
% [2,4) .. [64,128), and the Nyquist bin: 1 + 1 + 2*6 + 1 = 15 bands.
bands = gaussogram_scheme_bands(n, 'scheme', 'dyadic_real', 'bands_per_octave', 2);
assert_true(numel(bands.width) == 15, 'dyadic_real bpo=2 band count at n=256');
end

function t_bpo_roundtrip_dyadic_real()
n = 256;
rng(21);
sig = randn(n, 1);
c1 = gaussogram(sig, 'scheme', 'dyadic_real');
for bpo = [1, 2, 4, 8]
    c = gaussogram(sig, 'scheme', 'dyadic_real', 'bands_per_octave', bpo);
    assert_true(numel(c) == n / 2 + 1, 'dyadic_real bpo coeff count should be N/2+1');
    if bpo > 1
        assert_true(max(abs(c - c1)) > 1e-6, 'bpo > 1 should change the coefficients');
    end
    rec = gaussogram_inverse_real(c, 'bands_per_octave', bpo);
    err = max(abs(rec - sig));
    assert_true(err < 1e-9, sprintf('bpo=%d round-trip error too large: %g', bpo, err));
    rec_box = gaussogram_inverse_real( ...
        gaussogram(sig, 'scheme', 'dyadic_real', 'window', 'box', 'bands_per_octave', bpo), ...
        'window', 'box', 'bands_per_octave', bpo);
    assert_true(max(abs(rec_box - sig)) < 1e-9, sprintf('box bpo=%d round trip', bpo));
end
cdual = gaussogram(sig, 'bands_per_octave', 4, 'nyquist_flat_top', true);
assert_true(numel(cdual) == n - 1 && all(isfinite(cdual)), 'dual bpo=4 flat-top forward');
end

function t_bpo_handle_matches_function()
n = 128;
rng(22);
X = randn(n, 4);
for scheme = {'dyadic_dual_real', 'dyadic_real'}
    eng = Gaussogram1d(n, 'scheme', scheme{1}, 'bands_per_octave', 4);
    assert_true(eng.bands_per_octave == 4, 'engine should record bands_per_octave');
    expected = gaussogram(X, 'scheme', scheme{1}, 'bands_per_octave', 4);
    assert_true(isequal(eng.forward(X), expected), 'bpo engine batch should match function');
    assert_true(isequal(eng.forward(X(:, 2)), expected(:, 2)), ...
        'bpo engine single should match function');
    % A loaded copy rebuilds with the same bands_per_octave.
    f = [tempname, '.mat'];
    save(f, 'eng');
    s = load(f);
    delete(f);
    assert_true(isequal(s.eng.forward(X(:, 1)), expected(:, 1)), ...
        'a loaded bpo engine should rebuild with the same layout');
    delete(s.eng);
    delete(eng);
end
end

function t_bpo_rejects_invalid()
bad = {0, 3, 6, 1.5, -2, [1, 2], 'two', NaN, Inf};
for i = 1:numel(bad)
    calls = { ...
        @() gaussogram(randn(64, 1), 'bands_per_octave', bad{i}), ...
        @() gaussogram_output_len(64, 'bands_per_octave', bad{i}), ...
        @() Gaussogram1d(64, 'bands_per_octave', bad{i}), ...
        @() gaussogram_inverse_real(complex(zeros(33, 1)), 'bands_per_octave', bad{i})};
    for k = 1:numel(calls)
        threw = false;
        try
            calls{k}();
        catch err
            threw = strcmp(err.identifier, 'gaussogram:bands_per_octave');
        end
        assert_true(threw, sprintf('bands_per_octave = %s should be rejected (call %d)', ...
            mat2str(bad{i}), k));
    end
end
% dyadic_complex has no sub-octave variant.
threw = false;
try
    gaussogram(randn(64, 1), 'scheme', 'dyadic_complex', 'bands_per_octave', 2);
catch err
    threw = strcmp(err.identifier, 'gaussogram:create');
end
assert_true(threw, 'dyadic_complex with bands_per_octave > 1 should be rejected');
end

function t_bpo_to_grid()
n = 256;
t = (0:n-1)' / n;
sig = sin(2 * pi * 40 * t) + sin(2 * pi * 46 * t);
c1 = gaussogram(sig);
c4 = gaussogram(sig, 'bands_per_octave', 4);
for f = {'block', 'linear', 'gauss'}
    g1 = gaussogram_to_grid(c1, n, 'interp_freq', f{1});
    g4 = gaussogram_to_grid(c4, n, 'interp_freq', f{1}, 'bands_per_octave', 4);
    assert_true(isequal(size(g4), [n / 2 + 1, n]) && all(isfinite(g4(:))), ...
        sprintf('bpo=4 %s grid shape', f{1}));
    assert_true(~isequal(g1, g4), sprintf('bpo=4 %s grid should differ from bpo=1', f{1}));
end
end

function t_handle_inverse()
n = 128;
rng(23);
sig = randn(n, 1);
for bpo = [1, 2]
    eng = Gaussogram1d(n, 'scheme', 'dyadic_real', 'bands_per_octave', bpo);
    assert_true(eng.invertible, 'dyadic_real engine should be invertible');
    c = eng.forward(sig);
    rec = eng.inverse(c);
    assert_true(iscolumn(rec) && numel(rec) == n, 'handle inverse should return N x 1');
    assert_true(max(abs(rec - sig)) < 1e-9, 'handle inverse round trip');
    assert_true(isequal(rec, gaussogram_inverse_real(c, 'bands_per_octave', bpo)), ...
        'handle inverse should match gaussogram_inverse_real');
    assert_true(max(abs(eng.inverse(c.') - sig)) < 1e-9, 'row coeffs should be accepted');
    threw = false;
    try
        eng.inverse(c(1:end - 1));
    catch err
        threw = strcmp(err.identifier, 'gaussogram:inverse');
    end
    assert_true(threw, 'a wrong-length inverse should be rejected');
    delete(eng);
end
eng = Gaussogram1d(n);
assert_true(~eng.invertible, 'dual engine inverse is not wired into MATLAB');
threw = false;
try
    eng.inverse(eng.forward(sig));
catch err
    threw = strcmp(err.identifier, 'gaussogram:notInvertible');
end
assert_true(threw, 'dual engine inverse should be rejected clearly');
delete(eng);
end

% -- geometry / adjacency --------------------------------------------------

function t_geometry_fields()
n = 128;
for cfg = {{'dyadic_dual_real', 1}, {'dyadic_dual_real', 4}, {'dyadic_real', 2}, ...
        {'dyadic_complex', 1}}
    args = {'scheme', cfg{1}{1}, 'bands_per_octave', cfg{1}{2}};
    geo = gaussogram_coefficient_geometry(n, args{:});
    olen = gaussogram_output_len(n, args{:});
    bands = gaussogram_scheme_bands(n, args{:});
    assert_true(isequal(fieldnames(geo), ...
        {'time'; 'freq'; 'dt'; 'df'; 'logf'; 'level'; 'sub'; 'tiling'; 'band'}), ...
        'geometry field names should match Python CoeffGeometry');
    for f = fieldnames(geo).'
        assert_true(isequal(size(geo.(f{1})), [olen, 1]), ...
            sprintf('geometry field %s should be output_len x 1', f{1}));
    end
    assert_true(all(geo.dt .* geo.df == n), 'every tile should have area n');
    assert_true(all(geo.time >= 0 & geo.time < n), 'tile times should lie in [0, n)');
    assert_true(isequal(geo.df, bands.width(geo.band)), 'band index should be 1-based');
    assert_true(isequal(abs(geo.freq), bands.fcentre(geo.band)), 'freq is the band centre');
    assert_true(isequal(geo.logf, log2(max(abs(geo.freq), 0.5))), 'logf definition');
    assert_true(isequal(geo.level, floor(log2(max(abs(geo.freq), 1)))), 'level definition');
    if strcmp(cfg{1}{1}, 'dyadic_dual_real')
        assert_true(isequal(geo.tiling, double((1:olen).' > n / 2 + 1)), ...
            'tiling A is the first N/2+1 coefficients');
    else
        assert_true(all(geo.tiling == 0), 'single-tiling schemes have tiling 0');
    end
end
end

function t_adjacency_structure()
n = 128;
for cfg = {{'dyadic_dual_real', 1}, {'dyadic_dual_real', 2}, {'dyadic_real', 4}, ...
        {'dyadic_complex', 1}}
    for circular = [false, true]
        args = {'scheme', cfg{1}{1}, 'bands_per_octave', cfg{1}{2}};
        adj = gaussogram_coefficient_adjacency(n, args{:}, 'circular', circular);
        olen = gaussogram_output_len(n, args{:});
        e = numel(adj.edge_type);
        assert_true(isequal(fieldnames(adj), ...
            {'edge_index'; 'edge_type'; 'edge_weight'; 'type_names'}), 'adjacency field names');
        assert_true(isequal(size(adj.edge_index), [2, e]) && ...
            isequal(size(adj.edge_weight), [1, e]), 'adjacency field shapes');
        assert_true(all(adj.edge_index(1, :) < adj.edge_index(2, :)), 'edges stored with i < j');
        assert_true(min(adj.edge_index(:)) >= 1 && max(adj.edge_index(:)) <= olen, ...
            'edge indices should be 1-based coefficient indices');
        triples = [adj.edge_index; adj.edge_type].';
        assert_true(size(unique(triples, 'rows'), 1) == e, 'no duplicate edges');
        assert_true(all(adj.edge_weight == 1), 'structural weights are 1');
        has_dual = any(adj.edge_type == 2);
        assert_true(has_dual == strcmp(cfg{1}{1}, 'dyadic_dual_real'), ...
            'dual edges only for dyadic_dual_real');
        % The time edges within a band of width w number w-1 (+1 if circular, w >= 3).
        bands = gaussogram_scheme_bands(n, args{:});
        expected_time = sum(max(bands.width - 1, 0)) + circular * sum(bands.width >= 3);
        assert_true(nnz(adj.edge_type == 0) == expected_time, 'time edge count');
    end
end
end

function t_adjacency_circular_wrap()
n = 64;
bands = gaussogram_scheme_bands(n);
lin = gaussogram_coefficient_adjacency(n);
circ = gaussogram_coefficient_adjacency(n, 'circular', true);
% Each band of width >= 3 gains exactly its first<->last wrap-around time edge.
wrap = zeros(2, 0);
for b = find(bands.width >= 3).'
    wrap(:, end + 1) = bands.out_off(b) + [1; bands.width(b)];   %#ok<AGROW>
end
tl = lin.edge_index(:, lin.edge_type == 0);
tc = circ.edge_index(:, circ.edge_type == 0);
assert_true(size(tc, 2) - size(tl, 2) == size(wrap, 2), 'circular adds one time edge per band');
assert_true(isequal(sortrows(setdiff(tc.', tl.', 'rows')), sortrows(wrap.')), ...
    'the added time edges are the wrap-around pairs');
% No non-circular edge joins the end of a band to its start.
assert_true(~any(ismember(wrap.', tl.', 'rows')), 'non-circular graph has no wrap edges');
% Circular cross-band edges wrap too, so there are at least as many.
assert_true(nnz(circ.edge_type == 1) >= nnz(lin.edge_type == 1), 'circular band edges');
threw = false;
try
    gaussogram_coefficient_adjacency(n, 'circular', 'yes');
catch err
    threw = strcmp(err.identifier, 'gaussogram:circular');
end
assert_true(threw, 'a non-logical circular should be rejected');
end

function t_sparse_adjacency()
n = 64;
adj = gaussogram_coefficient_adjacency(n);
olen = gaussogram_output_len(n);
A = gaussogram_to_sparse_adjacency(adj, olen);
assert_true(issparse(A) && isequal(size(A), [olen, olen]), 'sparse adjacency shape');
assert_true(isequal(A, A.'), 'default adjacency should be symmetric');
ntb = nnz(adj.edge_type <= 1);
assert_true(full(sum(A(:))) == 2 * ntb, 'default types are time + band');
Ad = gaussogram_to_sparse_adjacency(adj, olen, 'types', {'time', 'band', 'dual'});
assert_true(full(sum(Ad(:))) == 2 * numel(adj.edge_type), 'all types include dual');
At = gaussogram_to_sparse_adjacency(adj, olen, 'types', 0, 'symmetric', false);
assert_true(nnz(At) == nnz(adj.edge_type == 0) && istriu(At), ...
    'one-directional time-only adjacency is upper triangular');
assert_true(isequal(At, gaussogram_to_sparse_adjacency(adj, olen, 'types', {'time'}, ...
    'symmetric', false)), 'type names and ids should select the same edges');
adjw = adj;
adjw.edge_weight(:) = 0.5;
Aw = gaussogram_to_sparse_adjacency(adjw, olen, 'weighted', true);
assert_true(isequal(Aw, 0.5 * A), 'weighted uses edge_weight');
% Each tiling's time + band graph is connected; the dual edges join the two.
assert_true(max(conncomp(graph(A))) == 2, 'time + band graph has one component per tiling');
assert_true(all(conncomp(graph(Ad)) == 1), 'all-types graph should be connected');
for bad = {{adj, olen - 1}, {adj, olen, 'types', {'scale'}}, {struct('x', 1), olen}}
    threw = false;
    try
        gaussogram_to_sparse_adjacency(bad{1}{:});
    catch err
        threw = startsWith(err.identifier, 'gaussogram:');
    end
    assert_true(threw, 'invalid sparse adjacency arguments should be rejected');
end
end

function t_geometry_adjacency_match_python()
% Reference values computed with the Python package (gaussogram.
% coefficient_geometry / coefficient_adjacency / to_sparse_adjacency) on the
% same configurations. Python indices are 0-based; these are converted to
% MATLAB's 1-based ones. Columns: n, scheme, bands_per_octave, circular,
% edge counts per type [time band dual], an order-sensitive edge hash
% sum(k * (i + 1000 j)) over edges k, geometry sums [time freq dt level sub
% tiling band logf sum(k*sub)], nnz and sum of the default sparse matrix, sum
% of the all-types sparse matrix.
ref = { ...
    {16, 'dyadic_dual_real', 1, false, [8, 17, 13], 8459237, ...
     [64.0, 59.0, 112.0, 23.0, 1.0, 6.0, 70.0, 25.509775004326936, 2.0], 50, 50, 76}, ...
    {64, 'dyadic_dual_real', 2, true, [53, 79, 66], 875766860, ...
     [1440.0, 966.0, 1152.0, 201.0, 30.0, 30.0, 737.0, 224.5843997480469, 1042.0], 264, 264, 396}, ...
    {64, 'dyadic_dual_real', 4, false, [35, 65, 61], 591567950, ...
     [1120.0, 1004.0, 1792.0, 201.0, 82.0, 30.0, 1087.0, 227.17472772880168, 2919.0], 200, 200, 322}, ...
    {64, 'dyadic_real', 2, true, [26, 45, 0], 58237975, ...
     [704.0, 542.0, 704.0, 103.0, 16.0, 0.0, 255.0, 118.39492473795067, 307.0], 142, 142, 142}, ...
    {32, 'dyadic_complex', 1, true, [26, 43, 0], 43486252, ...
     [352.0, -2.0, 320.0, 69.0, 2.0, 0.0, 176.0, 84.37895002019238, 33.0], 138, 138, 138}, ...
    {128, 'dyadic_dual_real', 1, true, [122, 215, 154], 10607122455, ...
     [7296.0, 3475.0, 1664.0, 522.0, 1.0, 62.0, 1147.0, 557.2676750447116, 2.0], 674, 674, 982}, ...
    };
for r = 1:numel(ref)
    [n, scheme, bpo, circular, counts, ehash, gsum, nz, asum, adsum] = ref{r}{:};
    label = sprintf('%s n=%d bpo=%d circular=%d', scheme, n, bpo, circular);
    args = {'scheme', scheme, 'bands_per_octave', bpo};
    adj = gaussogram_coefficient_adjacency(n, args{:}, 'circular', circular);
    geo = gaussogram_coefficient_geometry(n, args{:});
    got_counts = arrayfun(@(t) nnz(adj.edge_type == t), 0:2);
    assert_true(isequal(got_counts, counts), ['edge counts differ from Python: ', label]);
    k = 1:numel(adj.edge_type);
    got_hash = sum(k .* (adj.edge_index(1, :) + 1000 * adj.edge_index(2, :)));
    assert_true(got_hash == ehash, ['edge order/content differs from Python: ', label]);
    kk = (1:numel(geo.sub)).';
    got_g = [sum(geo.time), sum(geo.freq), sum(geo.dt), sum(geo.level), sum(geo.sub), ...
        sum(geo.tiling), sum(geo.band), sum(geo.logf), sum(kk .* geo.sub)];
    assert_true(max(abs(got_g - gsum)) < 1e-9, ['geometry differs from Python: ', label]);
    olen = gaussogram_output_len(n, args{:});
    A = gaussogram_to_sparse_adjacency(adj, olen);
    Ad = gaussogram_to_sparse_adjacency(adj, olen, 'types', {'time', 'band', 'dual'});
    assert_true(nnz(A) == nz && full(sum(A(:))) == asum && full(sum(Ad(:))) == adsum, ...
        ['sparse adjacency differs from Python: ', label]);
end
% Full edge list for the smallest case (Python edge_index + 1, edge_type).
edges16 = [ ...
    3, 5, 6, 7, 10, 12, 13, 14, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6, 7, 8, 10, 10, 11, 11, 11, ...
    2, 2, 3, 4, 3, 3, 4, 4, 4, 5, 6, 7, 8; ...
    4, 6, 7, 8, 11, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 6, 9, 9, 9, 9, 12, 13, 14, 15, 13, ...
    10, 11, 10, 11, 12, 13, 14, 15, 13, 12, 13, 14, 15];
types16 = [zeros(1, 8), ones(1, 17), 2 * ones(1, 13)];
adj = gaussogram_coefficient_adjacency(16);
assert_true(isequal(adj.edge_index, edges16) && isequal(adj.edge_type, types16), ...
    'n=16 edge list should match Python exactly');
end
