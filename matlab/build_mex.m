function build_mex()
%BUILD_MEX  Compile the gaussogram MEX against the Rust static library.
%   Builds crate gaussogram-ffi as a staticlib (cargo rustc) and links it into
%   matlab/gaussogram_mex.<mexext>. Run once after cloning or after changing the
%   Rust core/ffi. Requires cargo on PATH (or the CARGO env var) and a MEX C++
%   compiler configured (mex -setup C++).

root = fileparts(fileparts(mfilename('fullpath')));
releaseDir = fullfile(root, 'target', 'release');
includeDir = fullfile(root, 'crates', 'gaussogram-ffi', 'include');
sourceFile = fullfile(root, 'matlab', 'src', 'gaussogram_mex.cpp');
cargoExe = findCargoExecutable();

if ~isfile(sourceFile)
    error('build_mex:missingSources', 'Expected MEX source at %s.', sourceFile);
end

originalDir = pwd();
cleanup = onCleanup(@() cd(originalDir));  %#ok<NASGU>
cd(root);

if ismac
    % rustc fails if MACOSX_DEPLOYMENT_TARGET is set to an empty string.
    macosDeploymentTarget = '15.0';
    originalDeploymentTarget = getenv('MACOSX_DEPLOYMENT_TARGET');
    restoreDeploymentTarget = onCleanup(@() setenv('MACOSX_DEPLOYMENT_TARGET', originalDeploymentTarget)); %#ok<NASGU>
    if isempty(strtrim(originalDeploymentTarget))
        setenv('MACOSX_DEPLOYMENT_TARGET', macosDeploymentTarget);
    end
end

% Ensure the toolchain directory (which holds rustc next to cargo) is on PATH:
% when cargo is the real toolchain binary rather than the rustup proxy, it
% locates rustc via PATH, and MATLAB's system() PATH often omits it.
cargoDir = fileparts(char(cargoExe));
originalPath = getenv('PATH');
restorePath = onCleanup(@() setenv('PATH', originalPath));  %#ok<NASGU>
if ~isempty(cargoDir) && isempty(strfind([pathsep, originalPath, pathsep], [pathsep, cargoDir, pathsep])) %#ok<STREMP>
    setenv('PATH', [cargoDir, pathsep, originalPath]);
end

status = system(sprintf('"%s" rustc -p gaussogram-ffi --release -- --crate-type staticlib', cargoExe));
if status ~= 0
    error('build_mex:cargoFailed', 'cargo rustc failed with status %d using %s.', status, cargoExe);
end

libFile = findStaticLibrary(releaseDir);
if strlength(libFile) == 0
    error('build_mex:missingLibrary', 'Could not find gaussogram_ffi static library under %s.', releaseDir);
end

baseMexArgs = { ...
    'LINKEXPORTCPP=', ...                % export classic mexFunction (not the C++ MEX API)
    '-R2018a', ...                       % interleaved complex API
    '-outdir', fullfile(root, 'matlab'), ...
    ['-I', includeDir], ...
    char(libFile) ...
};

if ismac
    baseMexArgs = [
        baseMexArgs(1:2), ...
        { ...
            ['CXXFLAGS=$CXXFLAGS -mmacosx-version-min=', macosDeploymentTarget], ...
            ['LDFLAGS=$LDFLAGS -mmacosx-version-min=', macosDeploymentTarget] ...
        }, ...
        baseMexArgs(3:end) ...
    ];
end

mex(baseMexArgs{:}, '-output', 'gaussogram_mex', sourceFile);
fprintf('Built %s\n', fullfile(root, 'matlab', ['gaussogram_mex.', mexext]));
end

function libFile = findStaticLibrary(releaseDir)
patterns = {
    fullfile(releaseDir, 'libgaussogram_ffi*.a')
    fullfile(releaseDir, 'gaussogram_ffi*.lib')
    fullfile(releaseDir, 'libgaussogram_ffi*.lib')
    fullfile(releaseDir, 'deps', 'libgaussogram_ffi*.a')
    fullfile(releaseDir, 'deps', 'gaussogram_ffi*.lib')
    fullfile(releaseDir, 'deps', 'libgaussogram_ffi*.lib')
};

for idx = 1:numel(patterns)
    matches = dir(patterns{idx});
    if ~isempty(matches)
        [~, newest] = max([matches.datenum]);
        libFile = string(fullfile(matches(newest).folder, matches(newest).name));
        return;
    end
end

libFile = "";
end

function cargoExe = findCargoExecutable()
configured = getenv('CARGO');
if ~isempty(configured) && isfile(configured)
    cargoExe = string(configured);
    return;
end

if ispc
    homeDir = getenv('USERPROFILE');
    candidates = {
        fullfile(homeDir, '.cargo', 'bin', 'cargo.exe')
        fullfile(homeDir, '.cargo', 'bin', 'cargo')
    };
else
    homeDir = getenv('HOME');
    candidates = {
        fullfile(homeDir, '.cargo', 'bin', 'cargo')
        fullfile(homeDir, '.rustup', 'toolchains', 'stable-aarch64-apple-darwin', 'bin', 'cargo')
        '/opt/homebrew/bin/cargo'
        '/usr/local/bin/cargo'
    };
end

for idx = 1:numel(candidates)
    if isfile(candidates{idx})
        cargoExe = string(candidates{idx});
        return;
    end
end

if ispc
    [status, resolved] = system('where cargo');
else
    [status, resolved] = system('command -v cargo');
end
if status == 0
    resolvedLines = strsplit(strtrim(resolved), newline);
    if ~isempty(resolvedLines) && ~isempty(resolvedLines{1})
        cargoExe = string(strtrim(resolvedLines{1}));
        return;
    end
end

error('build_mex:missingCargo', ...
    ['Could not locate cargo. Set the CARGO environment variable or install Rust so ', ...
     'cargo is available at ~/.cargo/bin/cargo or a standard system path.']);
end
