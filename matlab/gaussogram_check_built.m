function gaussogram_check_built()
%GAUSSOGRAM_CHECK_BUILT  Error if the MEX has not been compiled yet.
if exist('gaussogram_mex', 'file') ~= 3
    error('gaussogram:notBuilt', ...
        'gaussogram_mex is not built. Run build_mex first.');
end
end
