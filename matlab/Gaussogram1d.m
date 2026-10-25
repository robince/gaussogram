classdef Gaussogram1d < handle
%GAUSSOGRAM1D  Reusable forward GFT engine (builds FFT plans once).
%   eng = GAUSSOGRAM1D(n)
%   eng = GAUSSOGRAM1D(n, 'scheme', s, 'window', w, 'top_window', t)
%   coeffs = eng.forward(signal)
%
%   Like the function GAUSSOGRAM, but the engine (and its FFT plans) is built
%   once at construction and reused across calls — prefer this in tight loops or
%   when transforming many length-N segments. Mirrors the Python
%   gaussogram.Gaussogram1d class.
%
%   The engine is fixed to length N and the scheme/window chosen at
%   construction. forward() accepts a length-N column/row vector (one segment)
%   or an N x count matrix (a batch, transformed in parallel on one thread per
%   CPU; inside parfor workers, setenv('RAYON_NUM_THREADS', '1') before the
%   first transform to avoid oversubscription). The handle is
%   freed automatically when the object is cleared or goes out of scope; you may
%   also call delete(eng) explicitly.
%
%   Example:
%     eng = Gaussogram1d(1024);
%     for k = 1:100
%         c = eng.forward(signals(:, k));   % no replanning per call
%     end
%
%   See also GAUSSOGRAM, GAUSSOGRAM_OUTPUT_LEN, GAUSSOGRAM_TO_GRID.

    properties (SetAccess = private)
        n                  % transform length (power of two)
        output_len         % packed complex-coefficient count
        scheme             % scheme name (char)
        window             % window name (char)
        top_window         % top-band window of dyadic_dual_real (char)
    end

    properties (Dependent)
        nyquist_flat_top   % deprecated: true when top_window is 'flat'
    end

    properties (Access = private, Transient)
        % Engine id issued by the MEX (0 = none). Transient: an id is meaningless
        % in another process or session, so saving/loading the object (or
        % sending it to parfor workers) restores 0 and the engine is rebuilt
        % on first use instead of reusing an id from another session.
        ptr = uint64(0)
    end

    methods (Access = private)
        function olen = create_engine(obj)
            if strcmpi(obj.scheme, 'dyadic_complex')
                error('gaussogram:scheme', ...
                    "'dyadic_complex' (complex input) is not available from MATLAB yet.");
            end
            [scheme_id, window_id] = gaussogram_ids(obj.scheme, obj.window);
            gaussogram_check_built();
            [~, top_id] = gaussogram_resolve_top_window(obj.top_window, []);
            [h, olen] = gaussogram_mex('create', obj.n, scheme_id, window_id, top_id);
            obj.ptr = h;
        end
    end

    methods
        function obj = Gaussogram1d(n, varargin)
            opts = gaussogram_parse_opts(varargin{:});
            obj.n = double(n);
            obj.scheme = opts.scheme;
            obj.window = opts.window;
            obj.top_window = opts.top_window;
            obj.output_len = obj.create_engine();
        end

        function tf = get.nyquist_flat_top(obj)
            tf = strcmp(obj.top_window, 'flat');
        end

        function coeffs = forward(obj, signal)
        %FORWARD  Transform a length-N segment (or N x count batch).
            if obj.ptr == 0
                % Only reachable after load / parfor transfer (Transient ptr):
                % a deleted handle object cannot have its methods called.
                obj.create_engine();
            end
            if ~isa(signal, 'double') || ~isreal(signal) || issparse(signal)
                error('gaussogram:signal', 'signal must be a real full double array.');
            end
            coeffs = gaussogram_mex('forward_handle', obj.ptr, signal);
        end

        function delete(obj)
        %DELETE  Free the underlying engine handle.
            if obj.ptr ~= 0
                try
                    gaussogram_mex('destroy', obj.ptr);
                catch
                    % MEX may be unloaded during interpreter teardown; ignore.
                end
                obj.ptr = uint64(0);
            end
        end
    end
end
