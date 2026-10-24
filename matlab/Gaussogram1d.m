classdef Gaussogram1d < handle
%GAUSSOGRAM1D  Reusable GFT engine (builds FFT plans once).
%   eng = GAUSSOGRAM1D(n)
%   eng = GAUSSOGRAM1D(n, 'scheme', s, 'window', w, 'nyquist_flat_top', tf, ...
%                      'bands_per_octave', b)
%   coeffs = eng.forward(signal)
%   signal = eng.inverse(coeffs)          % real-scheme engines
%
%   Like the function GAUSSOGRAM, but the engine (and its FFT plans) is built
%   once at construction and reused across calls — prefer this in tight loops or
%   when transforming many length-N segments. Mirrors the Python
%   gaussogram.Gaussogram1d class (forward_batch is forward on a matrix).
%
%   The engine is fixed to length N and the scheme/window/bands_per_octave
%   chosen at construction. forward() accepts a length-N column/row vector (one
%   segment) or an N x count matrix (a batch; real batches are transformed in
%   parallel on one thread per CPU; inside parfor workers,
%   setenv('RAYON_NUM_THREADS', '1') before the first transform to avoid
%   oversubscription). Signals are real for the real schemes and real or
%   complex for 'dyadic_complex'. inverse() reconstructs the length-N real
%   signal from a coefficient vector: exact for 'dyadic_real', the
%   least-squares frame inverse for 'dyadic_dual_real' (see
%   GAUSSOGRAM_INVERSE_REAL); 'dyadic_complex' engines are not invertible
%   (see the invertible property). The handle is freed automatically
%   when the object is cleared or goes out of scope; you may also call
%   delete(eng) explicitly.
%
%   Example:
%     eng = Gaussogram1d(1024);
%     for k = 1:100
%         c = eng.forward(signals(:, k));   % no replanning per call
%     end
%
%   See also GAUSSOGRAM, GAUSSOGRAM_INVERSE_REAL, GAUSSOGRAM_OUTPUT_LEN,
%   GAUSSOGRAM_TO_GRID.

    properties (SetAccess = private)
        n                  % transform length (power of two)
        output_len         % packed complex-coefficient count
        scheme             % scheme name (char)
        window             % window name (char)
        nyquist_flat_top   % logical
        bands_per_octave   % sub-bands per dyadic octave (power of two)
        invertible         % logical: inverse() is available (real schemes)
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
            [scheme_id, window_id] = gaussogram_ids(obj.scheme, obj.window);
            gaussogram_check_built();
            [h, olen] = gaussogram_mex('create', obj.n, scheme_id, window_id, ...
                double(obj.nyquist_flat_top), obj.bands_per_octave);
            obj.ptr = h;
        end

        function ensure_engine(obj)
            if obj.ptr == 0
                % Only reachable after load / parfor transfer (Transient ptr):
                % a deleted handle object cannot have its methods called.
                obj.create_engine();
            end
        end
    end

    methods
        function obj = Gaussogram1d(n, varargin)
            opts = gaussogram_parse_opts(varargin{:});
            obj.n = double(n);
            obj.scheme = lower(opts.scheme);
            obj.window = lower(opts.window);
            obj.nyquist_flat_top = opts.nyquist_flat_top;
            obj.bands_per_octave = opts.bands_per_octave;
            obj.invertible = ~strcmp(obj.scheme, 'dyadic_complex');
            obj.output_len = obj.create_engine();
        end

        function coeffs = forward(obj, signal)
        %FORWARD  Transform a length-N segment (or N x count batch).
            obj.ensure_engine();
            if ~isa(signal, 'double') || issparse(signal)
                error('gaussogram:signal', 'signal must be a full double array.');
            end
            if strcmp(obj.scheme, 'dyadic_complex')
                signal = complex(signal);   % the complex scheme takes complex input
            elseif ~isreal(signal)
                error('gaussogram:signal', ...
                    'signal must be real for the real schemes.');
            end
            coeffs = gaussogram_mex('forward_handle', obj.ptr, signal);
        end

        function signal = inverse(obj, coeffs)
        %INVERSE  Reconstruct the length-N real signal from coefficients.
        %   coeffs must be a vector of output_len coefficients from this
        %   engine's scheme and options. Returns an N x 1 column vector.
            if ~obj.invertible
                error('gaussogram:notInvertible', ...
                    "'%s' is not invertible.", ...
                    obj.scheme);
            end
            obj.ensure_engine();
            if ~isa(coeffs, 'double') || issparse(coeffs)
                error('gaussogram:coeffs', 'coeffs must be a full double array.');
            end
            signal = gaussogram_mex('inverse_handle', obj.ptr, coeffs, obj.n);
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
