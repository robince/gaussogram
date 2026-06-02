classdef Gaussogram1d < handle
%GAUSSOGRAM1D  Reusable forward GFT engine (builds FFT plans once).
%   eng = GAUSSOGRAM1D(n)
%   eng = GAUSSOGRAM1D(n, 'scheme', s, 'window', w, 'nyquist_flat_top', tf)
%   coeffs = eng.forward(signal)
%
%   Like the function GAUSSOGRAM, but the engine (and its FFT plans) is built
%   once at construction and reused across calls — prefer this in tight loops or
%   when transforming many length-N segments. Mirrors the Python
%   gaussogram.Gaussogram1d class.
%
%   The engine is fixed to length N and the scheme/window chosen at
%   construction. forward() accepts a length-N column/row vector (one segment)
%   or an N x count matrix (a batch, transformed in parallel). The handle is
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
        nyquist_flat_top   % logical
    end

    properties (Access = private)
        ptr = uint64(0)    % opaque GaussogramHandle* (0 once destroyed)
    end

    methods
        function obj = Gaussogram1d(n, varargin)
            opts = gaussogram_parse_opts(varargin{:});
            [scheme_id, window_id] = gaussogram_ids(opts.scheme, opts.window);
            gaussogram_check_built();
            [h, olen] = gaussogram_mex('create', double(n), scheme_id, window_id, ...
                double(opts.nyquist_flat_top));
            obj.ptr = h;
            obj.n = double(n);
            obj.output_len = olen;
            obj.scheme = opts.scheme;
            obj.window = opts.window;
            obj.nyquist_flat_top = opts.nyquist_flat_top;
        end

        function coeffs = forward(obj, signal)
        %FORWARD  Transform a length-N segment (or N x count batch).
            if obj.ptr == 0
                error('gaussogram:destroyed', 'This Gaussogram1d engine has been destroyed.');
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
