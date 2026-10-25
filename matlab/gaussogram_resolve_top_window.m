function [name, id] = gaussogram_resolve_top_window(top_window, nyquist_flat_top)
%GAUSSOGRAM_RESOLVE_TOP_WINDOW  Resolve 'top_window' and its deprecated alias.
%   [name, id] = GAUSSOGRAM_RESOLVE_TOP_WINDOW(top_window, nyquist_flat_top)
%   top_window is 'wide', 'gaussian', 'flat', or '' (unset, meaning 'wide').
%   nyquist_flat_top is [] (unset) or a logical; true is a deprecated alias
%   for 'flat'. Returns the name and the C ABI id (0 = wide, 1 = flat,
%   2 = gaussian).
name = char(lower(string(top_window)));
if ~isempty(nyquist_flat_top)
    % false means "no flat top" and so the default, as before; but the
    % default top window changed, so say how to get the old output.
    if logical(nyquist_flat_top)
        warning('gaussogram:deprecated', ...
            "'nyquist_flat_top' is deprecated; use 'top_window', 'flat'.");
    else
        warning('gaussogram:deprecated', ...
            "'nyquist_flat_top' is deprecated. 'nyquist_flat_top', false now gives " + ...
            "the default 'top_window', 'wide'; pass 'top_window', 'gaussian' " + ...
            "for the previous output.");
    end
    if logical(nyquist_flat_top)
        if ~isempty(name) && ~strcmp(name, 'flat')
            error('gaussogram:top_window', ...
                "'nyquist_flat_top', true conflicts with 'top_window', '%s'.", name);
        end
        name = 'flat';
    elseif strcmp(name, 'flat')
        error('gaussogram:top_window', ...
            "'nyquist_flat_top', false conflicts with 'top_window', 'flat'.");
    end
end
if isempty(name)
    name = 'wide';
end
switch name
    case 'wide'
        id = 0;
    case 'flat'
        id = 1;
    case 'gaussian'
        id = 2;
    otherwise
        error('gaussogram:top_window', ...
            "top_window must be 'wide', 'gaussian', or 'flat'.");
end
end
