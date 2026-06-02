function [scheme_id, window_id] = gaussogram_ids(scheme, window)
%GAUSSOGRAM_IDS  Map scheme/window name strings to the C ABI integer ids.
switch lower(string(scheme))
    case "dyadic_dual_real"
        scheme_id = 0;
    case "dyadic_real"
        scheme_id = 1;
    case "dyadic_complex"
        scheme_id = 2;
    otherwise
        error('gaussogram:scheme', ...
            "scheme must be 'dyadic_dual_real', 'dyadic_real', or 'dyadic_complex'.");
end

switch lower(string(window))
    case "gaussian"
        window_id = 0;
    case "box"
        window_id = 1;
    otherwise
        error('gaussogram:window', "window must be 'gaussian' or 'box'.");
end
end
