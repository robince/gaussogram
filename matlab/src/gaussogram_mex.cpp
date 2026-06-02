// MEX dispatch shim over the gaussogram C ABI (gaussogram-ffi).
//
// Called as gaussogram_mex(op, ...) where op is a string selecting the
// operation. The thin .m wrappers (gaussogram.m, gaussogram_to_grid.m, ...) map
// scheme/window names to integer ids and call here. Complex coefficients use
// MATLAB's interleaved-complex API (-R2018a), which matches the FFI's
// interleaved (re,im) double layout exactly, so no repacking is needed.

#include "mex.h"
#include "matrix.h"

extern "C" {
#include "gaussogram.h"
}

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* id, const char* message) {
    if (!condition) {
        mexErrMsgIdAndTxt(id, "%s", message);
    }
}

const char* status_message(int32_t status) {
    switch (status) {
        case GAUSSOGRAM_STATUS_POWER_OF_TWO:
            return "N must be a power of two";
        case GAUSSOGRAM_STATUS_TOO_SMALL:
            return "N is too small for this scheme";
        case GAUSSOGRAM_STATUS_UNKNOWN_WINDOW:
            return "unknown window type";
        case GAUSSOGRAM_STATUS_NOT_INVERTIBLE:
            return "scheme is not invertible";
        case GAUSSOGRAM_STATUS_WRONG_INPUT_KIND:
            return "wrong input kind for this scheme";
        case GAUSSOGRAM_STATUS_OUTPUT_LEN_MISMATCH:
            return "output length mismatch";
        case GAUSSOGRAM_STATUS_INPUT_LEN_MISMATCH:
            return "input length mismatch";
        case GAUSSOGRAM_STATUS_INTERNAL:
            return "internal error";
        default:
            return "unknown error";
    }
}

std::string get_op(const mxArray* arg) {
    require(mxIsChar(arg), "gaussogram:op", "First argument must be an operation string.");
    char buf[64];
    if (mxGetString(arg, buf, sizeof(buf)) != 0) {
        mexErrMsgIdAndTxt("gaussogram:op", "Operation string too long.");
    }
    return std::string(buf);
}

int32_t scalar_int(const mxArray* arg, const char* id, const char* name) {
    if (!mxIsScalar(arg) || mxIsComplex(arg) || !mxIsNumeric(arg)) {
        mexErrMsgIdAndTxt(id, "%s must be a real numeric scalar.", name);
    }
    return static_cast<int32_t>(mxGetScalar(arg));
}

size_t scalar_size(const mxArray* arg, const char* id, const char* name) {
    if (!mxIsScalar(arg) || mxIsComplex(arg) || !mxIsNumeric(arg)) {
        mexErrMsgIdAndTxt(id, "%s must be a real numeric scalar.", name);
    }
    double v = mxGetScalar(arg);
    if (v < 0.0) {
        mexErrMsgIdAndTxt(id, "%s must be nonnegative.", name);
    }
    return static_cast<size_t>(v);
}

void require_real_double(const mxArray* arg, const char* id, const char* name) {
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxIsSparse(arg)) {
        mexErrMsgIdAndTxt(id, "%s must be a real full double array.", name);
    }
}

// RAII handle wrapper so a thrown mexErr (longjmp) still frees the engine —
// actually mexErrMsgIdAndTxt longjmps and skips destructors, so we only call
// mexErr after destroying the handle. This struct just centralises cleanup.
struct Handle {
    GaussogramHandle* ptr = nullptr;
    ~Handle() {
        if (ptr) gaussogram_destroy(ptr);
    }
};

GaussogramHandle* make_handle(int32_t scheme_id, size_t n, int32_t window_id, int32_t nyquist) {
    GaussogramHandle* h = nullptr;
    int32_t st = gaussogram_create(scheme_id, n, window_id, nyquist, &h);
    if (st != GAUSSOGRAM_STATUS_OK || h == nullptr) {
        mexErrMsgIdAndTxt(
            "gaussogram:create",
            "gaussogram_create failed with status %d (%s).",
            st, status_message(st));
    }
    return h;
}

// op = 'forward': forward transform of a real signal matrix (n x count;
// each column is a segment). Returns complex (output_len x count).
void op_forward(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 5, "gaussogram:nrhs",
            "forward expects (op, signal, scheme_id, window_id, nyquist_flat_top).");
    const mxArray* sig_in = prhs[1];
    require_real_double(sig_in, "gaussogram:sigType", "signal");

    size_t rows = static_cast<size_t>(mxGetM(sig_in));
    size_t cols = static_cast<size_t>(mxGetN(sig_in));
    size_t n, count;
    if (rows == 1 || cols == 1) {  // vector -> single segment
        n = rows * cols;
        count = 1;
    } else {
        n = rows;
        count = cols;
    }
    require(n > 0, "gaussogram:empty", "signal must be non-empty.");

    int32_t scheme_id = scalar_int(prhs[2], "gaussogram:scheme", "scheme_id");
    int32_t window_id = scalar_int(prhs[3], "gaussogram:window", "window_id");
    int32_t nyquist = scalar_int(prhs[4], "gaussogram:nyquist", "nyquist_flat_top");

    Handle h;
    h.ptr = make_handle(scheme_id, n, window_id, nyquist);
    size_t olen = gaussogram_output_len(h.ptr);

    plhs[0] = mxCreateNumericMatrix(olen, count, mxDOUBLE_CLASS, mxCOMPLEX);
    double* out = reinterpret_cast<double*>(mxGetComplexDoubles(plhs[0]));
    const double* sig = mxGetDoubles(sig_in);

    int32_t st;
    if (count == 1) {
        st = gaussogram_forward_real(h.ptr, sig, n, out, olen);
    } else {
        st = gaussogram_forward_real_batch(h.ptr, sig, n, count, out, olen * count);
    }
    if (st != GAUSSOGRAM_STATUS_OK) {
        mexErrMsgIdAndTxt("gaussogram:forward",
                          "forward failed with status %d (%s).", st, status_message(st));
    }
    (void)nlhs;
}

// op = 'output_len': (op, n, scheme_id, window_id, nyquist) -> scalar double.
void op_output_len(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 5, "gaussogram:nrhs",
            "output_len expects (op, n, scheme_id, window_id, nyquist_flat_top).");
    size_t n = scalar_size(prhs[1], "gaussogram:n", "n");
    int32_t scheme_id = scalar_int(prhs[2], "gaussogram:scheme", "scheme_id");
    int32_t window_id = scalar_int(prhs[3], "gaussogram:window", "window_id");
    int32_t nyquist = scalar_int(prhs[4], "gaussogram:nyquist", "nyquist_flat_top");

    Handle h;
    h.ptr = make_handle(scheme_id, n, window_id, nyquist);
    plhs[0] = mxCreateDoubleScalar(static_cast<double>(gaussogram_output_len(h.ptr)));
    (void)nlhs;
}

// op = 'scheme_bands': (op, n, scheme_id, window_id, nyquist)
// -> double matrix (num_bands x 4): columns [src_lo, width, fcentre, out_off].
void op_scheme_bands(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 5, "gaussogram:nrhs",
            "scheme_bands expects (op, n, scheme_id, window_id, nyquist_flat_top).");
    size_t n = scalar_size(prhs[1], "gaussogram:n", "n");
    int32_t scheme_id = scalar_int(prhs[2], "gaussogram:scheme", "scheme_id");
    int32_t window_id = scalar_int(prhs[3], "gaussogram:window", "window_id");
    int32_t nyquist = scalar_int(prhs[4], "gaussogram:nyquist", "nyquist_flat_top");

    Handle h;
    h.ptr = make_handle(scheme_id, n, window_id, nyquist);
    int32_t nb = gaussogram_num_bands(h.ptr);
    require(nb > 0, "gaussogram:bands", "scheme has no bands.");

    std::vector<int64_t> lo(nb), width(nb), fcentre(nb), off(nb);
    int32_t got = gaussogram_scheme_bands(
        h.ptr, lo.data(), width.data(), fcentre.data(), off.data(),
        static_cast<size_t>(nb));
    require(got == nb, "gaussogram:bands", "scheme_bands returned an unexpected count.");

    plhs[0] = mxCreateDoubleMatrix(static_cast<mwSize>(nb), 4, mxREAL);
    double* m = mxGetDoubles(plhs[0]);
    for (int32_t i = 0; i < nb; ++i) {
        m[i] = static_cast<double>(lo[i]);              // column 0
        m[nb + i] = static_cast<double>(width[i]);      // column 1
        m[2 * nb + i] = static_cast<double>(fcentre[i]);// column 2
        m[3 * nb + i] = static_cast<double>(off[i]);    // column 3
    }
    (void)nlhs;
}

// op = 'inverse_real': (op, coeffs (complex vec, length N/2+1), window_id)
// -> real column vector (length N). Implicitly uses the dyadic_real scheme.
void op_inverse_real(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 3, "gaussogram:nrhs", "inverse_real expects (op, coeffs, window_id).");
    const mxArray* c_in = prhs[1];
    require(mxIsDouble(c_in) && !mxIsSparse(c_in), "gaussogram:coeffsType",
            "coeffs must be a full double array.");
    size_t clen = static_cast<size_t>(mxGetNumberOfElements(c_in));
    require(clen >= 2, "gaussogram:coeffsLen", "coeffs must have length N/2+1 >= 2.");
    int32_t window_id = scalar_int(prhs[2], "gaussogram:window", "window_id");

    size_t n = (clen - 1) * 2;

    // Get an interleaved complex pointer regardless of whether MATLAB stored the
    // array as real (imag==0) or complex.
    std::vector<double> packed;
    const double* coeffs;
    if (mxIsComplex(c_in)) {
        coeffs = reinterpret_cast<const double*>(mxGetComplexDoubles(c_in));
    } else {
        const double* re = mxGetDoubles(c_in);
        packed.resize(2 * clen, 0.0);
        for (size_t i = 0; i < clen; ++i) packed[2 * i] = re[i];
        coeffs = packed.data();
    }

    Handle h;
    h.ptr = make_handle(GAUSSOGRAM_SCHEME_DYADIC_REAL, n, window_id, 0);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* out = mxGetDoubles(plhs[0]);
    int32_t st = gaussogram_inverse_real(h.ptr, coeffs, clen, out, n);
    if (st != GAUSSOGRAM_STATUS_OK) {
        mexErrMsgIdAndTxt("gaussogram:inverse",
                          "inverse_real failed with status %d (%s).", st, status_message(st));
    }
    (void)nlhs;
}

// op = 'partitions': (op, n, kind) kind 0=real, 1=complex -> int32 column vector.
void op_partitions(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 3, "gaussogram:nrhs", "partitions expects (op, n, kind).");
    size_t n = scalar_size(prhs[1], "gaussogram:n", "n");
    int32_t kind = scalar_int(prhs[2], "gaussogram:kind", "kind");

    std::vector<int32_t> buf(64);
    int32_t count;
    if (kind == 0) {
        count = gaussogram_real_partitions(n, buf.data(), buf.size());
    } else {
        count = gaussogram_complex_partitions(n, buf.data(), buf.size());
    }
    require(count >= 0, "gaussogram:partitions", "partitions failed (invalid N?).");

    plhs[0] = mxCreateNumericMatrix(count, 1, mxINT32_CLASS, mxREAL);
    int32_t* out = static_cast<int32_t*>(mxGetData(plhs[0]));
    std::memcpy(out, buf.data(), static_cast<size_t>(count) * sizeof(int32_t));
    (void)nlhs;
}

}  // namespace

extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs >= 1, "gaussogram:nrhs", "Expected at least an operation string.");
    std::string op = get_op(prhs[0]);

    if (op == "forward") {
        op_forward(nlhs, plhs, nrhs, prhs);
    } else if (op == "output_len") {
        op_output_len(nlhs, plhs, nrhs, prhs);
    } else if (op == "scheme_bands") {
        op_scheme_bands(nlhs, plhs, nrhs, prhs);
    } else if (op == "inverse_real") {
        op_inverse_real(nlhs, plhs, nrhs, prhs);
    } else if (op == "partitions") {
        op_partitions(nlhs, plhs, nrhs, prhs);
    } else {
        mexErrMsgIdAndTxt("gaussogram:op", "Unknown operation '%s'.", op.c_str());
    }
}
