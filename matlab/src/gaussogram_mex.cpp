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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <limits>
#include <unordered_map>
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
        case GAUSSOGRAM_STATUS_BANDS_PER_OCTAVE:
            return "bands_per_octave must be a power of two >= 1";
        case GAUSSOGRAM_STATUS_UNKNOWN_SCHEME:
            return "unknown scheme id";
        case GAUSSOGRAM_STATUS_UNSUPPORTED_OPTION:
            return "option not supported by this scheme (nyquist_flat_top needs dyadic_dual_real)";
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
    // Also rejects NaN and Inf; the upper bound keeps the cast defined.
    if (!(v >= 0.0) || !std::isfinite(v) || v != std::floor(v) ||
        v >= static_cast<double>(std::numeric_limits<size_t>::max())) {
        mexErrMsgIdAndTxt(id, "%s must be a nonnegative integer.", name);
    }
    return static_cast<size_t>(v);
}

void require_real_double(const mxArray* arg, const char* id, const char* name) {
    if (!mxIsDouble(arg) || mxIsComplex(arg) || mxIsSparse(arg)) {
        mexErrMsgIdAndTxt(id, "%s must be a real full double array.", name);
    }
}

// Handle wrapper centralising cleanup. mexErrMsgIdAndTxt may longjmp and skip
// destructors (C Matrix API), so every error path with a live owned handle
// must call reset() explicitly BEFORE raising; the destructor only covers the
// success path and MATLAB versions where the error unwinds as a C++ exception.
struct Handle {
    GaussogramHandle* ptr = nullptr;
    void reset() {
        if (ptr) {
            gaussogram_destroy(ptr);
            ptr = nullptr;
        }
    }
    ~Handle() {
        reset();
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

// Interpret a real double signal matrix as either a single segment (vector) or
// a batch (n x count matrix), writing the inferred dimensions to *n / *count.
void infer_segments(const mxArray* sig_in, size_t* n, size_t* count) {
    size_t rows = static_cast<size_t>(mxGetM(sig_in));
    size_t cols = static_cast<size_t>(mxGetN(sig_in));
    if (rows == 1 || cols == 1) {  // vector -> single segment
        *n = rows * cols;
        *count = 1;
    } else {
        *n = rows;
        *count = cols;
    }
    require(*n > 0, "gaussogram:empty", "signal must be non-empty.");
}

// Run a forward transform through an existing engine into a freshly allocated
// complex output (output_len x count); errors via mexErr on a non-OK status.
// `owned` (nullable) is destroyed before raising so a longjmp cannot leak it;
// pass nullptr for persistent handles that must outlive the error.
mxArray* forward_into_output(GaussogramHandle* h, const mxArray* sig_in, Handle* owned) {
    require_real_double(sig_in, "gaussogram:sigType", "signal");
    size_t n, count;
    infer_segments(sig_in, &n, &count);
    size_t olen = gaussogram_output_len(h);

    // Uninitialised: the transform overwrites every element (no zero-fill pass).
    mxArray* result = mxCreateUninitNumericMatrix(olen, count, mxDOUBLE_CLASS, mxCOMPLEX);
    double* out = reinterpret_cast<double*>(mxGetComplexDoubles(result));
    const double* sig = mxGetDoubles(sig_in);

    int32_t st;
    if (count == 1) {
        st = gaussogram_forward_real(h, sig, n, out, olen);
    } else {
        st = gaussogram_forward_real_batch(h, sig, n, count, out, olen * count);
    }
    if (st != GAUSSOGRAM_STATUS_OK) {
        mxDestroyArray(result);
        if (owned) owned->reset();
        mexErrMsgIdAndTxt("gaussogram:forward",
                          "forward failed with status %d (%s).", st, status_message(st));
    }
    return result;
}

// Engines created by op 'create' and not yet destroyed, keyed by handle id.
// MATLAB holds handles as plain uint64 values, which can outlive their engine
// (a copied value after 'destroy') or come from elsewhere entirely. Ids are
// never reused (unlike engine addresses, which the allocator may hand out
// again), so a stale id can never resolve to, or destroy, a later engine.
std::unordered_map<uint64_t, GaussogramHandle*>& live_handles() {
    static std::unordered_map<uint64_t, GaussogramHandle*> handles;
    return handles;
}

uint64_t next_handle_id() {
    static uint64_t next = 0;
    return ++next;  // 0 is never issued; Gaussogram1d uses it for "no engine"
}

uint64_t handle_id_from_arg(const mxArray* arg) {
    require(mxIsUint64(arg) && mxIsScalar(arg) && !mxIsComplex(arg), "gaussogram:handle",
            "handle must be a uint64 scalar.");
    return *static_cast<const uint64_t*>(mxGetData(arg));
}

// Resolve a persistent engine handle passed back from MATLAB.
GaussogramHandle* read_handle(const mxArray* arg) {
    auto it = live_handles().find(handle_id_from_arg(arg));
    require(it != live_handles().end(), "gaussogram:handle",
            "handle is not a live gaussogram engine (destroyed, or loaded from a file?).");
    return it->second;
}

// op = 'forward': forward transform of a real signal matrix (n x count;
// each column is a segment). Returns complex (output_len x count).
void op_forward(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 5, "gaussogram:nrhs",
            "forward expects (op, signal, scheme_id, window_id, nyquist_flat_top).");
    const mxArray* sig_in = prhs[1];
    require_real_double(sig_in, "gaussogram:sigType", "signal");
    size_t n, count;
    infer_segments(sig_in, &n, &count);

    int32_t scheme_id = scalar_int(prhs[2], "gaussogram:scheme", "scheme_id");
    int32_t window_id = scalar_int(prhs[3], "gaussogram:window", "window_id");
    int32_t nyquist = scalar_int(prhs[4], "gaussogram:nyquist", "nyquist_flat_top");

    Handle h;
    h.ptr = make_handle(scheme_id, n, window_id, nyquist);
    plhs[0] = forward_into_output(h.ptr, sig_in, &h);
    (void)nlhs;
}

// op = 'create': build a persistent engine and return its id as a uint64
// scalar (plhs[0]); optionally its output_len (plhs[1]). mexLock keeps the MEX
// resident so live handles are not orphaned by `clear`. Pair with 'destroy'.
void op_create(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 5, "gaussogram:nrhs",
            "create expects (op, n, scheme_id, window_id, nyquist_flat_top).");
    size_t n = scalar_size(prhs[1], "gaussogram:n", "n");
    int32_t scheme_id = scalar_int(prhs[2], "gaussogram:scheme", "scheme_id");
    int32_t window_id = scalar_int(prhs[3], "gaussogram:window", "window_id");
    int32_t nyquist = scalar_int(prhs[4], "gaussogram:nyquist", "nyquist_flat_top");

    GaussogramHandle* h = make_handle(scheme_id, n, window_id, nyquist);
    uint64_t id = next_handle_id();
    live_handles().emplace(id, h);
    mexLock();

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(plhs[0])) = id;
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(static_cast<double>(gaussogram_output_len(h)));
    }
}

// op = 'forward_handle': forward transform through a persistent engine.
// (op, handle (uint64), signal) -> complex (output_len x count).
void op_forward_handle(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 3, "gaussogram:nrhs", "forward_handle expects (op, handle, signal).");
    GaussogramHandle* h = read_handle(prhs[1]);
    plhs[0] = forward_into_output(h, prhs[2], nullptr);  // persistent handle: keep alive
    (void)nlhs;
}

// op = 'destroy': free a persistent engine and release one mexLock.
void op_destroy(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 2, "gaussogram:nrhs", "destroy expects (op, handle).");
    // Unknown or already-destroyed handles are a no-op, never a double free.
    auto it = live_handles().find(handle_id_from_arg(prhs[1]));
    if (it != live_handles().end()) {
        GaussogramHandle* h = it->second;
        live_handles().erase(it);
        gaussogram_destroy(h);
        if (mexIsLocked()) {
            mexUnlock();
        }
    }
    (void)nlhs;
    (void)plhs;
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
    if (nb <= 0) {
        h.reset();
        mexErrMsgIdAndTxt("gaussogram:bands", "scheme has no bands.");
    }

    std::vector<int64_t> lo(nb), width(nb), fcentre(nb), off(nb);
    int32_t got = gaussogram_scheme_bands(
        h.ptr, lo.data(), width.data(), fcentre.data(), off.data(),
        static_cast<size_t>(nb));
    if (got != nb) {
        h.reset();
        mexErrMsgIdAndTxt("gaussogram:bands", "scheme_bands returned an unexpected count.");
    }

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
    // mxGetN is the product of all trailing dimensions, so reject N-D arrays
    // explicitly (a 1 x 3 x 11 array would otherwise pass as a row).
    require(mxGetNumberOfDimensions(c_in) == 2 &&
            (mxGetM(c_in) == 1 || mxGetN(c_in) == 1), "gaussogram:coeffsShape",
            "coeffs must be a vector.");
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

    plhs[0] = mxCreateUninitNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
    double* out = mxGetDoubles(plhs[0]);
    int32_t st = gaussogram_inverse_real(h.ptr, coeffs, clen, out, n);
    if (st != GAUSSOGRAM_STATUS_OK) {
        mxDestroyArray(plhs[0]);
        h.reset();
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
    if (count < 0) {
        mexErrMsgIdAndTxt("gaussogram:partitions", "partitions failed with status %d (%s).",
                          -count, status_message(-count));
    }

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
    } else if (op == "create") {
        op_create(nlhs, plhs, nrhs, prhs);
    } else if (op == "forward_handle") {
        op_forward_handle(nlhs, plhs, nrhs, prhs);
    } else if (op == "destroy") {
        op_destroy(nlhs, plhs, nrhs, prhs);
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
