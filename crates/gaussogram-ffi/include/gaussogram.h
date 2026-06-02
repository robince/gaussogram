#ifndef GAUSSOGRAM_H
#define GAUSSOGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes — must match gaussogram-core/src/error.rs. */
#define GAUSSOGRAM_STATUS_OK 0
#define GAUSSOGRAM_STATUS_POWER_OF_TWO 1
#define GAUSSOGRAM_STATUS_TOO_SMALL 2
#define GAUSSOGRAM_STATUS_UNKNOWN_WINDOW 3
#define GAUSSOGRAM_STATUS_NOT_INVERTIBLE 4
#define GAUSSOGRAM_STATUS_WRONG_INPUT_KIND 5
#define GAUSSOGRAM_STATUS_OUTPUT_LEN_MISMATCH 6
#define GAUSSOGRAM_STATUS_INPUT_LEN_MISMATCH 7
#define GAUSSOGRAM_STATUS_INTERNAL 99

/* scheme_id values for gaussogram_create. */
#define GAUSSOGRAM_SCHEME_DYADIC_DUAL_REAL 0
#define GAUSSOGRAM_SCHEME_DYADIC_REAL 1
#define GAUSSOGRAM_SCHEME_DYADIC_COMPLEX 2

/* window_id values for gaussogram_create. */
#define GAUSSOGRAM_WINDOW_GAUSSIAN 0
#define GAUSSOGRAM_WINDOW_BOX 1

/* Opaque engine handle: build once (plans precomputed), transform many. */
typedef struct GaussogramHandle GaussogramHandle;

/* Create an engine. On success writes the handle to *out_handle and returns
 * GAUSSOGRAM_STATUS_OK; otherwise returns a non-zero status code. */
int32_t gaussogram_create(
    int32_t scheme_id,
    size_t n,
    int32_t window_id,
    int32_t nyquist_flat_top,
    GaussogramHandle** out_handle
);

/* Destroy a handle created by gaussogram_create. */
void gaussogram_destroy(GaussogramHandle* handle);

/* Packed output length (number of complex coefficients). */
size_t gaussogram_output_len(const GaussogramHandle* handle);

/* Forward transform of one real segment. signal has n doubles; out has
 * 2 * out_len_complex doubles (interleaved re,im). */
int32_t gaussogram_forward_real(
    const GaussogramHandle* handle,
    const double* signal,
    size_t signal_len,
    double* out,
    size_t out_len_complex
);

/* Batch forward over count real segments laid out contiguously
 * (count * n doubles in; count * 2 * output_len doubles out). Segment i
 * occupies in[i*n .. (i+1)*n) and out[i*output_len .. ) (interleaved complex).
 * Parallelised across segments. */
int32_t gaussogram_forward_real_batch(
    const GaussogramHandle* handle,
    const double* signals,
    size_t n,
    size_t count,
    double* out,
    size_t out_len_complex
);

/* Inverse transform for invertible real schemes (currently dyadic_real).
 * coeffs has 2 * coeffs_len_complex doubles (interleaved); out has out_len
 * doubles (reconstructed signal, length N). */
int32_t gaussogram_inverse_real(
    const GaussogramHandle* handle,
    const double* coeffs,
    size_t coeffs_len_complex,
    double* out,
    size_t out_len
);

/* Number of bands in the engine's scheme, or -1 on a null handle. */
int32_t gaussogram_num_bands(const GaussogramHandle* handle);

/* Per-band layout into four i64 buffers of capacity cap. Returns the number of
 * bands written, or -1 on error. */
int32_t gaussogram_scheme_bands(
    const GaussogramHandle* handle,
    int64_t* out_lo,
    int64_t* out_width,
    int64_t* out_fcentre,
    int64_t* out_off,
    size_t cap
);

/* Legacy partition boundaries. Writes into out (capacity out_cap int32s) and
 * returns the count written, or -1. */
int32_t gaussogram_real_partitions(size_t n, int32_t* out, size_t out_cap);
int32_t gaussogram_complex_partitions(size_t n, int32_t* out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* GAUSSOGRAM_H */
