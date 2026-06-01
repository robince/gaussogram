/* Generates golden vectors from the reference gft.c for the dyadic_complex
 * parity test. Compiled once against FFTW; the produced text file is committed
 * so the Rust test never needs FFTW. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../extern/fst-uofc/gft.h"

#define PI 3.1415926535897931

/* Shared deterministic input (must match the Rust test's generator). */
static void make_input(double *sig, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / (double)N;
        double re = sin(2 * PI * 2.0 * i / N) + 0.5 * cos(2 * PI * 5.0 * i / N);
        double im = 0.25 * sin(2 * PI * 3.0 * i / N) - 0.1 * cos(2 * PI * 1.0 * i / N);
        re += 0.05 * t; /* mild ramp */
        sig[i * 2] = re;
        sig[i * 2 + 1] = im;
    }
}

int main(void) {
    int sizes[] = {8, 16, 32, 64};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    FILE *f = fopen("golden/complex_golden.txt", "w");
    if (!f) { perror("fopen"); return 1; }

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        double *sig = malloc(sizeof(double) * N * 2);
        make_input(sig, N);
        int *pars = gft_1dPartitions((unsigned)N);
        double *win = windows(N, gaussian);
        gft_1dComplex64(sig, (unsigned)N, win, pars, 1);
        fprintf(f, "N %d\n", N);
        for (int i = 0; i < N; i++) {
            fprintf(f, "%.17g %.17g\n", sig[i * 2], sig[i * 2 + 1]);
        }
        free(sig);
        free(pars);
        free(win);
    }
    fclose(f);
    printf("wrote golden/complex_golden.txt\n");
    return 0;
}
