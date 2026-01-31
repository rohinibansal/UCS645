#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1<<16)

void daxpy_serial(double *X, double *Y, double a) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double *X, double *Y, double a) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main() {
    double *X, *Y, a = 2.5;
    double t_serial, t_parallel, speedup;
    double start, end;

    X = (double*)malloc(N * sizeof(double));
    Y = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    start = omp_get_wtime();
    daxpy_serial(X, Y, a);
    end = omp_get_wtime();
    t_serial = end - start;

    for (int i = 0; i < N; i++) {
        X[i] = i * 1.0;
    }

    start = omp_get_wtime();
    daxpy_parallel(X, Y, a);
    end = omp_get_wtime();
    t_parallel = end - start;

    speedup = t_serial / t_parallel;

    printf("Sequential Time = %f seconds\n", t_serial);
    printf("Parallel Time   = %f seconds\n", t_parallel);
    printf("Speedup         = %f\n", speedup);

    free(X);
    free(Y);
    return 0;
}
