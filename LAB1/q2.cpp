#include <stdio.h>
#include <omp.h>

#define N 1000

double A[N][N], B[N][N], C[N][N];

void init_matrices() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
}

void matmul_serial() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void matmul_parallel_1D() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void matmul_parallel_2D() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

int main() {
    double t_serial, t_p1, t_p2;
    double start, end;

    init_matrices();
    start = omp_get_wtime();
    matmul_serial();
    end = omp_get_wtime();
    t_serial = end - start;

    init_matrices();
    start = omp_get_wtime();
    matmul_parallel_1D();
    end = omp_get_wtime();
    t_p1 = end - start;

    init_matrices();
    start = omp_get_wtime();
    matmul_parallel_2D();
    end = omp_get_wtime();
    t_p2 = end - start;

    printf("Sequential Time = %f\n", t_serial);
    printf("Parallel 1D Time = %f\n", t_p1);
    printf("Parallel 2D Time = %f\n", t_p2);

    printf("Speedup 1D = %f\n", t_serial / t_p1);
    printf("Speedup 2D = %f\n", t_serial / t_p2);

    return 0;
}
