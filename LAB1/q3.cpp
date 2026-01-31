#include <stdio.h>
#include <omp.h>

#define NUM_STEPS 10000000

double pi_serial() {
    double step = 1.0 / NUM_STEPS;
    double sum = 0.0;
    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

double pi_parallel() {
    double step = 1.0 / NUM_STEPS;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

int main() {
    double pi_s, pi_p;
    double t_serial, t_parallel;
    double start, end;

    start = omp_get_wtime();
    pi_s = pi_serial();
    end = omp_get_wtime();
    t_serial = end - start;

    start = omp_get_wtime();
    pi_p = pi_parallel();
    end = omp_get_wtime();
    t_parallel = end - start;

    printf("Pi (Serial)   = %f\n", pi_s);
    printf("Pi (Parallel) = %f\n", pi_p);
    printf("Serial Time   = %f seconds\n", t_serial);
    printf("Parallel Time = %f seconds\n", t_parallel);
    printf("Speedup       = %f\n", t_serial / t_parallel);

    return 0;
}
