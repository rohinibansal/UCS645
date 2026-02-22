#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000   // Large size for meaningful timing

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = N / size;

    static long long A[N];
    static long long B[N];
    static long long local_A[N];
    static long long local_B[N];

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            A[i] = 1;
            B[i] = 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, local_size, MPI_LONG_LONG,
                local_A, local_size, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, local_size, MPI_LONG_LONG,
                local_B, local_size, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    long long local_dot = 0;

    for (int i = 0; i < local_size; i++)
        local_dot += local_A[i] * local_B[i];

    long long global_dot = 0;

    MPI_Reduce(&local_dot, &global_dot, 1,
               MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("%d %f\n", size, end - start);
    }

    MPI_Finalize();
    return 0;
}
