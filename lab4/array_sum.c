#include <mpi.h>
#include <stdio.h>

#define N 1000000   // Increased size for meaningful timing

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = N / size;

    static int arr[N];
    static int local_arr[N];

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = 1;   // simple values for consistent result
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(arr, local_size, MPI_INT,
                local_arr, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_size; i++)
        local_sum += local_arr[i];

    long long global_sum = 0;

    MPI_Reduce(&local_sum, &global_sum, 1,
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
