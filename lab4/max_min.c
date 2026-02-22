#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000   // Increase workload for timing

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    int local_max = 0;
    int local_min = 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Generate large number of random values
    for (int i = 0; i < N/size; i++) {
        int num = rand() % 1001;

        if (num > local_max)
            local_max = num;

        if (num < local_min)
            local_min = num;
    }

    struct {
        int value;
        int rank;
    } local_maxloc, global_maxloc,
      local_minloc, global_minloc;

    local_maxloc.value = local_max;
    local_maxloc.rank = rank;

    local_minloc.value = local_min;
    local_minloc.rank = rank;

    MPI_Reduce(&local_maxloc, &global_maxloc, 1,
               MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_minloc, &global_minloc, 1,
               MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("%d %f\n", size, end - start);
    }

    MPI_Finalize();
    return 0;
}
