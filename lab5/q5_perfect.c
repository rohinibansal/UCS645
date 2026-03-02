#include <mpi.h>
#include <stdio.h>

int isPerfect(int n){
    int sum=1;
    for(int i=2;i<=n/2;i++)
        if(n%i==0) sum+=i;
    return sum==n && n!=1;
}

int main(int argc, char** argv){

    int rank,size;
    int max = 10000;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank==0){
        int number=2;
        int active = size-1;
        MPI_Status status;

        while(active){
            int msg;
            MPI_Recv(&msg,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);

            if(number<=max){
                MPI_Send(&number,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
                number++;
            }else{
                int stop=-1;
                MPI_Send(&stop,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
                active--;
            }

            if(msg>0)
                printf("Perfect: %d\n",msg);
        }
    }else{
        int request=0, num;

        while(1){
            MPI_Send(&request,1,MPI_INT,0,0,MPI_COMM_WORLD);
            MPI_Recv(&num,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            if(num<0) break;

            if(isPerfect(num))
                request=num;
            else
                request=-num;
        }
    }

    MPI_Finalize();
    return 0;
}
