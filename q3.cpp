#include <omp.h>
#include <iostream>
#include <vector>
using namespace std;

int main(){
    int N=200, steps=500;
    vector<vector<double>> T(N,vector<double>(N,0));
    vector<vector<double>> Tn=T;

    T[N/2][N/2]=100;

    double start=omp_get_wtime();

    for(int t=0;t<steps;t++){
        #pragma omp parallel for schedule(static)
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){
                Tn[i][j]=0.25*(T[i+1][j]+T[i-1][j]+T[i][j+1]+T[i][j-1]);
            }
        }
        T=Tn;
    }

    double end=omp_get_wtime();
    cout<<omp_get_max_threads()<<" "<<(end-start)<<endl;
}
