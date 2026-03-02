#include <omp.h>
#include <cmath>
#include <iostream>
using namespace std;

const int N=500;
double x[N], y[N], z[N], fx[N], fy[N], fz[N];

int main(){
    for(int i=0;i<N;i++){
        x[i]=y[i]=z[i]=i*0.01;
        fx[i]=fy[i]=fz[i]=0;
    }

    double energy=0;
    double start=omp_get_wtime();

    #pragma omp parallel for reduction(+:energy) schedule(dynamic)
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            double dx=x[i]-x[j];
            double dy=y[i]-y[j];
            double dz=z[i]-z[j];
            double r2=dx*dx+dy*dy+dz*dz;
            if(r2==0) continue;

            double inv6=1.0/(r2*r2*r2);
            double f=24*inv6*(2*inv6-1)/r2;

            #pragma omp atomic
            fx[i]+=f*dx;
            #pragma omp atomic
            fx[j]-=f*dx;

            energy+=4*inv6*(inv6-1);
        }
    }

    double end=omp_get_wtime();
    cout<<omp_get_max_threads()<<" "<<(end-start)<<endl;
}
