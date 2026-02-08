#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    string A="ACACACTA", B="AGCACACA";
    int n=A.size(), m=B.size();
    vector<vector<int>> H(n+1, vector<int>(m+1,0));

    double start=omp_get_wtime();

    for(int d=1; d<=n+m; d++){
        #pragma omp parallel for
        for(int i=1;i<=n;i++){
            int j=d-i;
            if(j>=1 && j<=m){
                int score=(A[i-1]==B[j-1])?2:-1;
                H[i][j]=max(0,max({
                    H[i-1][j-1]+score,
                    H[i-1][j]-1,
                    H[i][j-1]-1
                }));
            }
        }
    }

    double end=omp_get_wtime();
    cout<<omp_get_max_threads()<<" "<<(end-start)<<endl;
}


