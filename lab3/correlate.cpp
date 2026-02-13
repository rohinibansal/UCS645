#include "correlate.h"
#include <cmath>
#include <omp.h>
#include <vector>

using namespace std;

void correlate_par(int ny, int nx, const float* data, float* result)
{
    vector<double> mean(ny);
    vector<double> norm(ny);

    // Step 1: Compute mean
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        for (int k = 0; k < nx; k++)
            sum += data[k + i * nx];
        mean[i] = sum / nx;
    }

    // Step 2: Compute norm (std deviation)
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double sum = 0.0;
        for (int k = 0; k < nx; k++) {
            double val = data[k + i * nx] - mean[i];
            sum += val * val;
        }
        norm[i] = sqrt(sum);
    }

    // Step 3: Correlation using dot product
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {

            double dot = 0.0;

            for (int k = 0; k < nx; k++) {
                double xi = data[k + i * nx] - mean[i];
                double xj = data[k + j * nx] - mean[j];
                dot += xi * xj;
            }

            result[i + j * ny] = dot / (norm[i] * norm[j]);
        }
    }
}
