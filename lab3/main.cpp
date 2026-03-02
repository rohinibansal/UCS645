#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <random>
#include "correlate.h"

using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Usage: ./correlate ny nx" << endl;
        return 1;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);

    cout << "Matrix size: " << ny << " x " << nx << endl;

    vector<float> data(ny * nx);
    vector<float> result(ny * ny);

    // Random initialization
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0, 10.0);

    for (int i = 0; i < ny * nx; i++)
        data[i] = dis(gen);

    // Sequential timing
    auto start1 = high_resolution_clock::now();
    correlate_seq(ny, nx, data.data(), result.data());
    auto end1 = high_resolution_clock::now();

    auto seq_time = duration_cast<milliseconds>(end1 - start1).count();
    cout << "Sequential time: " << seq_time << " ms" << endl;

    // Parallel timing
    auto start2 = high_resolution_clock::now();
    correlate_par(ny, nx, data.data(), result.data());
    auto end2 = high_resolution_clock::now();

    auto par_time = duration_cast<milliseconds>(end2 - start2).count();
    cout << "Parallel time: " << par_time << " ms" << endl;

    cout << "Speedup: " << (double)seq_time / par_time << "x" << endl;

    return 0;
}
