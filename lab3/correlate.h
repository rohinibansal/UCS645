#ifndef CORRELATE_H
#define CORRELATE_H

void correlate_seq(int ny, int nx, const float* data, float* result);
void correlate_par(int ny, int nx, const float* data, float* result);

#endif
