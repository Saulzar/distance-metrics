#ifndef METRICS_H
#define METRICS_H


void distanceL1(float *ref, int ref_size,  float *query, int query_size, int dim, float *result);
void distanceL2(float *ref, int ref_size,  float *query, int query_size, int dim, float *result);
void distanceLP(float *ref, int ref_size,  float *query, int query_size, int dim,  float *result, float p);

#endif




 