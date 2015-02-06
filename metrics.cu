// Includes
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include <time.h>

#include "THC/THCTensor.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#include <stdexcept> 
#include <sstream>

//Code breaks with different values of this constant
#define BLOCK_DIM                     32
#define USE_BLOCK

#define cudaAssert(ans) { cudaAssert_((ans), __FILE__, __LINE__); }
inline void cudaAssert_(cudaError_t code, const char *file, int line)
{

  if (code != cudaSuccess) 
  {
    std::ostringstream out;
    out << "cuda error " << file << ":" << line << " " << cudaGetErrorString(code);
    
    throw std::logic_error(out.str());
    
  }
}

/*
template<typename Op>
__global__ void cuDistanceSimple( float* x1, int size1, float* x2, int size2,  int dim,  float* distances, Op const &op)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ int bx;
    __shared__ int by;
    
    bx = blockIdx.x * BLOCK_DIM;
    by = blockIdx.y * BLOCK_DIM;
       
    int inside1 = bx + tx < size1;
    int inside2 = by + ty < size2;
    
    float sum = 0.0;
    
      
    if (inside1 && inside2) {
      
      for (int k = 0; k < dim; ++k) {
        sum += op(x1[(bx + tx) * dim +  k], x2 [ (by + ty) * dim +  k]);
      }
    
      
    }

    if (inside1 && inside2) {
      distances[(bx + tx) * size2 + (by + ty)] = sum;
    }
}*/



template<typename Op>
__global__ void cuDistance( float* a, int size1, float* b, int size2,  int dim,  float* distances, Op const &op)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ int bx, by;    
    
    bx = blockIdx.x * BLOCK_DIM;
    by = blockIdx.y * BLOCK_DIM;
    
    __shared__ float sa[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sb[BLOCK_DIM][BLOCK_DIM];    
    
    int inside1 = (bx + tx < size1);   
    int inside2 = (by + ty < size2);   
    int inside3 = (by + tx < size2);
    
    int inside = inside1 && inside2;
    
    float sum = 0.0;
    
    for(int offk = 0; offk < dim; offk += BLOCK_DIM) {
         
      if (offk + ty < dim) {
         sa[tx][ty] = inside1 ? a[(bx + tx) * dim + ty + offk] : 0;
         sb[tx][ty] = inside3 ? b[(by + tx) * dim + ty + offk] : 0;
      } else {
        sa[tx][ty] = 0;
        sb[tx][ty] = 0;
      }
             
      __syncthreads();
      
      if(inside) {
        for (int k = 0; k < BLOCK_DIM; ++k) {
          sum += op(sa[tx][k], sb[ty][k]);
        }
      }
      
      __syncthreads();
    }

    if (inside) {
      distances[(bx + tx) * size2 + (by + ty)] = sum;
    }
}



struct LP {

  const float p;
  
  __host__ __device__ LP(const float &p) : p(p) { }
  
  __host__ __device__ float operator()(const float& x, const float& y) const {
    return pow(fabs(x - y), p);
  }
};


struct L1 {
   
  __host__ __device__ float operator()(const float& x, const float& y) const {
    return fabs(x - y);
  }
};


struct L2 {
   
  __host__ __device__ float operator()(const float& x, const float& y) const {
    float d = x - y;
    return d * d;
  }  
};



dim3 blocks(int ref_size, int query_size) {
  int bx = (ref_size  + BLOCK_DIM - 1)/BLOCK_DIM;
  int by = (query_size + BLOCK_DIM - 1)/BLOCK_DIM;

  return dim3(bx, by, 1);
}



template<typename Op>
void distance(float *ref, int ref_size,  float *query, int query_size, int dim, float *result, Op const &op) {
  dim3 grid = blocks(ref_size, query_size);
  dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);   
  
//   printf("dim = %d \n", dim);
  
  Op *deviceOp;
  cudaMalloc((void**)&deviceOp, sizeof(Op));
  cudaMemcpy((void*)deviceOp, (void*)&op, sizeof(Op), cudaMemcpyHostToDevice);
  
  cuDistance<<<grid, threads>>>(ref, ref_size, query, query_size, dim, result, *deviceOp); 
  cudaFree(deviceOp);
  
  cudaAssert(cudaPeekAtLastError());  
}




void distanceL1(float *ref, int ref_size,  float *query, int query_size, int dim, float *result) {
  return distance(ref, ref_size, query, query_size, dim, result, L1());
}

void distanceL2(float *ref, int ref_size,  float *query, int query_size, int dim, float *result) {
  return distance(ref, ref_size, query, query_size, dim, result, L2());  
}


void distanceLP(float *ref, int ref_size,  float *query, int query_size, int dim,  float *result, float p) {
  return distance(ref, ref_size, query, query_size, dim, result, LP(p));
}


// int main(int argc, char **argv) {
//   
//   int dim = 250;
//   int n = 250;
//   
//   thrust::host_vector<float> x(n * dim);
//   for (int i = 0; i < n * dim; ++i) {
//     x[i] = i;
//   }
//   
//   thrust::device_vector<float> dx = x;
//   thrust::device_vector<float> result(n * n);
//   
//   float *xPtr = thrust::raw_pointer_cast(dx.data());
//   float *resultPtr = thrust::raw_pointer_cast(result.data());
//   
//   
//   thrust::device_vector<float> result1(n * n);
//   float *result1Ptr = thrust::raw_pointer_cast(result1.data());
// 
//   distanceL2(xPtr, n, xPtr, n, dim, resultPtr);
//   distanceSimpleL2(xPtr, n, xPtr, n, dim, result1Ptr);
//   
//   thrust::host_vector<float> r = result;
//   thrust::host_vector<float> r1 = result1;
//   
//   for (int i = 0; i < n * n; ++i) {
//     float err = fabs(r[i] - r1[i]);
//     if(err > 1) {
//       printf("fail %d: %f %f\n", i, r[i], r1[i]);  
//       break;
//     }
//   }
//   
// }

