#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    __global__ void kernNaiveGPUScan(const int n, const int offset, const int* d_data_in, int* d_data_out) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx >= n) return;
      if (idx < offset) {
        d_data_out[idx] = d_data_in[idx];
      }
      else {
        d_data_out[idx] = d_data_in[idx] + d_data_in[idx - offset];
      }
    }

    __global__ void kernIncToExc(const int n, const int* d_data_in, int* d_data_out) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx == 0) {
        d_data_out[0] = PLUS_OP_IDENTITY;
      }
      else if (idx < n) {
        d_data_out[idx] = d_data_in[idx - 1];
      }
    }
    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int* odata, const int* idata) {
      dim3 blockSize(BLOCK_SIZE);
      dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

      // allocate memory
      int* d_data_in;
      int* d_data_out;
      cudaMalloc((void**)&d_data_in, n * sizeof(int));
      cudaMalloc((void**)&d_data_out, n * sizeof(int));
      cudaMemcpy(d_data_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      timer().startGpuTimer();
      for (int i = 0; i < ilog2ceil(n); ++i) {
        kernNaiveGPUScan << <gridSize, blockSize >> > (n, 1 << i, d_data_in, d_data_out);
        std::swap(d_data_in, d_data_out);
      }
      timer().endGpuTimer();
      // for readbility
      std::swap(d_data_in, d_data_out);
      cudaMemcpy(odata + 1, d_data_out, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
      odata[0] = 0;
      cudaFree(d_data_out);
      cudaFree(d_data_in);
    }
  }
}
