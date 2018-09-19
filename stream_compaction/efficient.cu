#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <memory>
#include <iostream>

namespace StreamCompaction {
  namespace Efficient {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }


    __global__ void kernEfficientGpuScan(int n, int *odata, const int *idata) {

    }

    __global__ void kernBuildTree(int n, const int* d_data_in, int* d_data_out) {

    }

    __global__ void kernEfficientScanUp(int n, int bitShift, int* d_data_in) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx >= n) {
        return;
      }
      int pow1 = 1 << bitShift;
      int pow2 = 1 << (bitShift + 1);
      d_data_in[idx * pow2 + pow2 - 1] += d_data_in[idx * pow2 + pow1 - 1];
    }

    __global__ void kernEfficientScanDown(int n, int bitShift, int* d_data_in) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx >= n) {
        return;
      }
      int pow1 = 1 << bitShift;
      int pow2 = 1 << (bitShift + 1);
      int pos1 = idx * pow2 + pow1 - 1;
      int pos2 = idx * pow2 + pow2 - 1;
      int temp = d_data_in[pos1];
      d_data_in[pos1] = d_data_in[pos2];
      d_data_in[pos2] += temp;
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      // make length to 2^n
      int level = ilog2ceil(n);
      int trueN = 1 << level;



      std::unique_ptr<int[]>trueIData{ new int[trueN] };

      // pad 0 to the end
      for (int i = 0; i < n; ++i) {
        trueIData[i] = idata[i];
      }
      for (int i = n; i < trueN; ++i) {
        trueIData[i] = 0;
      }

      // allocate memory
      int* d_data_in;
      cudaMalloc((void**)&d_data_in, trueN * sizeof(int));
      cudaMemcpy(d_data_in, trueIData.get(), trueN * sizeof(int), cudaMemcpyHostToDevice);
      timer().startGpuTimer();

      dim3 blockSize(BLOCK_SIZE);
      dim3 gridSize;
      int pow2;
      // go up
      for (int i = 0; i < level; ++i)
      {
        pow2 = 1 << (i + 1);
        gridSize = ((trueN / pow2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernEfficientScanUp << <gridSize, blockSize >> > (trueN / pow2, i, d_data_in);
      }
      cudaMemset(d_data_in + trueN - 1, 0, sizeof(int));

      // go down
      for (int i = level - 1; i > -1; --i)
      {
        pow2 = 1 << (i + 1);
        gridSize = ((trueN / pow2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernEfficientScanDown << <gridSize, blockSize >> > (trueN / pow2, i, d_data_in);
      }

      timer().endGpuTimer();
      // only need copy n, no need to copy trueN
      cudaMemcpy(odata, d_data_in, n * sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(d_data_in);
    }

    __global__ void kernValueMapToOne(int n, int* d_ones_out, int* d_data_in) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx >= n) {
        return;
      }
      if (d_data_in[idx] != 0) {
        d_ones_out[idx] = 1;
      }
    }

    __global__ void kernCompact(int n, int* d_indices, int* d_data_in, int* d_data_to_compact) {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if (idx >= n) {
        return;
      }
      int data = d_data_in[idx];
      if (data != 0) {
        d_data_to_compact[d_indices[idx]] = data;
      }
    }

    /**
     * Performs stream compaction on idata, storing the result into odata.
     * All zeroes are discarded.
     *
     * @param n      The number of elements in idata.
     * @param odata  The array into which to store elements.
     * @param idata  The array of elements to compact.
     * @returns      The number of elements remaining after compaction.
     */
    int compact(int n, int *odata, const int *idata) {
      // ugly implementation of reusing scan code
      // used for mapping to 1 and as the scan result
      int* indexArray = (int*)malloc(n * sizeof(int));
      int* d_data_in;
      int* d_compacted_data;
      cudaMalloc((void**)&d_data_in, n * sizeof(int));
      cudaMalloc((void**)&d_compacted_data, n * sizeof(int));

      cudaMemset(d_compacted_data, 0, n * sizeof(int));

      cudaMemcpy(d_data_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      int* d_ones_scan_result;

      // calc true values
      // make length to 2^n
      int level = ilog2ceil(n);
      int trueN = 1 << level;

      // set scan_result zero
      cudaMalloc((void**)&d_ones_scan_result, trueN * sizeof(int));
      cudaMemset(d_ones_scan_result, 0, trueN * sizeof(int));

      // useful constants
      dim3 blockSize(BLOCK_SIZE);
      dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
      int pow2;

      timer().startGpuTimer();
      // do ones
      kernValueMapToOne << <gridSize, blockSize >> > (n, d_ones_scan_result, d_data_in);

      // TODO(zichuanyu) make this a in the future
      // scan
      // go up
      for (int i = 0; i < level; ++i)
      {
        pow2 = 1 << (i + 1);
        gridSize = ((trueN / pow2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernEfficientScanUp << <gridSize, blockSize >> > (trueN / pow2, i, d_ones_scan_result);
      }
      cudaMemset(d_ones_scan_result + trueN - 1, 0, sizeof(int));
      // go down
      for (int i = level - 1; i > -1; --i)
      {
        pow2 = 1 << (i + 1);
        gridSize = ((trueN / pow2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernEfficientScanDown << <gridSize, blockSize >> > (trueN / pow2, i, d_ones_scan_result);
      }

      // compact, only use useful part of the array 
      gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      kernCompact << <gridSize, blockSize >> > (n, d_ones_scan_result, d_data_in, d_compacted_data);
      timer().endGpuTimer();
      cudaMemcpy(odata, d_compacted_data, n * sizeof(int), cudaMemcpyDeviceToHost);
      // count how many nums
      // cpu or gpu
      int num = 0;
      for (int i = 0; i < n; ++i) {
        if (odata[i] == 0) {
          break;
        }
        ++num;
      }
      cudaFree(d_compacted_data);
      cudaFree(d_ones_scan_result);
      cudaFree(d_data_in);
      return num;
    }
  }
}
