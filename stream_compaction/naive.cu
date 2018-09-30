#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>



namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

		__global__ void kernelScan(int n, int d, int* odata, int* idata)
		{
			int thID = threadIdx.x + blockDim.x * blockIdx.x;
			int temp = 1 << (d - 1);

			if (thID >= n) return;
			if (thID >= temp)
			{
				odata[thID] = idata[thID - temp] + idata[thID];
			}
			else
			{
				odata[thID] = idata[thID];
			}
			
		}
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int* dev_tempIn = NULL;
			int* dev_tempOut = NULL;

			cudaMalloc((void**)&dev_tempIn, n * sizeof(int));
			checkCUDAError("Malloc memory to dev_tempIn failed!");
			cudaMalloc((void**)&dev_tempOut, n * sizeof(int));
			checkCUDAError("Malloc memory to dev_tempOut failed!");
			cudaMemcpy(dev_tempIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Memory copy from host to device failed!");

			for (int d = 1; d <= ilog2ceil(n); ++d)
			{
				kernelScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_tempOut, dev_tempIn);
				std::swap(dev_tempOut, dev_tempIn);
			}
			std::swap(dev_tempOut, dev_tempIn);

			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_tempOut, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memory copy from device to host failed!");
			//for (int i = 0; i < n; ++i)
			//{
			//	std::cout << odata[i] << " ";
			//}
            timer().endGpuTimer();

			cudaFree(dev_tempIn);
			cudaFree(dev_tempOut);
        }
    }
}
