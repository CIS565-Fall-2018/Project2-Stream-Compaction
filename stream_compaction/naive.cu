#include <cuda.h>
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

        const int BLOCK_SIZE = 512;

        __global__ void kernScan(int n, int *odata, int *idata, int offset)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        __global__ void kernShift(int n, int *odata, int *idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            odata[index] = (index > 0) ? idata[index - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            for (int offset = 1; offset <= ilog2ceil(n); offset++)
            {
                kernScan<<<numBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, 1 << offset - 1);
                std::swap(dev_idata, dev_odata);
            }

            kernShift<<<numBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
