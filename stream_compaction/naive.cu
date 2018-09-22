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
        // TODO: __global__
		__global__ void kernScan(int n, int POT, int *odata, int *idata)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			if (index >= POT)
			{
				odata[index] = idata[index] + idata[index - POT];
			}
			else
			{
				odata[index] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int count = ilog2ceil(n);
			int *dev_odata;
			int *dev_idata;
			dim3 gridsize((n - 1) / blocksize + 1);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_odata");

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_idata");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
			for (int i = 0; i < count; i++)
			{
				kernScan << <gridsize, blocksize >> > (n, 1 << i, dev_odata, dev_idata);
#ifdef SYNC_GRID
				cudaThreadSynchronize();
#endif
				if (i != count - 1)//if not last time, exchange buffer for next kern
				{
					int *temp = dev_odata;
					dev_odata = dev_idata;
					dev_idata = temp;
				}
			}
            timer().endGpuTimer();

			//shift right and insert identity
			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(dev_odata);
			checkCUDAErrorFn("free dev_odata");

			cudaFree(dev_idata);
			checkCUDAErrorFn("free dev_idata");

        }
    }
}
