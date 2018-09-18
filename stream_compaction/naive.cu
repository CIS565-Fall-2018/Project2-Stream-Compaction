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

		__device__  int ilog2Device(int x) {
			int lg = 0;
			while (x >>= 1) {
				++lg;
			}
			return lg;
		}

		__device__ int ilog2ceilDevice(int x) {
			return x == 1 ? 0 : ilog2Device(x - 1) + 1;
		}

		__global__ void plusp(int n, int *idata, int *odata ,int d)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			if (idx < n)
			{
					if (idx >= (1 << (d - 1)))
					{
						odata[idx] = idata[idx-(1 << (d - 1))] + idata[idx];

					}
					
					else
					{
						odata[idx] = idata[idx];
					}
					__syncthreads();
			}
			
		}
		__global__ void resetidata(int n, int *idata, int *odata)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			if (idx < n)
			{
				idata[idx] = odata[idx];
				
			}

		}

		__global__ void toExclusive(int n, int *idata, int *odata)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			if (idx < n)
			{
				odata[0] = 0;
				if (idx > 0)
				{
					odata[idx] = idata[idx - 1];
				}

			}

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			if (n <= 0)return;
			odata[0] = 0;
			int blocksize = 1024;
			dim3 blocknum = (n + blocksize - 1) / blocksize;
			int *dev_idata, *dev_odata;
			cudaMalloc((void**) & dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)& dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

            timer().startGpuTimer();
            // TODO
			int dmax = ilog2ceil(n);
			for (int d = 1; d <= dmax; ++d)
			{
				plusp << <blocknum, blocksize >> > (n, dev_idata, dev_odata,d);
				resetidata << <blocknum, blocksize >> > (n, dev_idata, dev_odata);
			}
			toExclusive << < blocknum, blocksize >> > (n, dev_idata, dev_odata);
            timer().endGpuTimer();
			
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }

    }
}
