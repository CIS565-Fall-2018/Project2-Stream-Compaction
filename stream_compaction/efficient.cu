#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		int blocksize = 1024;
		dim3 blocknum;

		__global__ void GPUUpsweepreal(int n, int d, int *idata)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			int para = (1 << (d + 1));
			int para1 = 1 << d;
			if (idx < n)
			{
				idata[idx*para + para - 1] += idata[idx*para + para1 - 1];
			}
		}

		__global__ void GPUUpsweep(int n, int d, int *idata)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			int para = 1 << (d + 1);
			int para1 = 1 << d;
			if (idx < n)
			{
				if (idx >= 0 && idx%para == 0)
				{
					idata[idx + para - 1] += idata[idx + para1 - 1];
				}
			}
		}


		__global__ void GPUdownsweepreal(int n, int d, int *idata)
		{
			int idx= blockDim.x*blockIdx.x + threadIdx.x;
			int para = 1 << (d + 1);
			int para1 = 1 << d;
			if (idx < n)
			{
				int t = idata[idx*para + para1 - 1];
				idata[idx*para + para1 - 1] = idata[idx*para + para - 1];
				idata[idx*para + para - 1] += t;
			}
		}

		__global__ void GPUdownsweep(int n, int d, int *idata)
		{
			int idx = blockDim.x*blockIdx.x + threadIdx.x;
			int para = 1 << (d + 1);
			int para1 = 1 << d;
			if (idx < n)
			{
				if (idx >= 0 && idx%para == 0)
				{
					int t = idata[idx + para1 - 1];
					idata[idx + para1 - 1] = idata[idx + para - 1];
					idata[idx + para - 1] += t;
				}
			}

		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool istimer) {
			int dmax = ilog2ceil(n);
			int adjustlen = 1 << dmax;
			int *dev_arr;

			cudaMalloc((void**)& dev_arr, adjustlen* sizeof(int));
			checkCUDAError("cudaMalloc dev_arr failed!");

			cudaMemcpy(dev_arr, idata, adjustlen * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

			if(istimer)
            timer().startGpuTimer();

			for (int d = 0; d < dmax; d++)
			{
				blocknum = (adjustlen + blocksize - 1) / blocksize;
				GPUUpsweep << <blocknum, blocksize >> > (adjustlen, d, dev_arr);
			}
			cudaMemset(dev_arr + adjustlen - 1, 0, sizeof(int));
			for (int d = dmax - 1; d >= 0; d--)
			{
				blocknum = (adjustlen + blocksize - 1) / blocksize;
				GPUdownsweep << <blocknum, blocksize >> > (adjustlen, d, dev_arr);
			}
            // TODO
			if (istimer)
            timer().endGpuTimer();
			cudaMemcpy(odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_arr);
        }

		void realscan(int n, int *odata, const int *idata, bool istimer) {
			int dmax = ilog2ceil(n);
			int adjustlen = 1 << dmax;
			int *dev_arr;

			cudaMalloc((void**)& dev_arr, adjustlen * sizeof(int));
			checkCUDAError("cudaMalloc dev_arr failed!");

			cudaMemcpy(dev_arr, idata, adjustlen * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

			if (istimer)
				timer().startGpuTimer();

			for (int d = 0; d < dmax; d++)
			{
				int interval = (1 << (d + 1));
				blocknum = (adjustlen/interval + blocksize ) / blocksize;
				GPUUpsweepreal << <blocknum, blocksize >> > (adjustlen/interval, d, dev_arr);
			}

			cudaMemset(dev_arr + adjustlen - 1, 0, sizeof(int));
			for (int d = dmax - 1; d >= 0; d--)
			{
				int interval = (1 << (d + 1));
				blocknum = (adjustlen/interval + blocksize ) / blocksize;
				GPUdownsweepreal << <blocknum, blocksize >> > (adjustlen/interval, d, dev_arr);
			}
			// TODO
			if (istimer)
				timer().endGpuTimer();
			cudaMemcpy(odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_arr);
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
			int *dev_idata, *dev_odata, *dev_checker, *dev_indices;;

            
			

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_checker, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_checker failed!");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_idata,idata, n * sizeof(int), cudaMemcpyHostToDevice);
			
			//timer().startGpuTimer();
			
			blocknum = (n + blocksize ) / blocksize;
			Common::kernMapToBoolean << <blocknum, blocksize >> > (n, dev_checker, dev_idata);


			int *checker = new int[n];int *indices = new int[n];
			cudaMemcpy(checker, dev_checker, n * sizeof(int), cudaMemcpyDeviceToHost);
			
			realscan(n, indices, checker,true);

			cudaMemcpy(dev_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);
			
			int finalct = checker[n - 1] ? 1 : 0;

			int count = indices[n - 1]+finalct;

			blocknum = (n + blocksize) / blocksize;
			Common::kernScatter << <blocknum, blocksize >> > (n, dev_odata, dev_idata, dev_checker, dev_indices);
			//timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_checker);
			cudaFree(dev_indices);

			delete[]indices;
			delete[]checker;

            // TODO
            
            return count;

        }
    }
}
