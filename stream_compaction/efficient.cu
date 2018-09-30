#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernelScanReduce(int n, int d, int* idata)
		{
			int thID = threadIdx.x + blockDim.x * blockIdx.x;
			if (thID >= n) return;
			int temp = 1 << d;
			int temp2 = 1 << (d - 1);
			if ((thID % temp) == 0)
			{
				idata[thID + temp - 1] = idata[thID + temp2 - 1] + idata[thID + temp - 1];
			}
		}
		// two array to get result
		//__global__ void kernelScanReduce(int n, int d, int* odata, int* idata)
		//{
		//	int thID = threadIdx.x + blockDim.x * blockIdx.x;
		//	if (thID >= n) return;
		//	int temp = 1 << d;
		//	int temp2 = 1 << (d - 1);
		//	odata[thID] = idata[thID];
		//	if ((thID % temp) == 0)
		//	{
		//		odata[thID + temp - 1] = idata[thID + temp2 - 1] + idata[thID + temp - 1];
		//	}
		//}
		//__global__ void kernelScanDownSweep(int n, int d, int* odata, int* idata)
		//{
 	//		int thID = threadIdx.x + blockDim.x * blockIdx.x;
		//	if (thID >= n) return;
		//	int tempdp1 = 1 << (d + 1);
		//	int tempd = 1 << d;
		//	odata[thID] = idata[thID];
		//	if ((thID % tempdp1) == 0)
		//	{
		//		int t = idata[thID + tempd - 1];
		//		odata[thID + tempd - 1] = idata[thID + tempdp1 - 1];
		//		odata[thID + tempdp1 - 1] = t + idata[thID + tempdp1 - 1];
		//	}
		//}


		__global__ void kernelScanDownSweep(int n, int d, int* idata)
		{
			int thID = threadIdx.x + blockDim.x * blockIdx.x;
			if (thID >= n) return;
			int tempdp1 = 1 << (d + 1);
			int tempd = 1 << d;
			if ((thID % tempdp1) == 0)
			{
				int t = idata[thID + tempd - 1];
				idata[thID + tempd - 1] = idata[thID + tempdp1 - 1];
				idata[thID + tempdp1 - 1] = t + idata[thID + tempdp1 - 1];
			}
		}

		__global__ void kernelChangeN1(int *arr, int index, int identity)
		{
				arr[index] = identity;			
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

		void myScan(int n, int temp, int* idata)
		{
			dim3 fullBlocksPerGrid((temp + blockSize - 1) / blockSize);
			int myIdentity = 0;
			for (int d = 1; d <= ilog2ceil(n); ++d)
			{
				kernelScanReduce << <fullBlocksPerGrid, blockSize >> > (temp, d, idata);
			}

			kernelChangeN1 << < 1, 1 >> > (idata, temp - 1, myIdentity);

			for (int d = ilog2ceil(n) - 1; d >= 0; --d)
			{
				kernelScanDownSweep << <fullBlocksPerGrid, blockSize >> > (temp, d, idata);
			}
		}
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int temp = 1 << ilog2ceil(n);

			dim3 fullBlocksPerGrid((temp + blockSize - 1) / blockSize);
			int* dev_In = NULL;
			int* dev_Out = NULL;
			cudaMalloc((void**)&dev_In, temp * sizeof(int));
			checkCUDAError("Malloc dev_In failed!");
			cudaMalloc((void**)&dev_Out, temp * sizeof(int));
			checkCUDAError("Malloc dev_Out failed!");
			cudaMemcpy(dev_In, idata, temp * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy from idata to dev_In failed!");
			myScan(n, temp, dev_In);
			
			std::swap(dev_Out, dev_In);
			cudaMemcpy(odata, dev_Out, temp * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memcoy from dev_Out to odata failed!");

            timer().endGpuTimer();

			cudaFree(dev_In);
			cudaFree(dev_Out);
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


		__global__ void kernelPredicate(int n, int* odata, const int* idata)
		{
			int thID = threadIdx.x + blockDim.x * blockIdx.x;
			if (thID >= n) return;
			if (idata[thID] == 0)
			{
				odata[thID] = 0;
			}
			else
			{
				odata[thID] = 1;
			}

		}
		__global__ void kernelScatter(int n, int* odata, int* myBool ,int* address, int* idata)
		{
			int thID = threadIdx.x + blockDim.x * blockIdx.x;
			if (thID >= n) return;
			if (myBool[thID] == 1)
			{
				odata[address[thID]] = idata[thID];
			}

		}
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int temp = 1 << ilog2ceil(n);
			dim3 fullBlocksPerGrid((temp + blockSize - 1) / blockSize);
			int* dev_In = NULL;
			int* dev_Out = NULL;
			int* dev_Bool = NULL;
			int* dev_Address = NULL;

			cudaMalloc((void**)&dev_In, temp * sizeof(int));
			checkCUDAError("compact: malloc dev_In failed!");
			cudaMalloc((void**)&dev_Out, temp * sizeof(int));
			checkCUDAError("compact: malloc dev_Out failed!");
			cudaMalloc((void**)&dev_Bool, temp * sizeof(int));
			checkCUDAError("compact: malloc dev_Bool failed!");
			cudaMalloc((void**)&dev_Address, temp * sizeof(int));
			checkCUDAError("compact: malloc dev_Address failed!");

			cudaMemcpy(dev_In, idata, temp * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("compact: memcpy from idata to dev_In failed!");

			kernelPredicate << < fullBlocksPerGrid, blockSize >> > (temp, dev_Bool, dev_In);

			cudaMemcpy(dev_Address, dev_Bool, temp * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("compact: memcoy from dev_Bool to dev_Address failed!");

			myScan(n, temp, dev_Address);

			kernelScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_Out, dev_Bool, dev_Address, dev_In);
			
			cudaMemcpy(odata, dev_Out, temp * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_In);
			cudaFree(dev_Out);
			cudaFree(dev_Bool);
			cudaFree(dev_Address);

			int flag = 0;
			for (int i = 0; i < n; ++i)
			{
				if (odata[i] != 0)
				{
					flag++;
				}
				else
				{
					break;
				}
			}
			timer().endGpuTimer();
			return flag;
        }
    }
}
