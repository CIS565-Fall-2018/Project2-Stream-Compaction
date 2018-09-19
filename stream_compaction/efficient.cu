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

		//No-longer const int* idata, this time we do it in-place
		__global__ void upSweeping(int n, int stride, int *idata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n)
			{
				return;
			}

			int stride2times = stride * 2;
			if (idx % stride2times == 0)
			{
				idata[idx + stride2times - 1] += idata[idx + stride - 1];
			}
		}
		__global__ void downSweeping(int n, int stride, int *idata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n)
			{
				return;
			}

			int stride2times = stride * 2;
			if (idx % stride2times == 0)
			{
				int indexCorresponding = idx + stride - 1;
				int indexCorresponding2times = idx + stride2times - 1;
				int delta = idata[indexCorresponding];
				idata[indexCorresponding] = idata[indexCorresponding2times];
				idata[indexCorresponding2times] += delta;
			}
		}

		//Tell if a number is power of 2
		bool isPowerOfTwo(int n)
		{
			if (n == 0)
			{
				return false;
			}
			else
			{
				return (n & (n - 1)) == 0;
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int layer = ilog2ceil(n);
			int elementNumber = 0;

			if (isPowerOfTwo(n))
			{
				elementNumber = n;
			}
			else
			{
				elementNumber = 1 << layer;
			}
			int *paddedInput = new int[elementNumber];
			//Copy and pad the redundant elements with 0, if any
			for (int i = 0; i < elementNumber; ++i)
			{
				if (i >= n)
				{
					paddedInput[i] = 0;
				}
				else
				{
					paddedInput[i] = idata[i];
				}
			}

			int bufferSize = elementNumber * sizeof(int);
			int* paddedIDataDev = nullptr;

			cudaMalloc(reinterpret_cast<void**>(&paddedIDataDev), bufferSize);
			checkCUDAError("Malloc for padded input failed");
			cudaMemcpy(reinterpret_cast<void*>(paddedIDataDev), paddedInput, bufferSize, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy for padded input failed from host to device");

            timer().startGpuTimer();
            // TODO
			dim3 gridLayout((n - 1 + BLOCKSIZE) / BLOCKSIZE);
			dim3 blockLayout(BLOCKSIZE);

			for (int layerI = 0; layerI < layer - 1; ++layerI)
			{
				int stride = 1 << layerI;
				upSweeping<<<gridLayout, blockLayout>>>(elementNumber, stride, paddedIDataDev);
			}
			int initialZero = 0;
			cudaMemcpy(reinterpret_cast<void*>(paddedIDataDev + elementNumber - 1), &initialZero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy for the initial 0 failed from host to device");
			for (int layerI = layer - 1; layerI >= 0; --layerI)
			{
				int stride = 1 << layerI;
				downSweeping<<<gridLayout, blockLayout>>>(elementNumber, stride, paddedIDataDev);
			}

            timer().endGpuTimer();

			cudaMemcpy(odata, reinterpret_cast<void*>(paddedIDataDev), bufferSize, cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy for padded input failed from device to host");
			cudaFree(paddedIDataDev);
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
			int layer = ilog2ceil(n);
			int elementNumber = 0;

			if (isPowerOfTwo(n))
			{
				elementNumber = n;
			}
			else
			{
				elementNumber = 1 << layer;
			}
			int *paddedInput = new int[elementNumber];
			//Copy and pad the redundant elements with 0, if any
			for (int i = 0; i < elementNumber; ++i)
			{
				if (i >= n)
				{
					paddedInput[i] = 0;
				}
				else
				{
					paddedInput[i] = idata[i];
				}
			}

			int bufferSize = elementNumber * sizeof(int);
			
			int* paddedIDataDev;
			int* oDataDev;
			int* mDataDev;
			cudaMalloc(reinterpret_cast<void**>(&paddedIDataDev), bufferSize);
			checkCUDAError("Malloc for padded input failed");
			cudaMalloc(reinterpret_cast<void**>(&oDataDev), bufferSize);
			checkCUDAError("Malloc for output failed");
			cudaMalloc(reinterpret_cast<void**>(&mDataDev), bufferSize);
			checkCUDAError("Malloc for intermediate failed");
			cudaMemcpy(paddedIDataDev, paddedInput, bufferSize, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy for padded input failed");

			timer().startGpuTimer();
			// TODO
			dim3 gridLayout((n - 1 + BLOCKSIZE) / BLOCKSIZE);
			dim3 blockLayout(BLOCKSIZE);

			StreamCompaction::Common::kernMapToBoolean<<<gridLayout, blockLayout>>>(elementNumber, mDataDev, paddedIDataDev);
			int endingElement = 0;
			cudaMemcpy(&endingElement, reinterpret_cast<void**>(mDataDev + elementNumber - 1), sizeof(int), cudaMemcpyDeviceToHost);

			for (int layerI = 0; layerI < layer - 1; ++layerI)
			{
				int stride = 1 << layerI;
				upSweeping<<<gridLayout, blockLayout>>>(elementNumber, stride, mDataDev);
			}
			int initialZero = 0;
			cudaMemcpy(reinterpret_cast<void*>(mDataDev + elementNumber - 1), &initialZero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy for the initial 0 failed from host to device");
			for (int layerI = layer - 1; layerI >= 0; --layerI)
			{
				int stride = 1 << layerI;
				downSweeping<<<gridLayout, blockLayout>>>(elementNumber, stride, mDataDev);
			}
			int elements = 0;
			cudaMemcpy(&elements, reinterpret_cast<void*>(mDataDev + elementNumber - 1), sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy for element count failed from device to host");
			StreamCompaction::Common::kernScatter<<<gridLayout, blockLayout>>>(elementNumber, oDataDev, paddedIDataDev, mDataDev, mDataDev);


			timer().endGpuTimer();

			cudaMemcpy(odata, reinterpret_cast<void*>(oDataDev), bufferSize, cudaMemcpyDeviceToHost);
			if (endingElement == 1)
			{
				++elements;
			}
			cudaFree(paddedIDataDev);
			cudaFree(oDataDev);
			cudaFree(mDataDev);

            return elements;
        }
    }
}
