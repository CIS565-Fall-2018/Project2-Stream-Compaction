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
		// Stride: 2^n, the stride of the elements
		__global__ void streamCompactionNaive(int n, int stride, const int* idata, int* odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n)
			{
				return;
			}

			// Add element
			if (idx >= stride)
			{
				odata[idx] = idata[idx - stride] + idata[idx];
			}
			// Got no element to add 
			else
			{
				odata[idx] = idata[idx];
			}
		}

		void swapBuffer(int*& ptr1, int*& ptr2)
		{
			int* temp = ptr1;
			ptr1 = ptr2;
			ptr2 = temp;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int* idataDev = nullptr;
			int* odataDev = nullptr;
			size_t heapSize = sizeof(int) * n;
			cudaMalloc(reinterpret_cast<void**>(&idataDev), heapSize);
			checkCUDAError("Malloc for input data failed");
			cudaMalloc(reinterpret_cast<void**>(&odataDev), heapSize);
			checkCUDAError("Malloc for output data failed");

			cudaMemcpy(reinterpret_cast<void*>(idataDev), idata, heapSize, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy input data host 2 device failed");
			cudaMemcpy(reinterpret_cast<void*>(odataDev), odata, heapSize, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy input data host 2 device failed");

			dim3 gridLayout((n - 1 + BLOCKSIZE) / BLOCKSIZE);
			dim3 blockLayout(BLOCKSIZE);

			//Layers of the problem
			int layer = ilog2ceil(n);

			timer().startGpuTimer();
			// TODO
			for (int layerI = 0; layerI < layer; ++layerI)
			{
				int strideWidth = 1 << layerI;
				streamCompactionNaive<<<gridLayout, blockLayout>>>(n, strideWidth, idataDev, odataDev);
				
				swapBuffer(idataDev, odataDev);
			}
			swapBuffer(idataDev, odataDev);
            timer().endGpuTimer();

			cudaMemcpy(odata + 1, reinterpret_cast<void*>(odataDev), heapSize - sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
			checkCUDAError("Memcpy output data device 2 host failed");

			cudaFree(idataDev);
			cudaFree(odataDev);
        }
    }
}
