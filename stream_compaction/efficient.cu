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

		__global__ void kernUpSweep(int n, int POT, int POT_EX, int *data)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			if (index % POT_EX != 0) return;

			data[index + POT_EX - 1] += data[index + POT - 1];
		}

		__global__ void kernDownSweep(int n, int POT, int POT_EX, int *data)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
			if (index % POT_EX != 0) return;

			int temp = data[index + POT - 1];
			data[index + POT - 1] = data[index + POT_EX - 1];
			data[index + POT_EX - 1] += temp;
		}


		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			int count = ilog2ceil(n);
			int number = 1 << count;
			int *dev_data;
			dim3 gridsize((number - 1) / blocksize + 1);

			cudaMalloc((void**)&dev_data, number * sizeof(int));
			checkCUDAErrorFn("malloc dev_data");

			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			if (number > n)
			{
				cudaMemset(dev_data + n, 0, (number - n) * sizeof(int));
				checkCUDAErrorFn("set dev_data");
			}

			//start ticking
			timer().startGpuTimer();
			for (int i = 0; i < count; i++)
			{
				kernUpSweep << <gridsize, blocksize >> > (number, 1 << i, 1 << i + 1, dev_data);
#ifdef SYNC_GRID
				cudaThreadSynchronize();
#endif
			}

			//set data[number-1] to 0
			cudaMemset((void*)(dev_data + (number - 1)), 0, sizeof(int));
			checkCUDAErrorFn("set dev_data[number-1]");

			for (int i = count - 1; i >= 0; i--)
			{
				kernDownSweep << <gridsize, blocksize >> > (number, 1 << i, 1 << i + 1, dev_data);
#ifdef SYNC_GRID
				cudaThreadSynchronize();
#endif
			}

			//stop ticking
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_data);
			checkCUDAErrorFn("free dev_data");
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
			int result = 0;
			int count = ilog2ceil(n);
			int number = 1 << count;
			int *dev_idata;
			int *dev_odata;
			int *dev_indices;
			int *dev_bools;
			dim3 gridsize((number - 1) / blocksize + 1);
			dim3 gridsize_EXACT((n - 1) / blocksize + 1);

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_idata");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_odata");

			cudaMalloc((void**)&dev_indices, number * sizeof(int));
			checkCUDAErrorFn("malloc dev_indices");

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_bools");


			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("memcpy dev_idata");

			Common::kernMapToBoolean << <gridsize_EXACT, blocksize >> > (n, dev_bools, dev_idata);

			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

			if (number > n)
			{
				cudaMemset(dev_indices + n, 0, (number - n) * sizeof(int));
				checkCUDAErrorFn("set dev_indices");
			}

			//start ticking
			timer().startGpuTimer();

			for (int i = 0; i < count; i++)
			{
				kernUpSweep << <gridsize, blocksize >> > (number, 1 << i, 1 << i + 1, dev_indices);
#ifdef SYNC_GRID
				cudaThreadSynchronize();
#endif
			}

			//set data[number-1] to 0
			cudaMemset((void*)(dev_indices + (number - 1)), 0, sizeof(int));
			checkCUDAErrorFn("set dev_indices[number-1]");


			for (int i = count - 1; i >= 0; i--)
			{
				kernDownSweep << <gridsize, blocksize >> > (number, 1 << i, 1 << i + 1, dev_indices);
#ifdef SYNC_GRID
				cudaThreadSynchronize();
#endif
			}

			Common::kernScatter << <gridsize_EXACT, blocksize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

			//stop ticking
			timer().endGpuTimer();

			cudaMemcpy(&result, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
			result += (int)(idata[n - 1] != 0);
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			checkCUDAErrorFn("free dev_idata");

			cudaFree(dev_odata);
			checkCUDAErrorFn("free dev_odata");

			cudaFree(dev_indices);
			checkCUDAErrorFn("free dev_indices");

			cudaFree(dev_bools);
			checkCUDAErrorFn("free dev_bools");
			return result;
		}
	}
}