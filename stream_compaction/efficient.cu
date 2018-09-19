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

        const int BLOCK_SIZE = 256;
        
        __global__ void kernUpSweep(int n, int stride, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            int fullIndex = (index + 1) * stride - 1;
            int halfIndex = index * stride - 1 + (stride / 2);
            data[fullIndex] += data[halfIndex];
        }

        __global__ void kernDownSweep(int n, int stride, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            int fullIndex = (index + 1) * stride - 1;
            int halfIndex = index * stride - 1 + (stride / 2);

            int temp = data[halfIndex];
            data[halfIndex] = data[fullIndex];
            data[fullIndex] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int *dev_data;
            int width = 1;
            int squareN = pow(2, ilog2ceil(n));
            int iterations = ilog2(squareN) - 1;
            int numThreads, numBlocks;

            cudaMalloc((void**)&dev_data, squareN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            // up sweep
            for (int i = 0; i <= iterations; i++)
            {
                width = width * 2;
                numThreads = squareN / width;
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweep<<<numBlocks, BLOCK_SIZE>>>(numThreads, width, dev_data);
                checkCUDAErrorFn("kernUpSweep failed!");
            }

            int zero = 0;
            cudaMemcpy(&dev_data[squareN - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

            // down sweep
            width = pow(2, iterations + 2);
            for (int i = iterations; i >= 0; i--)
            {
                width = width / 2;
                numThreads = squareN / width;
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep<<<numBlocks, BLOCK_SIZE>>>(numThreads, width, dev_data);
                checkCUDAErrorFn("kernDownSweep failed!");
            }
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scanNoTimer(int n, int *odata, const int *idata)
        {
            int *dev_data;
            int width = 1;
            int squareN = pow(2, ilog2ceil(n));
            int iterations = ilog2(squareN) - 1;
            int numThreads, numBlocks;

            cudaMalloc((void**)&dev_data, squareN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // up sweep
            for (int i = 0; i <= iterations; i++)
            {
                width = width * 2;
                numThreads = squareN / width;
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, width, dev_data);
                checkCUDAErrorFn("kernUpSweep failed!");
            }

            int zero = 0;
            cudaMemcpy(&dev_data[squareN - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

            // down sweep
            width = pow(2, iterations + 2);
            for (int i = iterations; i >= 0; i--)
            {
                width = width / 2;
                numThreads = squareN / width;
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, width, dev_data);
                checkCUDAErrorFn("kernDownSweep failed!");
            }

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
        int compact(int n, int *odata, const int *idata) 
        {
            int count;
            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            // copy idata to dev_idata
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            // map idata to booleans
            Common::kernMapToBoolean<<<numBlocks, BLOCK_SIZE>>>(n, dev_bools, dev_idata);
            // copy booleans to odata
            cudaMemcpy(odata, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

            count = odata[n - 1];
            scanNoTimer(n, odata, odata);
            count += odata[n - 1];

            cudaMemcpy(dev_indices, odata, n * sizeof(int), cudaMemcpyHostToDevice);
            Common::kernScatter<<<numBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return count;
        }
    }
}
