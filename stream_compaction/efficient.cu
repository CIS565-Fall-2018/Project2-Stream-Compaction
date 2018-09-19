#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
#include <string>


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int n, int power, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (index % (2 * power) == 0) {
                    array[index + (2 * power) - 1] = array[index + power - 1] + array[index + (2 * power) - 1];
                }
            }
        }

        __global__ void kernDownsweep(int n, int power, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (index % (2 * power) == 0) {
                    int t = array[index + power - 1];
                    array[index + power - 1] = array[index + (2 * power) - 1];
                    array[index + (2 * power) - 1] = t + array[index + (2 * power) - 1];
                }
            }
        }

        __global__ void kernSetZero(int n, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (index == n - 1) {
                    array[index] = 0;
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *temp;

            int size = 1;
            while (size < n) {
                size *= 2;
            }

            cudaMalloc((void**)&temp, size * sizeof(int));
            cudaDeviceSynchronize();

            cudaMemcpy(temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int blockSize = 256;
            int blocks = (size + blockSize - 1) / blockSize;

            // TODO
            for (int d = 0; d < ilog2ceil(size); ++d) {
                kernUpsweep << <blocks, blockSize >> > (size, pow(2, d), temp);
                cudaDeviceSynchronize();
            }
            
            kernSetZero << <blocks, blockSize >> > (size, temp);

            for (int d = ilog2ceil(size) - 1; d >= 0; --d) {
                kernDownsweep << <blocks, blockSize >> > (size, pow(2, d), temp);
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, temp, n * sizeof(int), cudaMemcpyDeviceToHost);

        }













        void scanEfficient(int n, int *odata, const int *idata) {
             int *temp;

            int size = 1;
            while (size < n) {
                size *= 2;
            }

            cudaMalloc((void**)&temp, size * sizeof(int));
            cudaDeviceSynchronize();

            cudaMemcpy(temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            int blockSize = 256;
            int blocks = (size + blockSize - 1) / blockSize;

            // TODO
            for (int d = 0; d < ilog2ceil(size); ++d) {
                kernUpsweep << <blocks, blockSize >> > (size, pow(2, d), temp);
                cudaDeviceSynchronize();
            }

            kernSetZero << <blocks, blockSize >> > (size, temp);

            for (int d = ilog2ceil(size) - 1; d >= 0; --d) {
                kernDownsweep << <blocks, blockSize >> > (size, pow(2, d), temp);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(odata, temp, n * sizeof(int), cudaMemcpyDeviceToHost);

        }

        __global__ void kernMapToBoolean(int n, int *read, int *write) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (read[index] == 0) {
                    write[index] = 0;
                }
                else {
                    write[index] = 1;
                }
            }
        }

        __global__ void kernScatter(int n, int *idata, int *booleans, int *scan, int *odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (booleans[index] == 1) {
                    odata[scan[index]] = idata[index];
                }
            }
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
            timer().startGpuTimer();

            int blockSize = 256;
            int blocks = (n + blockSize - 1) / blockSize;

            int *dev_idata;
            int *booleans;
            int *scanArray;
            int *result;

            cudaMallocManaged(&dev_idata, n * sizeof(int));

            cudaDeviceSynchronize();
            
            for (int i = 0; i < n; ++i) {
                dev_idata[i] = idata[i];
            }
            
            cudaMallocManaged(&booleans, n * sizeof(int));
            cudaMallocManaged(&scanArray, n * sizeof(int));
            cudaMallocManaged(&result, n * sizeof(int));

            cudaDeviceSynchronize();

            // First map the initial array to booleans
            kernMapToBoolean << <blocks, blockSize >> > (n, dev_idata, booleans);

            cudaDeviceSynchronize();

            // Now do a scan
            scanEfficient(n, scanArray, booleans);

            // Now do a scatter
            int *dev_odata;
            cudaMallocManaged(&dev_odata, n * sizeof(int));
            kernScatter << <blocks, blockSize >> > (n, dev_idata, booleans, scanArray, dev_odata);

            cudaDeviceSynchronize();

            int finalCount = 0;
            for (int i = 0; i < n; ++i) {
                finalCount += booleans[i];
            }

            cudaMemcpy(odata, dev_odata, finalCount * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            return finalCount;
        }
    }
}
