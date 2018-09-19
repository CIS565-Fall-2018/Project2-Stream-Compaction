#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernScan(int n, int power, int *read, int *write) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (index >= power) {
                    write[index] = read[index - power] + read[index];
                }
                else {
                    write[index] = read[index];
                }
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            int *dev_read;
            cudaMalloc((void**)&dev_read, n * sizeof(int));

            int *dev_write;
            cudaMalloc((void**)&dev_write, n * sizeof(int));

            cudaMemcpy(dev_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int D = ilog2ceil(n);
            for (int d = 1; d < D + 1; ++d) {
                int blockSize = 256;
                int blocks = (n + blockSize - 1) / blockSize;

                int power = pow(2, d - 1);
                kernScan << <blocks, blockSize >> > (n, power, dev_read, dev_write);

                cudaDeviceSynchronize();

                int *temp = dev_read;
                dev_read = dev_write;
                dev_write = temp;
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_read, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = n - 1; i >= 1; --i) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
           
        }

        
    }
}
