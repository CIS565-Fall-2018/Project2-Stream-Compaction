#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		void scan(int n, int *odata, const int *idata) {
			// TODO use `thrust::exclusive_scan`
			// example: for device_vectors dv_in and dv_out:
			// thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			//////////////////////////////////////////////////////////////

			// NOT WORKING, TOO SLOW

			//thrust::device_vector<int> dv_in(thrust::host_vector<int>(idata, idata + n));
			//thrust::device_vector<int> dv_out(thrust::host_vector<int>(odata, odata + n));

            //timer().startGpuTimer();
			//thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            //timer().endGpuTimer();

			//cudaMemcpy(odata, dv_out.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("memcpy back failed!");

			//////////////////////////////////////////////////////////////

			// NOT WORKING, TOO SLOW

			//int *dev_in, *dev_out;

			//cudaMalloc((void**)&dev_in, n * sizeof(int));
			//checkCUDAError("cudaMalloc dev_in failed!");

			//cudaMalloc((void**)&dev_out, n * sizeof(int));
			//checkCUDAError("cudaMalloc dev_out failed!");

			//cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			//thrust::device_ptr<int> dv_in_ptr(dev_in);
			//thrust::device_ptr<int> dv_out_ptr(dev_out);

			//thrust::device_vector<int> dv_in(dev_in, dev_in + n);
			//thrust::device_vector<int> dv_out(dev_out, dev_out + n);

			//timer().startGpuTimer();
			//thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			//timer().endGpuTimer();

			//cudaMemcpy(odata, dv_out.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);

			//cudaFree(dev_in);
			//cudaFree(dev_out);

			//////////////////////////////////////////////////////////////////////////

			// NOT WORKING, TOO SLOW, MUST BE GPU ISSUE

			thrust::device_vector<int> d_data_in(idata, idata + n);
			thrust::device_vector<int> d_data_out(odata, odata + n);
			timer().startGpuTimer();
			thrust::exclusive_scan(d_data_in.begin(), d_data_in.end(), d_data_out.begin());
			timer().endGpuTimer();
			thrust::copy(d_data_out.begin(), d_data_out.end(), odata);
		}
    }
}
