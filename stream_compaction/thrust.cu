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
            timer().startGpuTimer();

			//thrust::host_vector<int> host_thrust_in = idata;
			//thrust::host_vector<int> host_thrust_out = odata;

			//thrust::host_vector<int> host_thrust_in(n);
			
			//thrust::device_vector<int> dev_thrust_in = host_thrust_in;
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			thrust::exclusive_scan(idata, idata + n, odata);
            timer().endGpuTimer();
        }
    }
}
