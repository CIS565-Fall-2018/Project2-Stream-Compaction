#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          odata[0] = 0;
          for (int i = 1; i < n; ++i) {
            odata[i] = odata[i - 1] + idata[i - 1];
          }
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          int non_zero_idx = 0;
          for (int i = 0; i < n; ++i) {
            if (idata[i] != 0) {
              odata[non_zero_idx] = idata[i];
              ++non_zero_idx;
            }
          }
	        timer().endCpuTimer();
          return non_zero_idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        
	        // allocate a temporary 0/1 accumulating array
          int* temp = new int[n];

          //for (int i = 0; i < n; ++i) {
          //  temp[i] = 0;
          //}
          temp[0] = 0;
          timer().startCpuTimer();
          // scan to 0/1 accumulating array
          for (int i = 1; i < n; ++i) {
            temp[i] = temp[i - 1] + (idata[i - 1] != 0);
          }
          
          // use temp to map to output
          int count = 0;
          for (int i = 0; i < n; ++i) {
            if (idata[i] != 0) {
              ++count;
              odata[temp[i]] = idata[i];
            }
          }
	        timer().endCpuTimer();
          delete[] temp;
          return count;
        }
    }
}
