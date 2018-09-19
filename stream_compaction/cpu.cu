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
            // TODO
			// Insert I
			odata[0] = 0;
			// Compute the sums
			for (int i = 1; i != n; ++i)
			{
				odata[i] = idata[i - 1] + odata[i - 1];
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
            // TODO
			int remaining = 0;
			for (int i = 0; i != n; ++i)
			{
				if (idata[i] != 0)
				{
					odata[remaining] = idata[i];
					++remaining;
				}
			}

	        timer().endCpuTimer();
            return remaining;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

			int* hasChanged = new int[n];
			int* sum = new int[n];

			timer().startCpuTimer();
	        // TODO
			
			for (int i = 0; i != n; ++i)
			{
				if (idata[i] == MEETCRITERION)
				{
					hasChanged[i] = HAS_MET;
				}
				else
				{
					hasChanged[i] = NOT_MET;
				}
			}

			sum[0] = 0;
			for (int i = 1; i != n; ++i)
			{
				sum[i] = hasChanged[i - 1] + sum[i - 1];
			}
			int remaining = 0;
			for (int i = 0; i < n; ++i)
			{
				if (hasChanged[i] == NOT_MET)
				{
					odata[sum[i]] = idata[i];
					++remaining;
				}
			}

	        timer().endCpuTimer();

			delete[] hasChanged;
			delete[] sum;

            return remaining;
        }
    }
}
