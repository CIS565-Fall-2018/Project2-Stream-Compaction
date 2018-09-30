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
			// Serial Scan: Exclusive
			int acc = 0;
			for (int i = 0; i < n; i++)
			{
				odata[i] = acc;
				acc = acc + idata[i];			
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
			int flag = 0;
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] != 0)
				{
					odata[flag] = idata[i];
					flag++;
				}
				
			}

	        timer().endCpuTimer();
            return flag;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			// predicate
			int acc = 0;
			int* tempdata = new int[n];
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] == 0)
				{
					tempdata[i] = 0;

				}
				else
				{
					tempdata[i] = 1;
					acc++;
				}
			}
			// ======= tempdata[] = 1110011111001111
			// scan sum
			int accSum = 0;
			int* tempData2 = new int[n];
			for (int i = 0; i < n; i++)
			{
				tempData2[i] = accSum;
				accSum += tempdata[i];

			}

			// idata[] = 3120043215002222
			// ======= tempdata[] = 1110011111001111
			// tempData2[] = 0123334567899910....
			// scatter
			for (int i = 0; i < n; ++i)
			{
				odata[tempData2[i]] = idata[i];
			}

			delete[] tempdata;
			delete[] tempData2;
	        timer().endCpuTimer();
            return acc;
        }
    }
}
