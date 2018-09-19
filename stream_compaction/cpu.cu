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

        void scanHelper(int n, int *odata, const int *idata)
        {
            if (n == 0) return;

            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            
        }

        void scan(int n, int *odata, const int *idata) 
        {
	        timer().startCpuTimer();
            
            scanHelper(n, odata, idata);

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        
            timer().startCpuTimer();

            int oIndex = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[oIndex] = idata[i];
                    oIndex++;
                }
            }

	        timer().endCpuTimer();
            return oIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
	        
            int* bdata = new int[n];
            int* sdata = new int[n];

            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    bdata[i] = 1;
                }
                else
                {
                    bdata[i] = 0;
                }
            }

            // cannot call scan because it uses startCpuTimer as well
            scanHelper(n, sdata, bdata);

            int sum = 0;
            for (int i = 0; i < n; i++)
            {
                if (bdata[i] != 0)
                {
                    odata[sdata[i]] = idata[i];
                    sum = sdata[i];
                }
            }

            timer().endCpuTimer();
            return sum + 1;
        }
    }
}
