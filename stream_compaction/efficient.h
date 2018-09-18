#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata , bool istimer);

		void realscan(int n, int *odata, const int *idata, bool istimer);

        int compact(int n, int *odata, const int *idata);
    }
}
