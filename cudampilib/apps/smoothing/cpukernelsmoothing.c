/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// idea: in most cases, input data can be uploaded to GPU's memory and
// consequently we only need to copy a pointer in kernel invocation
// in OpenCL we could hide any kernel invocation

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define ENABLE_LOGGING
#include "logger.h"
#include "smoothing_defines.h"

void appkernel(void *devPtr, unsigned long num_elements, int num_threads) 
{
    double *devPtra = (double *)(((void **)devPtr)[0]);
    double *devPtrc = (double *)(((void **)devPtr)[1]);
    double *devPtrk = (double *)(((void **)devPtr)[2]);

    #pragma omp parallel for num_threads(num_threads)
    for (unsigned long i = 0; i < num_elements; i++) {
        double sum = 0.0;
        for (int k = -GAUSSIAN_KERNEL_RADIUS; k <= GAUSSIAN_KERNEL_RADIUS; k++) {
            long idx = (long)i + k;
            if (idx >= 0 && (unsigned long)idx < num_elements) {
                sum += devPtra[idx] * devPtrk[k + GAUSSIAN_KERNEL_RADIUS];
            }
        }
        devPtrc[i] = sum;
    }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads) 
{
  log_message(LOG_DEBUG, "Launichng CPU Kernel with %llu elements and %i threads.", batchSize, num_threads);
  appkernel(devPtr, batchSize, num_threads);
}