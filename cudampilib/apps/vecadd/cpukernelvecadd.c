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

#include <stdio.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "vecadd_defines.h"

void appkernel(void *devPtr, int num_elements, int num_threads) 
{
    double *devPtra = (double *)(((void **)devPtr)[0]);
    double *devPtrb = (double *)(((void **)devPtr)[1]);
    double *devPtrc = (double *)(((void **)devPtr)[2]);

    #pragma omp parallel for num_threads(num_threads)
    {
        for (long my_index = 0; my_index < num_elements; my_index++) 
        {
            devPtrc[my_index] = devPtra[my_index] / 2 + devPtrb[my_index] / 3;
        }
    }
}

extern void launchcpukernel(void *devPtr, int num_threads)
{
    int num_elements = VECADD_BATCH_SIZE;
    log_message(LOG_DEBUG, "Launichng CPU Kernel with %i elements and %i threads.", num_elements, num_threads);
    appkernel(devPtr, num_elements, num_threads);
}
