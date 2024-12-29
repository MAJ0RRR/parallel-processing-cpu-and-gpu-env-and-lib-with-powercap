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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "twinprime_defines.h"

__device__ int is_prime(int n) {
  if (n < 2) return 0;
  for (int i = 2; i * i <= n; i++) {
      if (n % i == 0) return 0;
  }
  return 1;
}

__global__ void appkernel(void *devPtr) {
  int *output = (int *)(((void **)devPtr)[0]); // Wyjście: tablica wyników
  long my_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Sprawdzenie liczb bliźniaczych
  if (is_prime(my_index) && is_prime(my_index + 2)) {
      output[my_index] = 1; // Para bliźniaczych
  } else {
      output[my_index] = 0; // Brak pary
  }
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream) {

  dim3 blocksingrid(batchSize / PATTERNSEARCH_THREADS_IN_BLOCK);
  dim3 threadsinblock(PATTERNSEARCH_THREADS_IN_BLOCK);

  log_message(LOG_DEBUG, "Launichng GPU Kernel with %i blocks in grid and %i threads in block.", batchSize / PATTERNSEARCH_THREADS_IN_BLOCK, PATTERNSEARCH_THREADS_IN_BLOCK);
  appkernel<<<blocksingrid, threadsinblock, 0, stream>>>(devPtr);

  if (cudaSuccess != cudaGetLastError()) {
    log_message(LOG_ERROR, "Error during kernel launch in stream");
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize) { launchkernelinstream(devPtr, batchSize, 0); }
